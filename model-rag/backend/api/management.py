"""动态模型和关系管理；用户关系必须指向现有模型与原样正文依据。"""
from uuid import uuid4

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from backend.infra.database import KnowledgeModel, KnowledgeRelation, engine, records
from backend.infra.access import current_actor
from backend.rag.analysis import catalog, validate_candidate, VERBS

router = APIRouter(prefix='/api', tags=['模型与关系管理'])


class ModelInput(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    category: str = Field(default='自定义模型', min_length=1, max_length=40)
    description: str = Field(default='', max_length=500)
    aliases: list[str] = Field(default_factory=list, max_length=12)
    tags: list[str] = Field(default_factory=list, max_length=8)


def model_payload(value, identity, order=1000):
    name = value.name.strip()
    if not name: raise ValueError('模型名称不能为空')
    aliases = list(dict.fromkeys([name, *[a.strip() for a in value.aliases if a.strip()]]))
    if any(len(a) > 80 for a in aliases) or any(len(tag) > 30 for tag in value.tags): raise ValueError('别名或标签过长')
    return {'id': identity, 'name': name, 'category': value.category.strip() or '自定义模型',
            'kind': '文档' if value.category.strip() == '文档' else '用户模型',
            'description': value.description.strip(), 'aliases': aliases, 'tags': list(dict.fromkeys(tag.strip() for tag in value.tags if tag.strip())),
            'origin': 'user', 'order': order, 'ready': False, 'section_count': 0, 'source_ids': []}


def store_model(value, identity=None):
    with Session(engine) as db:
        row = db.get(KnowledgeModel, identity) if identity else None
        if identity and not row: raise LookupError('模型不存在')
        if row and row.payload.get('origin') != 'user': raise ValueError('预置模型由资料维护；此处只修改自定义模型')
        identity = identity or 'user-' + uuid4().hex[:24]
        if row and row.payload['name'] != value.name.strip():
            value = value.model_copy(update={'aliases': list(dict.fromkeys([row.payload['name'], *value.aliases]))})
        payload = model_payload(value, identity, row.payload['order'] if row else 1000)
        existing = db.query(KnowledgeModel).all()
        if any(item.id != identity and item.payload['name'].casefold() == payload['name'].casefold() for item in existing):
            raise ValueError('已有同名模型，请使用已有模型或补充版本名称')
        db.merge(KnowledgeModel(id=identity, payload=payload)); db.commit()
        return payload


class RelationInput(BaseModel):
    source: str = Field(max_length=80)
    target: str = Field(max_length=80)
    relation: str = Field(max_length=30)
    section_id: str = Field(max_length=120)
    quote: str = Field(min_length=15, max_length=700)


def store_relation(value, retriever):
    if value.source not in retriever.models or value.target not in retriever.models: raise ValueError('请选择两个已有模型')
    candidate = validate_candidate(value.model_dump(), {s['id']: s for s in retriever.sections}, value.source, catalog(retriever))
    if not candidate: raise ValueError('摘录必须来自所选模型的正文，包含目标模型名称或别名，并选择有效关系类型')
    section = next(s for s in retriever.sections if s['id'] == candidate['section_id'])
    if not section['source_ids']: raise ValueError('正文没有可追溯来源，不能创建知识关系')
    actor = current_actor.get()
    identity = 'user-' + (actor.id + '-' if actor is not None else '') + candidate['id']
    payload = {'id': identity, 'source': value.source, 'target': value.target,
               'source_name': retriever.models[value.source]['name'], 'target_name': retriever.models[value.target]['name'],
               'relation': value.relation, 'owner_model_id': value.source, 'kind': '用户关系', 'origin': 'user', 'status': 'accepted',
               'locator': candidate['locator'], 'scope': '用户依据原文创建并确认；正文可定位，关系语义仍可复核。',
               'source_ids': section['source_ids'], 'evidence': [{'section_id': section['id'], 'quote': candidate['quote']}]}
    with Session(engine) as db:
        db.merge(KnowledgeRelation(id=identity, payload=payload)); db.commit()
    return payload


async def mutate(operation):
    from backend import main
    from backend.rag.retrieval import Retriever
    from starlette.concurrency import run_in_threadpool
    try:
        async with main.knowledge_lock:
            result = await run_in_threadpool(operation)
            main.replace_retriever(await run_in_threadpool(Retriever))
            return result
    except LookupError as exc: raise HTTPException(404, str(exc)) from exc
    except ValueError as exc: raise HTTPException(422, str(exc)) from exc


@router.post('/models')
async def create_model(value: ModelInput):
    return await mutate(lambda: store_model(value))


@router.patch('/models/{identity}')
async def edit_model(identity: str, value: ModelInput):
    return await mutate(lambda: store_model(value, identity))


@router.get('/management')
def management_options():
    from backend.main import retriever
    return {'models': list(retriever.models.values()), 'relations': [r for r in records(KnowledgeRelation) if r.get('origin') == 'user'], 'verbs': sorted(VERBS)}


@router.post('/relations')
async def create_relation(value: RelationInput):
    from backend import main
    return await mutate(lambda: store_relation(value, main.retriever))


class RelationStatusInput(BaseModel):
    status: str = Field(pattern='^(accepted|rejected)$')


@router.patch('/relations/{identity}')
async def change_relation_status(identity: str, value: RelationStatusInput):
    def update():
        with Session(engine) as db:
            row = db.get(KnowledgeRelation, identity)
            if not row: raise LookupError('关系不存在')
            if row.payload.get('origin') != 'user': raise ValueError('此处只管理用户建立的关系')
            row.payload = {**row.payload, 'status': value.status}; db.commit()
            return row.payload
    return await mutate(update)
