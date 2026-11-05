"""审核队列：读取、通过、拒绝、编辑。

审核项存 knowledge_analyses(type='review_item')。通过时：
- 把分类字段写回 knowledge_sections；
- 只把 link_existing 的实体写成 knowledge_relations（写图谱）；
- 新节点/ambiguous 只保留在 section payload，不自动建模型节点。
"""
from __future__ import annotations

import asyncio
import re
from datetime import datetime, timezone
from uuid import uuid4

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.infra.database import KnowledgeAnalysis, KnowledgeModel, KnowledgeRelation, KnowledgeSection, records, upsert_records
from backend.rag.analysis import mentions

router = APIRouter(prefix='/api/review', tags=['知识审核'])


class EditInput(BaseModel):
    proposed: dict = Field(default_factory=dict)


def _items() -> list[dict]:
    return [item for item in records(KnowledgeAnalysis) if item.get('type') == 'review_item']


def _get(item_id: str) -> dict:
    item = next((row for row in _items() if row.get('id') == item_id), None)
    if item is None:
        raise HTTPException(404, '审核项不存在')
    return item


def _save(item: dict) -> None:
    upsert_records(KnowledgeAnalysis, [item])


def _section(section_id: str) -> dict:
    section = next((row for row in records(KnowledgeSection) if row.get('id') == section_id), None)
    if section is None:
        raise HTTPException(404, '审核项对应的章节不存在')
    return section


def _models() -> dict[str, dict]:
    return {model['id']: model for model in records(KnowledgeModel) if model.get('id')}


def _entity_quote(section: dict, entity: dict, target: dict) -> str:
    """引文必须原样存在且明确提到目标；无手动引文时定位别名附近的原句。"""
    content = section.get('content') or ''
    aliases = list(dict.fromkeys([target.get('name', ''), *(target.get('aliases') or [])]))
    target_info = {'aliases': [alias for alias in aliases if alias.strip()]}
    quote = entity.get('evidence_quote')
    if quote is None:
        for alias in sorted(target_info['aliases'], key=len, reverse=True):
            pattern = (r'(?<![A-Za-z0-9])' + re.escape(alias) + r'(?![A-Za-z0-9])') if alias.isascii() else re.escape(alias)
            found = re.search(pattern, content, re.I)
            if found:
                quote = content[max(0, found.start() - 80):min(len(content), found.end() + 180)]
                break
    if not isinstance(quote, str) or not 15 <= len(quote) <= 700 or quote not in content or not mentions(quote, target_info):
        raise HTTPException(422, '关系引文必须来自当前章节，且包含目标名称或别名；请修改实体链接或引文')
    if not section.get('source_ids'):
        raise HTTPException(422, '章节缺少来源，不能写入关系')
    return quote


def _existing_relation(source: str, target: str) -> dict | None:
    """同一 source/target 只保留一条用户关系；不同章节追加 evidence。"""
    for relation in records(KnowledgeRelation):
        if relation.get('source') == source and relation.get('target') == target and relation.get('origin') == 'user':
            return relation
    return None


@router.get('/items')
def list_items(status: str | None = None, document_id: str | None = None):
    rows = _items()
    if status:
        rows = [row for row in rows if row.get('status') == status]
    if document_id:
        rows = [row for row in rows if row.get('document_id') == document_id]
    return sorted(rows, key=lambda row: row.get('created_at', ''), reverse=True)


@router.post('/items/{item_id}/approve')
def approve(item_id: str):
    item = _get(item_id)
    if item.get('status') != 'pending':
        raise HTTPException(409, '该审核项不是 pending 状态')
    section = _section(item.get('target_id'))
    if section.get('document_id'):
        from backend.infra.database import KnowledgeDocument
        document = next((d for d in records(KnowledgeDocument) if d.get('id') == section['document_id']), None)
        if document and document.get('status') == 'archived':
            raise HTTPException(409, '该审核项属于历史版本，请审核当前版本')
    proposed = item.get('proposed') or {}
    classification = proposed.get('classification') or {}
    links = proposed.get('entity_links') or []
    if classification.get('skipped'):
        raise HTTPException(422, '分类已跳过，请先依据原文编辑分类，不可直接通过占位结果')
    models = _models()
    verified = []
    for entity in links:
        link = entity.get('link') or {}
        target = link.get('model_id')
        if link.get('decision') != 'link_existing':
            continue
        if target not in models or target == section.get('model_id'):
            raise HTTPException(422, '实体链接目标不存在或指向文档自身，请先修改')
        verified.append((entity, target, _entity_quote(section, entity, models[target])))
    now = datetime.now(timezone.utc).isoformat()

    updated = {
        **section,
        'domain_path': classification.get('domain_path'),
        'section_role': classification.get('section_role'),
        'knowledge_type': classification.get('knowledge_type'),
        'entities': classification.get('entities') or [],
        'classification_confidence': classification.get('confidence'),
        'entity_links': links,
        'review_status': 'approved',
        'reviewed_at': now,
        'review_origin': proposed.get('review_origin', 'user'),
        'review_note': proposed.get('review_note', ''),
    }
    upsert_records(KnowledgeSection, [updated])

    for entity, target, quote in verified:
        existing = _existing_relation(section.get('model_id'), target)
        if existing:
            evidence = list(existing.get('evidence') or [])
            if not any(item.get('section_id') == section.get('id') for item in evidence):
                evidence.append({'section_id': section.get('id'), 'quote': quote})
            existing['evidence'] = evidence
            source_ids = list(existing.get('source_ids') or [])
            for sid in section.get('source_ids') or []:
                if sid not in source_ids:
                    source_ids.append(sid)
            existing['source_ids'] = source_ids
            existing['owner_model_id'] = existing.get('owner_model_id') or section.get('model_id')
            existing['source_name'] = models[section['model_id']]['name']
            existing['target_name'] = models[target]['name']
            existing['review_status'] = 'approved'
            existing['updated_at'] = now
            upsert_records(KnowledgeRelation, [existing])
            continue
        relation = {
            'id': f"rel-{uuid4().hex[:24]}",
            'source': section.get('model_id'),
            'target': target,
            'source_name': models[section['model_id']]['name'],
            'target_name': models[target]['name'],
            'owner_model_id': section.get('model_id'),
            'relation': '提及',
            'kind': '用户关系',
            'origin': 'user',
            'status': 'accepted',
            'review_status': 'approved',
            'source_ids': section.get('source_ids') or [],
            'evidence': [{'section_id': section.get('id'), 'quote': quote}],
            'locator': section.get('locator', ''),
            'document_id': section.get('document_id'),
            'entity': entity.get('entity'),
        }
        upsert_records(KnowledgeRelation, [relation])

    item['status'] = 'approved'
    item['reviewed_at'] = now
    item['review_origin'] = proposed.get('review_origin', 'user')
    _save(item)
    return item


@router.post('/items/{item_id}/reject')
def reject(item_id: str):
    item = _get(item_id)
    if item.get('status') != 'pending':
        raise HTTPException(409, '该审核项不是 pending 状态')
    item['status'] = 'rejected'
    item['reviewed_at'] = datetime.now(timezone.utc).isoformat()
    _save(item)
    return item


@router.post('/items/{item_id}/edit')
def edit(item_id: str, value: EditInput):
    item = _get(item_id)
    if item.get('status') != 'pending':
        raise HTTPException(409, '该审核项不是 pending 状态')
    item['proposed'] = value.proposed
    _save(item)
    return item


@router.post('/refresh')
async def refresh_retriever():
    """审核写库后刷新内存 retriever；/api/graph 读的是这份内存数据。"""
    try:
        import backend.main as main
        from backend.rag.retrieval import Retriever
        async with main.knowledge_lock:
            main.replace_retriever(await asyncio.to_thread(Retriever))
        return {'ok': True}
    except Exception as exc:
        raise HTTPException(500, f'刷新检索器失败: {type(exc).__name__}')
