"""文档硬删除：级联清理文档、章节、来源、关系、审核项和文件，并刷新检索器。

单用户 MVP 用文档 id 作为二次确认，不做账号权限；开源后再加请求头 + 角色。
"""
from __future__ import annotations

import asyncio
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from backend.infra import config
from backend.infra.database import (
    KnowledgeAnalysis,
    KnowledgeDocument,
    KnowledgeModel,
    KnowledgeRelation,
    KnowledgeSection,
    KnowledgeSource,
    engine,
)

router = APIRouter(prefix='/api/documents', tags=['文档管理'])


class PriorityInput(BaseModel):
    priority: float = Field(ge=1.0, le=1.25)


class CloudConsentInput(BaseModel):
    cloud_allowed: bool


@router.patch('/{document_id}/cloud-consent')
async def set_cloud_consent(document_id: str, value: CloudConsentInput):
    from backend import main
    from backend.rag.retrieval import Retriever
    async with main.knowledge_lock:
        with Session(engine) as db:
            row = db.get(KnowledgeDocument, document_id, with_for_update=True)
            if not row:
                raise HTTPException(404, '文档不存在')
            if row.payload.get('status') != 'imported':
                raise HTTPException(409, '只能修改当前文档版本的处理许可')
            row.payload = {**row.payload, 'cloud_allowed': value.cloud_allowed}
            for section in db.query(KnowledgeSection).all():
                if section.payload.get('document_id') == document_id:
                    section.payload = {**section.payload, 'cloud_allowed': value.cloud_allowed}
            db.commit()
        main.replace_retriever(await asyncio.to_thread(Retriever))
    return {'id': document_id, 'cloud_allowed': value.cloud_allowed}


@router.post('/{document_id}/reprocess/preview')
async def reprocess_preview(document_id: str):
    from backend import main
    from backend.infra.document_versions import preview_reprocess, VersionConflict
    try:
        async with main.knowledge_lock:
            return await asyncio.to_thread(preview_reprocess, document_id)
    except VersionConflict as exc:
        raise HTTPException(409, str(exc)) from exc
    except LookupError as exc:
        raise HTTPException(404, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@router.patch('/{document_id}/priority')
async def set_priority(document_id: str, value: PriorityInput):
    from backend import main
    from backend.rag.retrieval import Retriever
    async with main.knowledge_lock:
        with Session(engine) as db:
            row = db.get(KnowledgeDocument, document_id, with_for_update=True)
            if not row:
                raise HTTPException(404, '文档不存在')
            if row.payload.get('status') != 'imported':
                raise HTTPException(409, '只能调整当前已入库版本的优先级')
            row.payload = {**row.payload, 'retrieval_priority': value.priority}
            db.commit()
        main.replace_retriever(await asyncio.to_thread(Retriever))
    return {'id': document_id, 'retrieval_priority': value.priority}


class DeleteInput(BaseModel):
    confirm: str


@router.delete('/{document_id}')
async def delete_document(document_id: str, value: DeleteInput):
    if value.confirm != document_id:
        raise HTTPException(400, '确认信息不匹配')
    counts = {'sections': 0, 'relations': 0, 'analyses': 0, 'sources': 0, 'files': 0, 'model': 0}
    with Session(engine) as db:
        document_row = db.get(KnowledgeDocument, document_id)
        if document_row is None:
            raise HTTPException(404, '文档不存在')
        document = dict(document_row.payload)
        model_id = document.get('model_id')

        section_rows = db.query(KnowledgeSection).all()
        section_ids = {
            row.payload.get('id') for row in section_rows
            if row.payload.get('document_id') == document_id
        }

        for row in db.query(KnowledgeRelation).all():
            payload = row.payload
            if payload.get('document_id') == document_id or any(
                item.get('section_id') in section_ids for item in payload.get('evidence') or []
            ):
                db.delete(row)
                counts['relations'] += 1

        for row in db.query(KnowledgeAnalysis).all():
            payload = row.payload
            if payload.get('document_id') == document_id or payload.get('target_id') in section_ids:
                db.delete(row)
                counts['analyses'] += 1

        for row in section_rows:
            if row.payload.get('document_id') == document_id:
                db.delete(row)
                counts['sections'] += 1

        source = db.get(KnowledgeSource, f'D{document_id}')
        if source is not None:
            db.delete(source)
            counts['sources'] += 1

        db.delete(document_row)
        db.flush()

        remaining = db.query(KnowledgeDocument).all()
        still_used = any(row.payload.get('model_id') == model_id for row in remaining)
        model_row = db.get(KnowledgeModel, model_id) if model_id else None
        if model_row is not None and not still_used and model_row.payload.get('kind') == '文档':
            db.delete(model_row)
            counts['model'] = 1

        db.commit()

    upload_root = Path(config.DATA_ROOT) / 'uploads'
    for suffix in {str(document.get('format') or 'pdf'), 'md'}:
        path = upload_root / f'{document_id}.{suffix}'
        if path.exists():
            path.unlink()
            counts['files'] += 1

    try:
        import backend.main as main
        from backend.rag.retrieval import Retriever
        async with main.knowledge_lock:
            main.replace_retriever(await asyncio.to_thread(Retriever))
    except Exception:
        pass

    return {'deleted': document_id, 'title': document.get('title'), **counts}
