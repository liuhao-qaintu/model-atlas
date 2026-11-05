"""本机重解析产生新版本；确认入库才归档旧版，保留历史引用。"""
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from sqlalchemy.orm import Session

from backend.infra.database import KnowledgeAnalysis, KnowledgeDocument, engine

PROCESSING_POLICY = 'layout-paragraph-v3-20261104'


class VersionConflict(ValueError):
    pass


def active_document(identity):
    with Session(engine) as db:
        for _ in range(32):
            row = db.get(KnowledgeDocument, identity)
            if not row:
                return None
            document = dict(row.payload)
            if document.get('status') != 'archived' or not document.get('superseded_by'):
                return document
            identity = document['superseded_by']
    raise VersionConflict('版本链异常，请检查文档记录')


def preview_reprocess(identity):
    from backend.infra.ingestion import UPLOAD_ROOT, parse_document
    with Session(engine) as db:
        row = db.get(KnowledgeDocument, identity)
        if not row:
            raise LookupError('文档不存在')
        previous = dict(row.payload)
    if previous.get('status') != 'imported':
        raise VersionConflict('请对当前已入库版本重新处理')
    suffix = previous.get('format')
    if suffix not in {'pdf', 'md', 'txt'}:
        raise ValueError('当前格式无法重新处理')
    root = UPLOAD_ROOT.resolve()
    original = (root / f"{identity}.{suffix}").resolve()
    if original.parent != root or not original.is_file():
        raise ValueError('找不到保存的原始文件，无法重新解析')
    raw = original.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if previous.get('sha256') and digest != previous['sha256']:
        raise ValueError('原始文件与入库指纹不一致，请先核对文件')
    pages = ','.join(map(str, previous.get('selected_pages') or []))
    parsed = parse_document(previous['filename'], raw, pages)
    new_id = uuid4().hex
    now = datetime.now(timezone.utc).isoformat()
    document = {**parsed, 'id': new_id, 'filename': previous['filename'], 'model_id': previous['model_id'],
                'title': previous['title'], 'source_url': previous.get('source_url', ''), 'status': 'preview',
                'cloud_allowed': previous.get('cloud_allowed', False), 'sha256': digest, 'created_at': now,
                'owner_user_id': previous.get('owner_user_id', 'local-user'),
                'revision_of': identity, 'revision_number': previous.get('revision_number', 1) + 1,
                'lineage_id': previous.get('lineage_id', identity),
                'provenance_document_id': previous.get('provenance_document_id', identity),
                'retrieval_added_at': previous.get('retrieval_added_at') or previous.get('imported_at') or previous.get('created_at'),
                'retrieval_priority': previous.get('retrieval_priority', 1.0), 'processing_policy': PROCESSING_POLICY,
                'previous_part_count': len(previous.get('parts', []))}
    (root / f'{new_id}.{suffix}').write_bytes(raw)
    if parsed.get('markdown') and suffix != 'md':
        (root / f'{new_id}.md').write_text(parsed['markdown'], encoding='utf-8')
    with Session(engine) as db:
        db.add(KnowledgeDocument(id=new_id, payload=document))
        db.commit()
    return document


def activate_revision(db, document, timestamp):
    """与新section写入共用同一个事务；运行中的归类任务不可被切换版本。"""
    if not document.get('revision_of'):
        return
    from backend.infra import ingest_job
    identity = document['revision_of']
    previous = db.get(KnowledgeDocument, identity, with_for_update=True)
    if not previous or previous.payload.get('status') != 'imported':
        raise VersionConflict('原版本已变化，请重新预览，避免覆盖另一次更新')
    if any(j['status'] in {'pending', 'running'} for j in ingest_job.list_jobs(identity)):
        raise VersionConflict('原版本正在归类，请等待任务结束后更新')
    old = dict(previous.payload)
    document['retrieval_priority'] = old.get('retrieval_priority', 1.0)
    document['retrieval_added_at'] = old.get('retrieval_added_at') or old.get('imported_at') or old.get('created_at')
    previous.payload = {**old, 'status': 'archived', 'superseded_by': document['id'], 'archived_at': timestamp}
    for item in db.query(KnowledgeAnalysis).all():
        if item.payload.get('type') == 'review_item' and item.payload.get('document_id') == identity and item.payload.get('status') == 'pending':
            item.payload = {**item.payload, 'status': 'superseded', 'superseded_by': document['id']}
