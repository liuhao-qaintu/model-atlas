"""文档阅读报告任务：后台读正文，前台确认文档标签，不生成逐片段审核卡。"""
from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from backend.infra import ingest_job
from backend.infra.database import KnowledgeAnalysis, KnowledgeDocument, KnowledgeModel, KnowledgeSection, records, upsert_records

router = APIRouter(prefix='/api/ingest', tags=['知识入库'])
_tasks: dict[str, asyncio.Task] = {}


def _sections(document_id: str) -> list[dict]:
    rows = [r for r in records(KnowledgeSection) if r.get('document_id') == document_id]
    return sorted(rows, key=lambda item: (item.get('page') or 0, item.get('part') or 0))


def _linkable_models() -> list[dict]:
    return [
        model for model in records(KnowledgeModel)
        if not str(model.get('id', '')).startswith('user-doc-') and model.get('kind') not in {'文档', 'document', '知识标签'}
    ]


async def _run(job, document: dict, sections: list[dict], models: list[dict]) -> None:
    from backend.rag.document_report import analyze_document
    from backend.infra.document_reports import retire_placeholders
    try:
        job.status = 'running'
        await ingest_job.publish(job, {
            'type': 'job_started', 'job_id': job.id, 'document_id': document.get('id'), 'total': 0,
        })
        await asyncio.to_thread(retire_placeholders, document['id'])
        cloud_allowed = document.get('cloud_allowed', False) and all(section.get('cloud_allowed', False) for section in sections)
        if not cloud_allowed:
            await ingest_job.finish(job, 'skipped', '这份资料仅在本地处理，尚未调用大模型；可阅读原文件，允许云端处理后再生成分析报告。')
            return
        completed = 0
        async def progress(index, total):
            nonlocal completed
            if job.cancelled:
                raise asyncio.CancelledError()
            if index:
                completed += 1
            await ingest_job.publish(job, {'type': 'progress', 'stage': '阅读正文', 'progress': completed, 'total': total + 1})
        report = await analyze_document(document, sections, models, progress)
        if job.cancelled:
            await ingest_job.finish(job, 'cancelled')
        else:
            from backend.infra.ingestion import document_record
            current = await asyncio.to_thread(document_record, document['id'])
            if not current or current.get('status') != 'imported' or not current.get('cloud_allowed'):
                raise ValueError('资料版本或云端许可已变化，报告未保存为当前结果')
            await asyncio.to_thread(upsert_records, KnowledgeAnalysis, [report])
            job.stats.update(report['coverage'])
            await ingest_job.publish(job, {'type': 'report_ready', 'report_id': report['id']})
            await ingest_job.publish(job, {'type': 'progress', 'progress': completed + 1, 'total': completed + 1, 'stage': '报告已生成'})
            await ingest_job.finish(job, 'done')
    except asyncio.CancelledError:
        await ingest_job.finish(job, 'cancelled')
        raise
    except Exception as exc:
        message = '模型响应超时，报告未完成；可稍后重试或缩小分析页码。' if isinstance(exc, (TimeoutError, asyncio.TimeoutError)) else f'分析未完成（{type(exc).__name__}）：{str(exc)[:160]}'
        await ingest_job.finish(job, 'failed', message)


@router.post('/documents/{document_id}')
async def start_ingest(document_id: str):
    documents = [item for item in records(KnowledgeDocument) if item.get('id') == document_id]
    if not documents:
        raise HTTPException(404, '文档不存在')
    document = documents[0]
    if document.get('status') != 'imported':
        raise HTTPException(409, '只能对当前已入库版本归类')
    running = [job for job in ingest_job.list_jobs(document_id) if job.get('status') in {'pending', 'running'}]
    if running:
        raise HTTPException(409, '该文档已有归类任务在运行')
    sections = _sections(document_id)
    if not sections:
        raise HTTPException(422, '文档没有可分类的片段')
    models = _linkable_models()
    from backend.infra.access import current_actor
    actor = current_actor.get()
    job = ingest_job.create_job(document_id, actor.id if actor is not None else document.get('owner_user_id', 'local-user'))
    task = asyncio.create_task(_run(job, document, sections, models))
    _tasks[job.id] = task
    task.add_done_callback(lambda _: _tasks.pop(job.id, None))
    return job.snapshot()


@router.get('/documents/{document_id}/report')
def get_document_report(document_id: str):
    from backend.infra.document_reports import latest_report
    rows = [d for d in records(KnowledgeDocument) if d['id'] == document_id]
    if not rows:
        raise HTTPException(404, '文档不存在')
    jobs = [j for j in records(KnowledgeAnalysis) if j.get('type') == 'ingestion_job' and j.get('document_id') == document_id]
    return {'report': latest_report(document_id), 'job': max(jobs, key=lambda j: j.get('created_at', '')) if jobs else None}


class ReportConfirmation(BaseModel):
    tag_ids: list[str] = Field(default_factory=list, max_length=24)


@router.post('/reports/{report_id}/approve')
async def approve_document_report(report_id: str, value: ReportConfirmation):
    from backend.infra.document_reports import confirm_report
    from backend import main
    from backend.rag.retrieval import Retriever
    try:
        async with main.knowledge_lock:
            result = await asyncio.to_thread(confirm_report, report_id, value.tag_ids)
            main.replace_retriever(await asyncio.to_thread(Retriever))
        return result
    except LookupError as exc:
        raise HTTPException(404, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@router.get('/jobs/{job_id}')
def job_status(job_id: str):
    job = ingest_job.get_job(job_id)
    if job is None:
        raise HTTPException(404, '任务不存在')
    return job.snapshot()


@router.get('/jobs/{job_id}/events')
async def job_events(job_id: str):
    job = ingest_job.get_job(job_id)
    if job is None:
        raise HTTPException(404, '任务不存在')

    async def event_gen():
        async for event in ingest_job.events(job):
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_gen(),
        media_type='text/event-stream',
        headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'},
    )


@router.post('/jobs/{job_id}/cancel')
async def cancel_ingest(job_id: str):
    job = ingest_job.get_job(job_id)
    if job is None:
        raise HTTPException(404, '任务不存在')
    await ingest_job.cancel(job)
    task = _tasks.get(job_id)
    if task:
        task.cancel()
    return job.snapshot()
