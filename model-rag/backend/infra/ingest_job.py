"""上传归类任务的进程内注册表：事件队列 + 状态落 knowledge_analyses。

任务按登录账号隔离；运行队列位于单个进程内，进程重启后丢失；状态另存一份到数据库，
供页面刷新后看到最近一次 job 的结果。所有 LLM/DB 工作由调用方（api/ingest.py）
驱动，本模块只负责 job 生命周期和事件分发。
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, AsyncIterator
from uuid import uuid4

from backend.infra.database import KnowledgeAnalysis, upsert_records
from backend.infra.access import current_actor

TERMINAL_STATUS = {"done", "failed", "cancelled", "skipped"}
PERSIST_EVENTS = {"job_started", "section_classified", "needs_review", "report_ready", "progress", "job_done", "job_failed", "job_cancelled", "job_skipped"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class IngestJob:
    id: str
    document_id: str
    owner_user_id: str
    status: str = "pending"
    progress: int = 0
    total: int = 0
    stats: dict[str, int] = field(default_factory=dict)
    error: str | None = None
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)
    queue: asyncio.Queue = field(default_factory=lambda: asyncio.Queue(maxsize=256))
    cancelled: bool = False

    def payload(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": "ingestion_job",
            "pipeline": "document-report-v1",
            "document_id": self.document_id,
            "owner_user_id": self.owner_user_id,
            "status": self.status,
            "progress": self.progress,
            "total": self.total,
            "stats": dict(self.stats),
            "error": self.error,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    def snapshot(self) -> dict[str, Any]:
        return {k: v for k, v in self.payload().items() if k != "type"}


_jobs: dict[str, IngestJob] = {}


def create_job(document_id: str, owner_user_id: str, total: int = 0) -> IngestJob:
    job = IngestJob(id=f"ing-{uuid4().hex[:24]}", document_id=document_id, owner_user_id=owner_user_id, total=total)
    _jobs[job.id] = job
    return job


def get_job(job_id: str) -> IngestJob | None:
    job = _jobs.get(job_id)
    actor = current_actor.get()
    return job if job and (actor is None or job.owner_user_id == actor.id) else None


def list_jobs(document_id: str | None = None) -> list[dict[str, Any]]:
    jobs = _jobs.values()
    actor = current_actor.get()
    if actor is not None:
        jobs = [job for job in jobs if job.owner_user_id == actor.id]
    if document_id:
        jobs = [job for job in jobs if job.document_id == document_id]
    return [job.snapshot() for job in sorted(jobs, key=lambda item: item.created_at, reverse=True)]


def persist_job(job: IngestJob) -> None:
    """把 job 状态写进 knowledge_analyses；失败不影响主流程。"""
    try:
        upsert_records(KnowledgeAnalysis, [job.payload()])
    except Exception:
        # 事件流优先；状态落库失败不打断归类。
        pass


async def publish(job: IngestJob, event: dict[str, Any]) -> None:
    """把事件放进 job 队列，必要时同步状态到数据库。"""
    job.updated_at = _now()
    if event.get("type") == "progress":
        job.progress = int(event.get("progress", job.progress))
        job.total = int(event.get("total", job.total))
    if event.get("type") == "section_classified":
        job.progress = min(job.total, job.progress + 1)
        job.stats["sections"] = job.stats.get("sections", 0) + 1
    if event.get("type") == "needs_review":
        job.stats["needs_review"] = job.stats.get("needs_review", 0) + 1
    await job.queue.put(event)
    if event.get("type") in PERSIST_EVENTS:
        await asyncio.to_thread(persist_job, job)


async def finish(job: IngestJob, status: str = "done", error: str | None = None) -> None:
    if status not in TERMINAL_STATUS:
        raise ValueError("非法终止状态")
    job.status = status
    job.error = error
    job.updated_at = _now()
    kind = {'done': 'job_done', 'failed': 'job_failed', 'cancelled': 'job_cancelled', 'skipped': 'job_skipped'}[status]
    event = {"type": kind, "status": status, "error": error, "stats": dict(job.stats)}
    await job.queue.put(event)
    await asyncio.to_thread(persist_job, job)


async def cancel(job: IngestJob) -> None:
    job.cancelled = True
    job.status = "cancelled"
    job.updated_at = _now()
    await job.queue.put({"type": "job_cancelled", "status": "cancelled"})
    await asyncio.to_thread(persist_job, job)


async def events(job: IngestJob) -> AsyncIterator[dict[str, Any]]:
    """SSE 消费端：15 秒没有事件就发心跳；终止事件后结束。"""
    while True:
        try:
            event = await asyncio.wait_for(job.queue.get(), timeout=15)
        except asyncio.TimeoutError:
            yield {"type": "heartbeat", "progress": job.progress, "total": job.total, "status": job.status}
            continue
        yield event
        if event.get("type") in {"job_done", "job_failed", "job_cancelled", "job_skipped"}:
            return
