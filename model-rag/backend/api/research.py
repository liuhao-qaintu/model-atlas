"""独立研究入口；保存报告，用户点击继续追问时再接入聊天。"""
import asyncio
import httpx
from contextlib import suppress

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from backend.infra import config
from backend.infra.database import KnowledgeAnalysis, engine
from backend.infra.streaming import relay_events
from backend.rag.research import ResearchRunner, allowed_scope

router = APIRouter(prefix='/api/research', tags=['资料研究'])
run_lock = asyncio.Lock()


def research_error(exc):
    if isinstance(exc, (httpx.ConnectError, httpx.ProxyError, httpx.ConnectTimeout)):
        return 503, '无法连接模型服务，请检查网络、代理或后端运行环境后重试。本轮未保存为成功报告。', 'model_connection_failed'
    if isinstance(exc, (TimeoutError, httpx.TimeoutException)):
        return 504, '本轮研究超时，请缩小问题范围后重试。未保存为成功报告。', 'research_timeout'
    return 502, '本轮研究未完整完成，请重试或检查后端诊断记录。未保存为成功报告。', 'research_failed'


class ResearchInput(BaseModel):
    question: str = Field(min_length=1, max_length=1500)
    document_ids: list[str] = Field(default_factory=list, max_length=3)


def prepare(value):
    from backend.main import scoped_retriever
    retriever = scoped_retriever()
    if not value.question.strip():
        raise HTTPException(422, '请输入要研究的问题')
    try:
        allowed_scope(retriever, value.document_ids)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    if not config.LLM_KEY:
        raise HTTPException(422, '尚未配置研究模型')
    if run_lock.locked():
        raise HTTPException(409, '已有研究在运行，请稍后再试')
    return retriever


def save_report(result):
    with Session(engine) as db:
        db.merge(KnowledgeAnalysis(id=result['id'], payload=result))
        db.commit()


@router.get('/runs/{identity}')
def report(identity: str):
    with Session(engine) as db:
        row = db.get(KnowledgeAnalysis, identity)
        if not row or row.payload.get('type') != 'agentic_research_run':
            raise HTTPException(404, '研究记录不存在')
        return row.payload


@router.post('')
async def research(value: ResearchInput):
    active = prepare(value)
    async with run_lock:
        try:
            result = await ResearchRunner(active, value.document_ids).run(value.question.strip())
            await asyncio.to_thread(save_report, result)
            return result
        except (RuntimeError, ValueError, TimeoutError, KeyError, httpx.HTTPError) as exc:
            status, detail, _ = research_error(exc)
            raise HTTPException(status, detail) from exc


@router.post('/runs/{identity}/conversation')
def continue_conversation(identity: str):
    from backend.main import scoped_retriever
    retriever = scoped_retriever()
    from backend.infra.research_conversation import open_research_conversation
    try:
        return open_research_conversation(identity, retriever, engine)
    except LookupError as exc:
        raise HTTPException(404, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc


async def research_events(value, active):
    queue = asyncio.Queue(maxsize=16)

    async def emit(name, data):
        await queue.put((name, data))

    async def produce():
        async with run_lock:
            try:
                result = await ResearchRunner(active, value.document_ids, emit=emit).run(value.question.strip())
                await asyncio.to_thread(save_report, result)
                await queue.put(('done', result))
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                _, detail, code = research_error(exc)
                await queue.put(('error', {'detail': detail, 'code': code}))

    task = asyncio.create_task(produce())
    try:
        while True:
            name, data = await queue.get()
            yield name, data
            if name in {'done', 'error'}:
                return
    finally:
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task


@router.post('/stream')
async def research_stream(value: ResearchInput, connection: Request):
    active = prepare(value)
    return StreamingResponse(relay_events(research_events(value, active), connection.is_disconnected),
                             media_type='text/event-stream', headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})
