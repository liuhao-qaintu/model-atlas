import re
import asyncio
import hashlib
import json
from contextlib import asynccontextmanager, aclosing
from uuid import uuid4
from time import monotonic
from urllib.parse import urlparse

import httpx
from fastapi import FastAPI, HTTPException, File, Form, UploadFile, Request, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from backend.infra import config
from backend.infra.database import KnowledgeModel, KnowledgeSection, KnowledgeSource, history, initialize_tables, records, save_turn, user_questions, session_is_cloud_safe
from backend.infra.conversation import conversation_reply, is_conversation_query
from backend.infra.knowledge import seed_knowledge
from backend.infra.ingestion import MAX_BYTES, UPLOAD_ROOT, commit_document, document_record, preview_document
from backend.rag.retrieval import Retriever, provider_post, provider_timeout
from backend.rag.planning import history_is_cloud_safe, plan_question
from backend.rag.pipeline import execute_plan
from backend.generation.evidence_context import context_text, prepare_evidence
from backend.infra.streaming import relay_events, stream_completion
from backend.rag.graph import build_graph
from backend.infra.figures import figure_records, answer_figures
from backend.api.analysis_routes import router as analysis_router
from backend.api.sessions import list_sessions, session_page, rename_session
from backend.api.management import router as management_router
from backend.api.evaluation import router as evaluation_router
from backend.api.comparison import router as comparison_router
from backend.api.ingest import router as ingest_router
from backend.api.review import router as review_router
from backend.api.document_admin import router as document_admin_router
from backend.api.research import router as research_router
from backend.generation.answer_validation import validate_citations, keep_known_links as preserve_known_links
from backend.generation.answer_policy import VERSION as ANSWER_POLICY, answer_profile
from backend.generation.generation_prompt import VERSION as GENERATION_PROMPT, generation_messages
from backend.generation.grounding_contract import qualify_absence_scope
from backend.generation.source_guard import inspect_strict_answer, official_excerpt_answer
from backend.rag.source_scope import VERSION as SOURCE_QUALITY_CONTRACT
from backend.generation.answer_review import VERSION as ANSWER_REVIEW, review_answer, review_trigger
from backend.api.auth import router as auth_router
from backend.infra.auth import AuthenticationMiddleware,ScopedJSONResponse
from backend.infra.access import current_actor,check_session,acting_as
from backend.infra.secure_retriever import RetrieverRegistry

retriever = None
knowledge_lock = asyncio.Lock()
STRICT_GENERATION_SECONDS = 60


@asynccontextmanager
async def lifespan(app):
    global retriever
    initialize_tables()
    seed_knowledge()
    retriever = RetrieverRegistry()
    yield


app = FastAPI(title="深度学习模型知识库", version="0.1.0", lifespan=lifespan,
              default_response_class=ScopedJSONResponse,
              docs_url='/docs' if config.ENABLE_API_DOCS else None,
              redoc_url=None,openapi_url='/openapi.json' if config.ENABLE_API_DOCS else None)
app.include_router(auth_router)
app.include_router(analysis_router)
app.include_router(management_router)
app.include_router(evaluation_router)
app.include_router(comparison_router)
app.include_router(ingest_router)
app.include_router(review_router)
app.include_router(document_admin_router)
app.include_router(research_router)
app.add_middleware(AuthenticationMiddleware)
app.add_middleware(CORSMiddleware, allow_origins=["http://127.0.0.1:5174", "http://localhost:5174"], allow_methods=["GET", "POST", "PATCH", "DELETE"], allow_headers=["Content-Type","Authorization","X-CSRF-Token"],allow_credentials=True)


@app.exception_handler(RequestValidationError)
async def validation_error(request,error):
    details = [{key:value for key,value in item.items() if key in {'loc','msg','type'}} for item in error.errors()]
    return JSONResponse({'detail':details},status_code=422)


def scoped_retriever():
    return retriever.current() if isinstance(retriever,RetrieverRegistry) else retriever


def replace_retriever(updated):
    global retriever
    if isinstance(retriever,RetrieverRegistry):
        retriever.install(updated)
    else:
        retriever = updated


def reload_shared_knowledge():
    with acting_as(None):
        seed_knowledge()


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    model_id: str | None = None
    document_id: str | None = Field(default=None, max_length=40)
    document_ids: list[str] = Field(default_factory=list, max_length=3)
    document_policy: str = Field(default='auto', pattern='^(auto|strict)$')
    section_id: str | None = None
    mode: str = Field(default="learning", pattern="^(learning|research)$")
    session_id: str | None = Field(default=None, pattern=r"^[a-f0-9-]{36}$")
    turn_id: str | None = Field(default=None, pattern=r"^[a-f0-9-]{36}$")
    activities: list[dict[str, str]] = Field(default_factory=list, max_length=3)


@app.get("/api/health")
def health():
    actor = current_actor.get()
    if actor is not None and actor.id is None:
        return {'status':'ok'}
    result = {"status": "ok", "models": len(retriever.models), "ready_models": sum(m["ready"] for m in retriever.models.values()), "documents": len(retriever.documents), "chunks": len(retriever.chunks), "vector_ready": retriever.vectors is not None, "index": retriever.index_status(), "generation_configured": bool(config.LLM_KEY)}
    if actor is None or actor.maintainer:
        result.update(retrieval_pipeline="bounded-multistep-v1",chat_transport="sse-v1",conversation="history-routing-v1",source_quality_contract=SOURCE_QUALITY_CONTRACT,answer_policy=ANSWER_POLICY,answer_review=ANSWER_REVIEW,generation_prompt=GENERATION_PROMPT,database=str(config.DATABASE_URL.get_backend_name()),embedding_model=config.EMBEDDING_MODEL,rerank_configured=bool(config.RERANK_KEY),execution=execution_info())
    return result


@app.get("/api/models")
def models():
    return sorted(retriever.models.values(), key=lambda model: (not model["ready"], model["order"]))


@app.get("/api/models/{model_id}")
def model_detail(model_id: str):
    if model_id not in retriever.models:
        raise HTTPException(404, "模型不存在")
    return {**retriever.models[model_id], "sections": [s for s in retriever.sections if s["model_id"] == model_id], "paths": retriever.paths([model_id]), 'figures': figure_records(model_id)}


@app.get("/api/sources")
def sources():
    return list(retriever.sources.values())


@app.get("/api/graph")
def graph(root: str | None = Query(default=None, max_length=120), hops: int = Query(default=2, ge=1, le=2), kind: str | None = Query(default=None, max_length=40)):
    try:
        return build_graph(retriever, root, hops, kind)
    except LookupError as exc:
        raise HTTPException(404, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@app.get('/api/sessions')
def sessions(query: str = Query(default='', max_length=80), before: int | None = Query(default=None, ge=1),
             limit: int = Query(default=20, ge=1, le=50)):
    return list_sessions(query, before, limit)


class SessionTitleRequest(BaseModel):
    title: str = Field(min_length=1, max_length=120)


@app.get('/api/sessions/{session_id}/messages')
def session_message_page(session_id: str, before: int | None = Query(default=None, ge=1),
                         limit: int = Query(default=100, ge=2, le=200)):
    try:
        return session_page(session_id, before, limit)
    except LookupError as exc:
        raise HTTPException(404, str(exc)) from exc


@app.patch('/api/sessions/{session_id}')
def edit_session_title(session_id: str, request: SessionTitleRequest):
    try:
        return rename_session(session_id, request.title)
    except LookupError as exc:
        raise HTTPException(404, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@app.get("/api/sessions/{session_id}")
def session_messages(session_id: str):
    return history(session_id, 100)


@app.post("/api/index")
@app.post("/api/documents/index")
async def build_index():
    try:
        async with knowledge_lock:
            return await retriever.build_index()
    except (RuntimeError, httpx.HTTPError) as exc:
        raise HTTPException(502, "向量索引构建失败，请检查后端模型配置或服务状态。") from exc


@app.post("/api/knowledge/refresh")
async def refresh_knowledge():
    global retriever
    async with knowledge_lock:
        await run_in_threadpool(reload_shared_knowledge)
        updated = await run_in_threadpool(Retriever)
        replace_retriever(updated)
    return {"models": len(updated.models), "index": updated.index_status()}


class ImportRequest(BaseModel):
    title: str = Field(min_length=1, max_length=180)
    source_url: str = Field(default="", max_length=2000)
    cloud_allowed: bool = False


@app.get("/api/documents")
def documents():
    from sqlalchemy import select
    from sqlalchemy.orm import Session
    from backend.infra.database import engine,KnowledgeDocument
    actor = current_actor.get()
    owned_ids = None
    if actor is not None:
        with Session(engine) as db:
            owned_ids = set(db.scalars(select(KnowledgeDocument.id).where(KnowledgeDocument.owner_id == actor.id)))
    return [{key: value for key, value in document.items() if key not in {"parts", "markdown"}} | {"part_count": len(document["parts"])} for document in retriever.documents if owned_ids is None or document['id'] in owned_ids]


@app.get("/api/documents/{document_id}")
def document_detail(document_id: str):
    document = document_record(document_id)
    if not document:
        raise HTTPException(404, "资料不存在")
    sections = [s for s in retriever.sections if s.get('document_id') == document_id]
    if document.get('status') == 'archived':
        sections = sorted((s for s in records(KnowledgeSection) if s.get('document_id') == document_id), key=lambda s: s.get('order', 0))
    return {**document, 'sections': sections}


@app.get("/api/documents/{document_id}/file")
def original_file(document_id: str, inline: bool = False):
    document = document_record(document_id)
    if not document:
        raise HTTPException(404, "资料不存在")
    path = UPLOAD_ROOT / f"{document['id']}.{document['format']}"
    if not path.is_file():
        raise HTTPException(404, "原始文件不存在")
    return FileResponse(path, filename=document["filename"], content_disposition_type="inline" if inline else "attachment")


@app.post("/api/documents/preview")
async def preview(file: UploadFile = File(...), model_id: str = Form(...), pages: str = Form(default='', max_length=500)):
    if model_id not in retriever.models:
        raise HTTPException(422, "请先选择关联模型")
    try:
        raw = await file.read(MAX_BYTES + 1)
        async with knowledge_lock:
            return await run_in_threadpool(preview_document, file.filename, raw, model_id, pages)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    finally:
        await file.close()


@app.post("/api/documents/{document_id}/import")
async def import_document(document_id: str, request: ImportRequest):
    global retriever
    try:
        async with knowledge_lock:
            document = await run_in_threadpool(commit_document, document_id, request.title, request.source_url.strip(), request.cloud_allowed)
            replace_retriever(await run_in_threadpool(Retriever))
        return {"document": {k: v for k, v in document.items() if k != "parts"}, "part_count": len(document["parts"]), "index": retriever.index_status()}
    except ValueError as exc:
        from backend.infra.document_versions import VersionConflict
        if isinstance(exc, VersionConflict):
            raise HTTPException(409, str(exc)) from exc
        raise HTTPException(422, str(exc)) from exc
    except LookupError as exc:
        raise HTTPException(404, str(exc)) from exc


@app.post("/api/search")
async def search(request: ChatRequest):
    validate_context(request)
    from backend.rag.document_scope import selected_documents
    documents = selected_documents(request, retriever)
    if documents:
        ids = {d['id'] for d in documents}
        allowed = {s['id'] for s in retriever.sections if s.get('document_id') in ids}
        owner = documents[0]['model_id'] if len({d['model_id'] for d in documents}) == 1 else None
        return await retriever.search(request.question.strip(), owner, request.section_id, strict_focus=True, allowed_section_ids=allowed)
    return await retriever.search(request.question.strip(), request.model_id, request.section_id)


def validate_context(request, active_retriever=None):
    check_session(request.session_id)
    active_retriever = active_retriever or retriever
    if any(item.get('model_id') not in active_retriever.models or item.get('action') not in {'model', 'graph'} for item in request.activities):
        raise HTTPException(422, '浏览上下文包含无效模型或操作')
    if request.model_id and request.model_id not in active_retriever.models:
        raise HTTPException(422, "所选模型不存在")
    if request.document_id:
        document = next((d for d in active_retriever.documents if d['id'] == request.document_id), None)
        if not document or (not request.document_ids and request.model_id and document['model_id'] != request.model_id):
            raise HTTPException(422, '所选文献不存在、已归档或与当前范围不匹配')
    if len(set(request.document_ids + ([request.document_id] if request.document_id else []))) > 3:
        raise HTTPException(422, '本轮最多选择3篇文献')
    if any(not any(d['id'] == identity for d in active_retriever.documents) for identity in request.document_ids):
        raise HTTPException(422, '所选文献不存在或已归档')
    if request.section_id:
        section = next((s for s in active_retriever.sections if s["id"] == request.section_id), None)
        if section is None or section["model_id"] != request.model_id:
            raise HTTPException(422, "阅读章节与模型不匹配")
    if not request.question.strip():
        raise HTTPException(422, "请输入问题")


@app.post("/api/chat")
async def chat(request: ChatRequest):
    active_retriever = scoped_retriever()
    validate_context(request, active_retriever)
    async for name, data in chat_events(request, active_retriever):
        if name == "done":
            return data


@app.post("/api/chat/stream", response_class=StreamingResponse)
async def chat_stream(request: ChatRequest, connection: Request):
    active_retriever = scoped_retriever()
    validate_context(request, active_retriever)
    return StreamingResponse(
        relay_events(chat_events(request, active_retriever, streaming=True), connection.is_disconnected),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


async def chat_events(request, active_retriever, streaming=False):
    session_id = request.session_id or str(uuid4())
    previous = history(session_id, 20)
    question = request.question.strip()
    response_profile = answer_profile(question, request.mode)
    if is_conversation_query(question):
        yield "stage", {"stage": "history", "label": "正在读取本会话的已保存提问"}
        questions = user_questions(session_id)
        last = next((item.get("metadata", {}) for item in reversed(previous) if item["role"] == "assistant"), {})
        focus = request.model_id or last.get("model_id")
        ids = [focus] if focus else last.get("plan", {}).get("model_ids", [])
        selected_section = request.section_id or (last.get("section_id") if last.get("model_id") == focus else None)
        selected_section = next((s["id"] for s in active_retriever.sections if s["id"] == selected_section and s["model_id"] == focus), None)
        plan = {"original_query": question, "normalized_query": question, "mode": "conversation", "method": "history", "model_id": focus,
                "model_ids": ids, "section_id": selected_section, "queries": [], "context_origin": "session_history",
                "clarification": None, "followup": True, "duration_ms": 0, "warnings": []}
        result = {"session_id": session_id, "turn_id": request.turn_id, "answer": conversation_reply(question, questions), "model_id": focus,
                  "section_id": plan["section_id"], "mode": request.mode, "normalized_query": question, "plan": plan, "response_kind": "conversation",
                  "retrieval_steps": [], "retrieval_stop_reason": "按消息顺序读取当前会话，不检索模型资料", "timing": {"planning_ms": 0, "retrieval_ms": 0},
                  "evidence": [], "links": [], "paths": [], "warnings": [], "channels": {}, "reranked": False, "generated": False,
                  "history_questions": len(questions), "cloud_allowed": session_is_cloud_safe(session_id), "execution": execution_info(False)}
        yield "stage", {"stage": "saving", "label": "正在保存会话回答"}
        result['document_id'] = request.document_id or last.get('document_id')
        result['document_ids'] = request.document_ids or ([request.document_id] if request.document_id else [])
        result['document_policy'] = request.document_policy
        await asyncio.sleep(0)
        save_turn(session_id, question, result["answer"], result)
        yield "done", result
        return
    yield "stage", {"stage": "planning", "label": "正在确认问题、模型与上下文"}
    selected_model = request.model_id
    has_conversation_model = any(item.get('metadata', {}).get('model_id') or item.get('metadata', {}).get('plan', {}).get('model_ids') for item in previous)
    if not selected_model and not has_conversation_model and request.activities:
        selected_model = request.activities[0]['model_id']
    plan = await plan_question(question, active_retriever, selected_model, request.section_id, previous)
    from backend.rag.document_scope import selected_documents, pin_document_plan
    document_scope = selected_documents(request, active_retriever)
    original_plan = plan
    if plan.get('followup') and history_is_cloud_safe(previous):
        last_scope = next((m.get('metadata', {}).get('retrieval_scope') or {} for m in reversed(previous) if m.get('role') == 'assistant'), {})
        original_plan = {**plan, 'supplementary_model_ids': last_scope.get('supplementary_model_ids', [])}
    if document_scope:
        plan = pin_document_plan(plan, document_scope)
    if not request.model_id and selected_model and plan['context_origin'] == 'selected_model':
        plan['context_origin'] = 'recent_activity'
    plan['history_questions'] = [item['content'] for item in previous if item['role'] == 'user'][-3:]
    plan['activities'] = [{'model_id': item['model_id'], 'name': active_retriever.models[item['model_id']]['name'], 'action': item['action']} for item in request.activities]
    plan['model_names'] = [active_retriever.models[mid]['name'] for mid in plan['model_ids']]
    plan['planning_basis'] = [
        {'question': '按本轮明确提到的模型确定范围。', 'selected_model': '采用页面中选择的模型范围。', 'last_turn': '从本会话上一轮确定代词指向。', 'recent_activity': '采用你启用的最近浏览模型。'}.get(plan['context_origin'], '结合本轮问题和已登记实体确定范围。'),
        '比较对象分别查找；关系问题只有前一步正文支持时才继续扩展。' if plan['mode'] != 'single' else '本轮按单个问题查找相关正文。',
    ]
    yield 'plan', plan
    focus, normalized = plan["model_id"], plan["normalized_query"]
    yield "stage", {"stage": "retrieving", "label": "正在检索资料并合并依据"}
    retrieval = await execute_plan(plan, active_retriever)
    from backend.rag.adaptive_chat import needs_agent, refine_events
    if needs_agent(question, plan, retrieval, history_is_cloud_safe(previous)):
        async for name, value in refine_events(active_retriever, plan, retrieval):
            if name == 'retrieval':
                retrieval = value
            else:
                yield name, value
    if document_scope:
        from backend.rag.scope_expansion import expand_if_needed
        yield 'stage', {'stage': 'retrieving', 'label': '正在核对资料并按需补查知识库'}
        retrieval = await expand_if_needed(question, request, original_plan, plan, retrieval, active_retriever)
    evidence = prepare_evidence(retrieval["evidence"], active_retriever, plan['normalized_query'])
    warnings = retrieval["warnings"]
    if any(item['generation_selection']['strategy'] == 'formula_gate' for item in evidence):
        warnings.append('部分资料页为 PDF 公式版式，文字提取不可靠，未送入生成；公式细节请查看原文对应页。')
    links = [{"label": f"{active_retriever.models[s['model_id']]['name']} · {s['title']}" if len(plan["model_ids"]) > 1 else s["title"], "url": s.get("reader_url", f"/models/{s['model_id']}?section={s['slug']}")} for s in active_retriever.sections if s["model_id"] in plan["model_ids"] and s["slug"] not in ("figures", "followups", "sources")][:12]
    if document_scope:
        links = [{'label': d['title'], 'url': f"/documents/{d['id']}"} for d in document_scope]
        links += [{'label': active_retriever.models[mid]['name'], 'url': f'/models/{mid}'}
                  for mid in retrieval.get('scope', {}).get('supplementary_model_ids', [])]
    generated = False
    generation_diagnostics = {}
    if plan["clarification"]:
        answer = plan["clarification"]
    elif not evidence:
        available_names = "、".join(model["name"] for model in active_retriever.models.values() if model["searchable"])
        missing_names = "、".join(active_retriever.models[mid]["name"] for mid in plan["model_ids"] if not active_retriever.models[mid]["searchable"])
        answer = f"当前资料不足以回答这个问题。{missing_names + ' 的资料尚未完成整理。' if missing_names else '本轮没有检索到可用正文。'}可先查看已有资料的模型：{available_names}。"
    elif not retrieval.get("remote_allowed", True) or any(not item.get("cloud_allowed", True) for item in evidence):
        answer = "本轮命中仅允许本地处理的资料，以下为原文摘录：\n\n" + "\n\n".join(
            f"**[{i}] {item['title']}**\n"
            + (item['generation_excerpt'] if item['generation_selection']['strategy'] == 'formula_gate' else item['content'][:500])
            for i, item in enumerate(evidence, 1))
        warnings.append("未将这批资料发送到云端生成；可通过资料页查看原文。")
    else:
        if any(item['generation_selection']['strategy'] in ('head', 'python_units') for item in evidence):
            warnings.append('本轮长正文提供有界节选；可核对实际节选与原文，完整函数以节选记录为准。')
        context = context_text(evidence)
        graph = json_paths(retrieval["paths"])
        scope = plan.get('source_scope', {})
        if retrieval.get('agentic'):
            scope = {**scope, 'adaptive_lookup': True,
                     'evidence_checks': retrieval['agentic'].get('final_checks', []),
                     'check_boundary': '机器查证项目只用于组织回答，仍须逐项核对当前正文；missing/conflict 不得写成已证实的实测结论。'}
        safe_history = previous if history_is_cloud_safe(previous) else []
        messages = generation_messages(question, normalized, context, graph, warnings,
                                       response_profile, request.mode, scope, safe_history)
        generation_start, pieces = monotonic(), []
        try:
            yield "stage", {"stage": "generating", "label": f"已找到 {len(evidence)} 个片段，正在生成回答"}
            body = {"model": config.LLM_MODEL, "messages": messages, "temperature": 0.2, "max_tokens": response_profile['max_tokens']}
            if scope.get('expanded'):
                body['stream'] = True
                if 'siliconflow.' in config.LLM_BASE and 'deepseek' in config.LLM_MODEL.lower():
                    body['enable_thinking'] = False
            deadline = STRICT_GENERATION_SECONDS if scope.get('strict') else 90 if scope.get('expanded') else None
            generation_diagnostics = {'model': config.LLM_MODEL, 'max_tokens': body['max_tokens'],
                                      'answer_policy': ANSWER_POLICY, 'answer_profile': response_profile,
                                      'generation_prompt': GENERATION_PROMPT, 'streaming': streaming,
                                      'system_chars': len(messages[0]['content']), 'context_chars': len(messages[-1]['content']),
                                      'provider_read_timeout_seconds': 65 if streaming else provider_timeout('chat/completions', body),
                                      'generation_deadline_seconds': deadline,
                                      'prompt_sha256': hashlib.sha256(json.dumps(messages, ensure_ascii=False, sort_keys=True).encode()).hexdigest()}
            async with asyncio.timeout(deadline):
                if streaming:
                    finish_reason = None
                    async with aclosing(stream_completion(config.LLM_BASE, config.LLM_KEY, body)) as stream:
                        async for item in stream:
                            if item["text"]:
                                generation_diagnostics.setdefault('first_output_ms', round((monotonic() - generation_start) * 1000))
                                pieces.append(item["text"])
                                yield "delta", {"text": item["text"]}
                            finish_reason = item.get("finish_reason") or finish_reason
                    answer = "".join(pieces).strip()
                    generation_diagnostics.update(finish_reason=finish_reason, answer_chars=len(answer))
                    if finish_reason == "length":
                        warnings.append("回答达到本轮长度上限，可继续追问具体部分。")
                    elif finish_reason not in (None, "stop"):
                        raise RuntimeError("模型未正常完成正文")
                else:
                    result = await provider_post(config.LLM_BASE, config.LLM_KEY, "chat/completions", body)
                    answer = result["choices"][0]["message"]["content"].strip()
                    usage = result.get('usage') or {}
                    generation_diagnostics.update(finish_reason=result['choices'][0].get('finish_reason'),
                                                  answer_chars=len(answer), completion_tokens=usage.get('completion_tokens'),
                                                  reasoning_chars=len(result['choices'][0]['message'].get('reasoning_content') or ''))
                    if generation_diagnostics['finish_reason'] == 'length':
                        warnings.append('回答达到本轮长度上限，可继续追问具体部分。')
            generation_diagnostics['generation_ms'] = round((monotonic() - generation_start) * 1000)
            if not answer:
                raise RuntimeError("模型未返回正文")
            review_enabled = request.mode == 'research' or bool(retrieval.get('agentic'))
            if review_trigger(response_profile, scope, question, enabled=review_enabled, draft=answer):
                yield 'stage', {'stage': 'reviewing', 'label': '正在核对计算、适用条件与引用依据'}
            answer, review = await review_answer(answer, messages, response_profile, scope, len(evidence),
                                                config.LLM_BASE, config.LLM_KEY, config.LLM_MODEL, provider_post,
                                                question=question, enabled=review_enabled)
            generation_diagnostics['review'] = review
            generation_diagnostics['final_answer_chars'] = len(answer)
            if review['status'] == 'unavailable':
                warnings.append('本轮回答已生成，但补充核对未完成；请结合引用检查关键计算与条件。')
            answer, invalid_citations = validate_citations(answer, len(evidence))
            if invalid_citations:
                warnings.append("生成结果中的无效引用编号已移除。")
            answer, qualification = qualify_absence_scope(answer)
            generation_diagnostics['scope_qualification'] = qualification
            allowed_urls = {source["url"] for item in evidence for source in item["sources"]} | {link["url"] for link in links}
            answer, dropped_links = keep_known_links(answer, allowed_urls)
            if dropped_links:
                warnings.append("生成结果中没有对应资料入口的链接已转为文字；请使用下方章节入口。")
            source_guard = inspect_strict_answer(answer, evidence, scope)
            generation_diagnostics['source_guard'] = source_guard
            if source_guard['status'] == 'blocked':
                generation_diagnostics['rejected_answer'] = answer
                answer = official_excerpt_answer(evidence)
                warnings.append('严格来源检查发现未经正文支持的延伸，当前展示原文节选，未完成可采用的模型解释。')
            generation_diagnostics['saved_answer_sha256'] = hashlib.sha256(answer.encode()).hexdigest()
            generation_diagnostics['final_answer_chars'] = len(answer)
            generated = source_guard['status'] != 'blocked'
        except (TimeoutError, RuntimeError, httpx.HTTPError, KeyError, IndexError, AttributeError, TypeError) as exc:
            generation_diagnostics['error_type'] = type(exc).__name__
            generation_diagnostics['generation_ms'] = round((monotonic() - generation_start) * 1000)
            if pieces:
                generation_diagnostics['rejected_partial_answer'] = ''.join(pieces)
            if isinstance(exc, (httpx.TimeoutException, TimeoutError)):
                warnings.append('生成服务本轮等待超时；已停止本次生成并保留资料入口，以下摘录不是生成回答。')
            elif generation_diagnostics.get('answer_chars') == 0:
                warnings.append('模型未返回回答正文；已保留完成原因及用量信息供核查。')
            answer = "本轮生成服务暂不可用。已找到以下资料，可通过来源卡查看：\n\n" + "\n\n".join(f"**[{i}] {item['title']}**\n{item['content'][:500]}" for i, item in enumerate(evidence, 1))
            warnings.append("当前展示的是检索摘录，未完成模型生成。")
    result = {"session_id": session_id, "turn_id": request.turn_id, "answer": answer, "model_id": focus, "section_id": plan["section_id"], "mode": request.mode, "normalized_query": normalized, "plan": plan, "retrieval_steps": retrieval["steps"], "retrieval_stop_reason": retrieval["stop_reason"], "timing": {"planning_ms": plan["duration_ms"], "retrieval_ms": retrieval["duration_ms"]}, "evidence": evidence, "links": links, "paths": retrieval["paths"], "warnings": warnings, "channels": retrieval["channels"], "reranked": retrieval["reranked"], "generated": generated, "execution": execution_info(generated)}
    result['document_id'] = plan.get('document_id')
    result['document_ids'] = plan.get('document_ids', [])
    result['document_policy'] = retrieval.get('scope', {}).get('policy', request.document_policy)
    result['retrieval_scope'] = retrieval.get('scope')
    result['figures'] = answer_figures(evidence)
    result['generation_diagnostics'] = generation_diagnostics
    result['agentic'] = retrieval.get('agentic')
    yield "stage", {"stage": "saving", "label": "正在检查引用并保存完整回答"}
    await asyncio.sleep(0)  # 断连/停止请求在提交完整会话前仍有取消机会。
    save_turn(session_id, question, answer, result)
    yield "done", result


def json_paths(paths):
    return "\n".join("；".join(f"{edge['source_name']} → {edge['relation']} → {edge['target_name']}（{','.join(edge['source_ids'])}，{edge['locator']}）" for edge in path["edges"]) for path in paths[:10]) or "本轮无相关关系路径。"


def keep_known_links(answer, allowed_urls):
    return preserve_known_links(answer, allowed_urls)


def execution_info(generated=None):
    host = urlparse(config.LLM_BASE).hostname
    location = "local" if host in {"127.0.0.1", "localhost", "::1"} else "remote"
    return {"retrieval": "local", "knowledge": "model_cards_and_imported_documents", "generation": location if generated is not False else "none", "llm_model": config.LLM_MODEL, "provider_host": host, "providers": {"embedding": urlparse(config.EMBEDDING_BASE).hostname, "rerank": urlparse(config.RERANK_BASE).hostname, "generation": host}, "generated": generated}
