"""在标注题集上复用真实检索流程，保存检索指标与人工回答评分。"""
import asyncio
import hashlib
import json
import httpx
from datetime import datetime, timezone
from time import perf_counter
from uuid import uuid4

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from backend.infra import config
from backend.infra.database import ChatMessage, ChatSession, KnowledgeAnalysis, engine, records, session_is_cloud_safe
from backend.generation.answer_assessment import (VERSION as ASSESSMENT_VERSION, assess_with_model, current_assessment,
                                fingerprint as answer_fingerprint, validate_assessment, validate_contract)
from backend.generation.evaluation_metrics import (PROTOCOL, DIMENSIONS, aggregate, annotation_fingerprint, normalize_run,
                                 qrels_key, rescore_row, retrieval_metrics, reuse_judgments, seed_judgments)
from backend.rag.pipeline import execute_plan
from backend.rag.planning import rule_plan

router = APIRouter(prefix='/api/evaluation', tags=['RAG质量评测'])
run_lock = asyncio.Lock()


def cases():
    seed = json.loads((config.DATA_ROOT / 'evaluation-cases.json').read_text(encoding='utf-8'))
    challenge = json.loads((config.DATA_ROOT / 'evaluation-challenge-cases.json').read_text(encoding='utf-8'))
    contracts = json.loads((config.DATA_ROOT / 'answer-evaluation-cases.json').read_text(encoding='utf-8'))
    extra = [r['case'] for r in records(KnowledgeAnalysis) if r.get('type') == 'evaluation_case']
    return [{**c, 'split': c.get('split', 'development'), 'annotation_origin': c.get('annotation_origin', 'assistant'),
             'reference_review': c.get('reference_review', 'pending')} for c in seed + challenge + contracts + extra]


def save(record, related=None):
    with Session(engine) as db:
        db.merge(KnowledgeAnalysis(id=record['id'], payload=record))
        if related: db.merge(KnowledgeAnalysis(id=related['id'], payload=related))
        db.commit()


def get_run(identity):
    with Session(engine) as db:
        row = db.get(KnowledgeAnalysis, identity)
        if not row or row.payload.get('type') != 'evaluation_run': raise HTTPException(404, '评测记录不存在')
        return normalize_run(row.payload)


def corpus_fingerprint(retriever):
    payload = {'sections': [(s['id'], s['content']) for s in retriever.sections],
               'index_policy': getattr(retriever, 'index_policy', 'legacy'),
               'exclusions': getattr(retriever, 'index_exclusions', []),
               'document_kinds': getattr(retriever, 'document_kinds', {}),
               'document_priority': getattr(retriever, 'document_priority_snapshot', {})}
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


@router.get('/cases')
def list_cases(): return cases()


class CaseInput(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    model_id: str | None = Field(default=None, max_length=80)
    reference_answer: str = Field(min_length=1, max_length=5000)
    relevant_section_ids: list[str] = Field(min_length=1, max_length=10)
    cloud_allowed: bool = False
    split: str = Field(default='development', pattern='^(development|challenge)$')


@router.post('/cases')
def add_case(value: CaseInput):
    from backend.main import retriever
    sections = {s['id'] for s in retriever.sections if not value.model_id or s['model_id'] == value.model_id}
    if not value.question.strip() or not value.reference_answer.strip() or not set(value.relevant_section_ids) <= sections:
        raise HTTPException(422, '问题、参考答案和所选模型的相关片段标注必须有效')
    record = {'id': 'ec-' + uuid4().hex[:24], 'type': 'evaluation_case', 'case': {**value.model_dump(), 'annotation_origin': 'user', 'reference_review': 'user_authored', 'id': 'custom-' + uuid4().hex[:16]}}
    save(record); return record['case']


class RunInput(BaseModel):
    mode: str = Field(default='rag', pattern='^(rag|local)$')
    case_ids: list[str] | None = Field(default=None, min_length=1, max_length=50)
    split: str = Field(default='development', pattern='^(development|challenge|all)$')
    purpose: str = Field(default='evaluation', pattern='^(evaluation|regression)$')


def select_cases(dataset, case_ids, split='all'):
    if split != 'all': dataset = [case for case in dataset if case.get('split', 'development') == split]
    if case_ids is None: return dataset
    lookup = {case['id']: case for case in dataset}
    if not set(case_ids) <= set(lookup):
        raise HTTPException(422, '选中的题目已不存在，请刷新题目列表')
    return [lookup[identity] for identity in dict.fromkeys(case_ids)]


@router.get('/runs')
def list_runs():
    runs = [r for r in records(KnowledgeAnalysis) if r.get('type') == 'evaluation_run']
    return [{'id': r['id'], 'created_at': r['created_at'], 'mode': r['mode'], 'summary': normalize_run(r)['summary']} for r in sorted(runs, key=lambda r: r['created_at'], reverse=True)[:20]]


async def retrieve_case(case, active, remote):
    """追问前序也实际检索，历史元数据来自流程结果而非手工指定答案。"""
    previous, prelude = [], []
    for turn in case.get('prior_turns', []):
        plan = rule_plan(turn['question'], active, turn.get('model_id'), previous=previous)
        plan['remote_allowed'] = remote
        result = await execute_plan(plan, active)
        prelude.append({'question': turn['question'], 'plan': plan, 'evidence_ids': [e['section_id'] for e in result['evidence']]})
        previous.extend([{'role': 'user', 'content': turn['question'], 'metadata': {}},
                         {'role': 'assistant', 'content': '', 'metadata': {'model_id': plan['model_id'], 'plan': plan}}])
    plan = rule_plan(case['question'], active, case.get('model_id'), previous=previous)
    plan['remote_allowed'] = remote
    return plan, await execute_plan(plan, active), prelude


@router.get('/runs/{identity}')
def run_detail(identity: str): return get_run(identity)


@router.post('/runs')
async def run_evaluation(value: RunInput):
    from backend.main import retriever
    if run_lock.locked(): raise HTTPException(409, '已有评测在运行，请稍后查看记录')
    async with run_lock:
        active = retriever; dataset = select_cases(cases(), value.case_ids, value.split); semaphore = asyncio.Semaphore(3)
        if not dataset: raise HTTPException(422, '本范围没有评测题')
        corpus = corpus_fingerprint(active)
        pools = {r['id']: r for r in records(KnowledgeAnalysis) if r.get('type') == 'evaluation_qrels'}
        section_ids = {s['id'] for s in active.sections}
        async def evaluate(case):
            row = {'case': case, 'answer': None, 'grade': None, 'metrics': None,
                   'judgments': reuse_judgments(case, corpus, pools), 'judgment_history': [], 'qrels_id': qrels_key(case, corpus)}
            if not set(case['relevant_section_ids']) <= section_ids:
                return {**row, 'unscorable': True, 'error': '标注片段已不存在，需先更新题目'}
            if case.get('answer_contract'):
                try: validate_contract(case, active.sections)
                except ValueError as exc: return {**row, 'unscorable': True, 'error': str(exc)}
            async with semaphore:
                started = perf_counter()
                try:
                    plan, result, prelude = await asyncio.wait_for(retrieve_case(case, active, value.mode == 'rag' and case.get('cloud_allowed', True)), timeout=100)
                    evidence = result['evidence']
                    row = {**row, 'plan': plan, 'retrieval_prelude': prelude, 'retrieval_steps': result['steps'],
                            'evidence': [{key: item.get(key) for key in ['section_id', 'model_id', 'title', 'reader_url', 'channels', 'content', 'locator']} for item in evidence],
                            'metrics': retrieval_metrics([item['section_id'] for item in evidence], case['relevant_section_ids'], judgments=row['judgments']),
                            'warnings': result['warnings'], 'seconds': round(perf_counter() - started, 2), 'reranked': result['reranked']}
                    rescore_row(row); return row
                except (RuntimeError, TimeoutError, ValueError, httpx.HTTPError) as exc:
                    return {**row, 'error': type(exc).__name__, 'seconds': round(perf_counter() - started, 2)}
        rows = await asyncio.gather(*(evaluate(case) for case in dataset))
        record = {'id': 'er-' + uuid4().hex[:24], 'type': 'evaluation_run', 'created_at': datetime.now(timezone.utc).isoformat(),
                  'protocol': PROTOCOL, 'dataset_split': value.split, 'purpose': value.purpose,
                  'answer_assessment_version': ASSESSMENT_VERSION,
                  'qrels_sha256': annotation_fingerprint(rows),
                  'mode': value.mode, 'planning': 'deterministic_rules', 'corpus_sha256': corpus, 'dataset_sha256': hashlib.sha256(json.dumps(dataset, ensure_ascii=False, sort_keys=True).encode()).hexdigest(),
                  'embedding_model': config.EMBEDDING_MODEL, 'rerank_model': config.RERANK_MODEL, 'rows': rows, 'summary': aggregate(rows)}
        save(record); return record


@router.post('/runs/{identity}/answers/{case_id}')
async def generate_answer(identity: str, case_id: str):
    from backend.main import ChatRequest, chat, retriever
    record = get_run(identity)
    row = next((row for row in record['rows'] if row['case']['id'] == case_id), None)
    if not row: raise HTTPException(404, '题目不存在')
    if not row['case'].get('cloud_allowed', True): raise HTTPException(422, '本题未授权云端模型处理，不能调用生成')
    if row.get('generated') or row.get('answer_ready'): return record
    if corpus_fingerprint(retriever) != record['corpus_sha256']:
        raise HTTPException(409, '知识库已变化，请重新运行评测后生成回答，避免混合不同版本')
    # 各题使用独立临时会话；追问经历真实前序回答，结束后不污染用户会话列表。
    session_id, prelude = str(uuid4()), []
    try:
        for turn in row['case'].get('prior_turns', []):
            prior = await chat(ChatRequest(question=turn['question'], model_id=turn.get('model_id'), session_id=session_id, mode='learning'))
            prelude.append({'question': turn['question'], 'answer': prior['answer'], 'plan': prior['plan'], 'generated': prior['generated'],
                            'cloud_allowed': all(e.get('cloud_allowed', False) for e in prior.get('evidence', []))})
        result = await chat(ChatRequest(question=row['case']['question'], model_id=row['case'].get('model_id'), session_id=session_id, mode='learning'))
    finally:
        with Session(engine) as db:
            db.execute(delete(ChatMessage).where(ChatMessage.session_id == session_id))
            db.execute(delete(ChatSession).where(ChatSession.id == session_id)); db.commit()
    record = get_run(identity); row = next(row for row in record['rows'] if row['case']['id'] == case_id)
    if row.get('answer') is not None:
        row.setdefault('generation_attempts', []).append({key: row.get(key) for key in ['answer', 'generated', 'answer_evidence', 'answer_plan', 'answer_warnings', 'generation_diagnostics']})
    row.update(answer=result['answer'], generated=result['generated'], answer_ready=result['generated'] or result.get('response_kind') == 'conversation' or bool(result['plan'].get('clarification')),
               answer_prelude=prelude, answer_evidence=[e['section_id'] for e in result['evidence']], answer_plan=result['plan'], answer_paths=result.get('paths', []),
               answer_evidence_details=[{key: e.get(key) for key in ('section_id', 'model_id', 'title', 'reader_url', 'content', 'locator', 'source_type', 'generation_excerpt', 'generation_selection')} for e in result['evidence']],
               answer_warnings=result.get('warnings', []), generation_diagnostics=result.get('generation_diagnostics', {}),
               answer_cloud_allowed=all(e.get('cloud_allowed', False) for e in result['evidence']) and all(t['cloud_allowed'] for t in prelude),
               answer_origin='evaluation_generation', grade=None, answer_rubric={}, machine_assessment=None, human_assessment=None)
    for target, original in zip(row['answer_evidence_details'], result['evidence']):
        target['cloud_allowed'] = original.get('cloud_allowed', False)
    rescore_row(row)
    record['summary'] = aggregate(record['rows']); save(record); return record


class GradeInput(BaseModel):
    grade: str = Field(pattern='^(correct|partial|incorrect)$')
    origin: str = Field(default='user', pattern='^(user|assistant)$')
    note: str = Field(default='', max_length=2000)
    dimensions: dict[str, str] = Field(default_factory=dict, max_length=4)


@router.patch('/runs/{identity}/grades/{case_id}')
def grade_answer(identity: str, case_id: str, value: GradeInput):
    record = get_run(identity); row = next((r for r in record['rows'] if r['case']['id'] == case_id), None)
    if not row: raise HTTPException(404, '题目不存在')
    if not (row.get('generated') or row.get('answer_ready')): raise HTTPException(422, '需先获得完整回答，再进行回答评分')
    if row['case'].get('answer_contract'):
        raise HTTPException(422, '本题已设置逐项标准，请完成逐项复核后保存，综合结论将按关键错误计算')
    if not set(value.dimensions) <= set(DIMENSIONS) or not set(value.dimensions.values()) <= {'correct', 'partial', 'incorrect', 'not_applicable'}:
        raise HTTPException(422, '回答维度或评分无效')
    row.setdefault('grade_history', []).append({'before': {key: row.get(key) for key in ('grade', 'grade_origin', 'grade_note', 'answer_rubric')},
                                               'after': value.model_dump(), 'at': datetime.now(timezone.utc).isoformat()})
    row.update(grade=value.grade, grade_origin=value.origin, grade_note=value.note, answer_rubric=value.dimensions)
    record['summary'] = aggregate(record['rows']); save(record); return record


class AssessmentInput(BaseModel):
    checks: list[dict] = Field(min_length=1, max_length=12)
    issues: list[dict] = Field(default_factory=list, max_length=12)
    origin: str = Field(default='user', pattern='^(user|assistant)$')


def assessment_row(identity, case_id):
    record = get_run(identity)
    row = next((r for r in record['rows'] if r['case']['id'] == case_id), None)
    if not row: raise HTTPException(404, '题目不存在')
    if not row['case'].get('answer_contract'): raise HTTPException(422, '本题尚未设置逐项标准')
    return record, row


def store_assessment(record, row, assessment):
    key = 'human_assessment' if assessment['origin'] == 'user' else 'machine_assessment'
    row.setdefault('assessment_history', []).append({'before': row.get(key), 'after': assessment})
    row[key] = assessment
    if assessment['origin'] == 'user':
        row.update(grade=assessment['overall'] if assessment['overall'] != 'pending' else None,
                   grade_origin='user', grade_note='逐项复核；关键错误不可由其他通过项抵消。')
    record['summary'] = aggregate(record['rows']); save(record); return record


@router.patch('/runs/{identity}/assessments/{case_id}')
def review_answer_items(identity: str, case_id: str, value: AssessmentInput):
    record, row = assessment_row(identity, case_id)
    try: assessment = validate_assessment(row, value.model_dump(), value.origin)
    except (ValueError, TypeError, KeyError) as exc: raise HTTPException(422, str(exc)) from exc
    return store_assessment(record, row, assessment)


@router.post('/runs/{identity}/assessments/{case_id}/machine')
async def machine_review_answer(identity: str, case_id: str):
    record, row = assessment_row(identity, case_id)
    if current_assessment(row, 'machine_assessment'): return record
    if run_lock.locked(): raise HTTPException(409, '已有评测操作在运行')
    async with run_lock:
        initial = answer_fingerprint(row)
        try:
            assessment = await assess_with_model(row)
        except (ValueError, RuntimeError, TimeoutError, httpx.HTTPError) as exc:
            record, row = assessment_row(identity, case_id)
            row.setdefault('assessment_attempts', []).append({'status': 'unavailable', 'error': type(exc).__name__,
                                                             'reason': str(exc) if isinstance(exc, ValueError) else '外部审阅未完成',
                                                             'diagnostics': getattr(exc, 'diagnostics', {}),
                                                             'at': datetime.now(timezone.utc).isoformat()})
            save(record); return record
        record, row = assessment_row(identity, case_id)
        if answer_fingerprint(row) != initial: raise HTTPException(409, '回答或标准已变化，丢弃过期审阅')
        return store_assessment(record, row, assessment)


class CaptureInput(BaseModel):
    session_id: str = Field(pattern=r'^[a-f0-9-]{36}$')
    turn_id: str = Field(pattern=r'^[a-f0-9-]{36}$')


@router.post('/runs/{identity}/capture/{case_id}')
def capture_existing_answer(identity: str, case_id: str, value: CaptureInput):
    record, row = assessment_row(identity, case_id)
    if row.get('answer') is not None: raise HTTPException(409, '本题已有回答，请新建记录以保留原始结果')
    with Session(engine) as db:
        messages = list(reversed(db.scalars(select(ChatMessage).where(ChatMessage.session_id == value.session_id)
                                           .order_by(ChatMessage.sequence.desc()).limit(120)).all()))
    index = next((i for i, m in enumerate(messages) if m.role == 'assistant' and m.payload.get('turn_id') == value.turn_id), None)
    if index is None or index < 1: raise HTTPException(404, '在该会话最近60轮中未找到指定回答')
    source = messages[index]; prior = [m.content for m in messages[:index] if m.role == 'user']
    if not prior or prior[-1] != row['case']['question']: raise HTTPException(422, '历史问题与本题不一致，不能混用答案')
    expected = [t['question'] for t in row['case'].get('prior_turns', [])]
    if expected and prior[-len(expected)-1:-1] != expected: raise HTTPException(422, '历史前序问题与本题不一致')
    result = source.payload
    row.update(answer=source.content, generated=result.get('generated', False),
               answer_ready=bool(result.get('generated') or result.get('response_kind') == 'conversation' or result.get('plan', {}).get('clarification')),
               answer_plan=result.get('plan', {}), answer_warnings=result.get('warnings', []),
               answer_evidence_details=[{key: e.get(key) for key in ('section_id', 'model_id', 'title', 'reader_url', 'content', 'locator', 'source_type', 'generation_excerpt', 'generation_selection', 'cloud_allowed')} for e in result.get('evidence', [])],
               answer_origin='captured_session', answer_cloud_allowed=session_is_cloud_safe(value.session_id),
               answer_prelude=([{'question': messages[i-1].content, 'answer': m.content} for i, m in enumerate(messages[:index]) if m.role == 'assistant' and i > 0][-len(expected):] if expected else []),
               capture={'session_id': value.session_id, 'turn_id': value.turn_id, 'message_id': source.id, 'original_corpus_version': 'not_recorded'},
               grade=None, answer_rubric={})
    record['summary'] = aggregate(record['rows']); save(record); return record


class JudgmentInput(BaseModel):
    section_id: str = Field(min_length=1, max_length=160)
    label: str = Field(pattern='^(direct|supporting|irrelevant|unjudged)$')
    origin: str = Field(default='user', pattern='^(user|assistant)$')
    note: str = Field(min_length=1, max_length=2000)


@router.patch('/runs/{identity}/judgments/{case_id}')
def judge_evidence(identity: str, case_id: str, value: JudgmentInput):
    from backend.main import retriever
    record = get_run(identity)
    row = next((r for r in record['rows'] if r['case']['id'] == case_id), None)
    if not row: raise HTTPException(404, '题目不存在')
    if not value.note.strip(): raise HTTPException(422, '请填写判断依据')
    if corpus_fingerprint(retriever) != record['corpus_sha256']:
        raise HTTPException(409, '资料已变化，请重新运行评测，不能混合不同版本的证据标注')
    if value.section_id not in {s['id'] for s in retriever.sections}:
        raise HTTPException(422, '证据章节不存在')
    # 可以补标未返回的原文，扩大相关证据池；指标只用本次实际返回的前五条。
    row['judgment_history'].append({'section_id': value.section_id, 'before': row['judgments'].get(value.section_id),
                                   'after': value.model_dump(exclude={'section_id'}), 'at': datetime.now(timezone.utc).isoformat()})
    row['judgments'][value.section_id] = value.model_dump(exclude={'section_id'})
    rescore_row(row); record['summary'] = aggregate(record['rows'])
    record['qrels_sha256'] = annotation_fingerprint(record['rows'])
    pool = {'id': qrels_key(row['case'], record['corpus_sha256']), 'type': 'evaluation_qrels',
            'case_id': row['case']['id'], 'corpus_sha256': record['corpus_sha256'],
            'judgments': row['judgments'], 'updated_at': datetime.now(timezone.utc).isoformat(),
            'source_run_id': identity}
    # 同一事务保存本轮快照和下次可复用的标注；旧记录不会跟随最新标注变化。
    save(record, related=pool); return record


@router.get('/runs/{identity}/pool/{case_id}')
def evidence_pool(identity: str, case_id: str):
    from backend.main import retriever
    record = get_run(identity); row = next((r for r in record['rows'] if r['case']['id'] == case_id), None)
    if not row: raise HTTPException(404, '题目不存在')
    if corpus_fingerprint(retriever) != record['corpus_sha256']: raise HTTPException(409, '资料已变化，请重新评测')
    ids = set(row.get('plan', {}).get('model_ids', []))
    return [{'section_id': s['id'], 'title': s['title'], 'model_id': s['model_id'], 'content': s['content'],
             'reader_url': s.get('reader_url', f"/models/{s['model_id']}?section={s['slug']}")}
            for s in retriever.sections if not ids or s['model_id'] in ids]
