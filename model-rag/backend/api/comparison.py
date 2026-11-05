"""固定资料、规划、向量和标注池的检索对照；不生成答案或冒充盲测。"""
import asyncio
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from time import perf_counter
from uuid import uuid4
from pathlib import Path

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.infra import config
from backend.infra.database import KnowledgeAnalysis, records
from backend.api.evaluation import cases, corpus_fingerprint, save, select_cases
from backend.generation.evaluation_metrics import aggregate, annotation_fingerprint, qrels_key, rescore_row, reuse_judgments
from backend.rag.pipeline import execute_plan
from backend.rag.planning import rule_plan

import backend.infra.public_sources
import backend.rag.graph
import backend.rag.pipeline
import backend.rag.planning
import backend.rag.retrieval

# 参与检索对照指纹的源码文件。分层重构后它们不在同一目录，按模块定位而不是按同目录文件名。
SOURCE_MODULES = {
    'comparison.py': None,
    'retrieval.py': backend.rag.retrieval,
    'pipeline.py': backend.rag.pipeline,
    'planning.py': backend.rag.planning,
    'graph.py': backend.rag.graph,
    'public_sources.py': backend.infra.public_sources,
}

router = APIRouter(prefix='/api/evaluation/comparisons', tags=['检索方案对照'])
comparison_lock = asyncio.Lock()
VARIANTS = {
    'dense': {'label': '纯向量', 'channels': ['向量'], 'graph': False},
    'hybrid': {'label': '混合检索', 'channels': ['关键词', '向量'], 'graph': False},
    'hybrid_graph': {'label': '混合检索＋图谱扩展', 'channels': ['关键词', '向量'], 'graph': True},
}


def prepare_plan(case, active):
    previous = []
    for turn in case.get('prior_turns', []):
        prior = rule_plan(turn['question'], active, turn.get('model_id'), previous=previous)
        previous.extend([{'role': 'user', 'content': turn['question'], 'metadata': {}},
                         {'role': 'assistant', 'content': '', 'metadata': {'model_id': prior['model_id'], 'plan': prior}}])
    return rule_plan(case['question'], active, case.get('model_id'), previous=previous)


class SharedEmbeddings:
    def __init__(self, active):
        self.active, self.cache, self.calls = active, {}, 0

    async def embed(self, texts, query=False):
        if not query:
            raise RuntimeError('对照任务不重建文档索引')
        missing = list(dict.fromkeys(t for t in texts if t not in self.cache))
        if missing:
            values = await self.active.embed(missing, query=True)
            self.cache.update(zip(missing, values, strict=True))
            self.calls += 1
        import numpy as np
        return np.stack([self.cache[t] for t in texts])


class ControlledRetriever:
    document_boost_enabled = False
    def __init__(self, active, variant, embeddings):
        self.active, self.variant, self.embeddings = active, variant, embeddings

    def __getattr__(self, name):
        return getattr(self.active, name)

    async def embed(self, texts, query=False):
        return await self.embeddings.embed(texts, query=query)

    async def search(self, query, *args, **kwargs):
        kwargs['document_boost'] = False
        if not kwargs.get('allow_remote', True) or not kwargs.get('allow_vectors', True):
            raise RuntimeError('本题向量调用未执行，不能以本地降级结果代替对照')
        if kwargs.get('query_vector') is None:
            kwargs['query_vector'] = (await self.embed([query], query=True))[0]
        result = await self.active.search(query, *args, **kwargs, retrieval_channels=VARIANTS[self.variant]['channels'], source_boost=False)
        if '向量' not in result['channels']:
            raise RuntimeError('本题向量检索失败；本轮对照无效')
        # 基础检索返回的目录路径不是实际执行的图跳；只保留 execute_plan 的证据驱动路径。
        result['paths'] = []
        return result


def rescore_comparison(record):
    output = deepcopy(record)
    for row in output['rows']:
        for result in row['results'].values():
            result['case'], result['judgments'] = row['case'], row['judgments']
            rescore_row(result)
    output['summaries'] = {name: aggregate([row['results'][name] for row in output['rows']]) for name in VARIANTS}
    output['qrels_sha256'] = annotation_fingerprint(output['rows'])
    output['complete'] = all(not result.get('error') for row in output['rows'] for result in row['results'].values())
    return output


def get_comparison(identity):
    record = next((r for r in records(KnowledgeAnalysis) if r['id'] == identity and r.get('type') == 'retrieval_comparison'), None)
    if not record:
        raise HTTPException(404, '检索对照记录不存在')
    return rescore_comparison(record)


class ComparisonInput(BaseModel):
    case_ids: list[str] = Field(min_length=1, max_length=12)


@router.get('')
def list_comparisons():
    found = sorted((r for r in records(KnowledgeAnalysis) if r.get('type') == 'retrieval_comparison'), key=lambda r: r['created_at'], reverse=True)
    return [{'id': r['id'], 'created_at': r['created_at'], 'case_count': len(r['rows']), 'complete': r['complete']} for r in found[:20]]


@router.get('/{identity}')
def comparison_detail(identity: str):
    return get_comparison(identity)


@router.post('')
async def run_comparison(value: ComparisonInput):
    from backend import main
    if comparison_lock.locked():
        raise HTTPException(409, '已有检索对照在运行')
    async with comparison_lock:
        active = main.retriever
        dataset = select_cases(cases(), value.case_ids)
        if any(not c.get('cloud_allowed', True) for c in dataset):
            raise HTTPException(422, '选中题目中含有未允许 API 处理的问题')
        index = active.index_status()
        if active.vectors is None or index['indexed_chunks'] != index['eligible_chunks']:
            raise HTTPException(409, '文档向量未完整就绪，请先完成索引')
        corpus = corpus_fingerprint(active)
        pools = {r['id']: r for r in records(KnowledgeAnalysis) if r.get('type') == 'evaluation_qrels'}
        record = {'id': 'rc-' + uuid4().hex[:24], 'type': 'retrieval_comparison',
                  'created_at': datetime.now(timezone.utc).isoformat(), 'protocol': 'controlled-retrieval-v1',
                  'corpus_sha256': corpus, 'embedding_model': config.EMBEDDING_MODEL,
                  'dataset_sha256': hashlib.sha256(json.dumps(dataset, ensure_ascii=False, sort_keys=True).encode()).hexdigest(),
                  'code_sha256': {name: hashlib.sha256(Path(module.__file__ if module else __file__).read_bytes()).hexdigest()
                                  for name, module in SOURCE_MODULES.items()},
                  'variants': VARIANTS, 'rows': [], 'controls': {
                      'top_k': 5, 'per_channel_limit': 16, 'fused_parent_limit': 18,
                      'planning': 'same_deterministic_rules', 'rerank': False,
                      'entity_channel': False, 'source_boost': False, 'definition_expansion': False,
                      'query_vectors': 'shared_within_comparison', 'generation': False,
                      'latency_note': '复用查询向量，耗时仅供排错，不是冷启动延迟对比。',
                      'scope_note': '比较固定领域规划下的召回渠道与顺序图扩展；不是完全移除图知识的端到端消融，也不等于线上完整配置。'}}
        embeddings = SharedEmbeddings(active)
        for case in dataset:
            plan = prepare_plan(case, active)
            row = {'case': case, 'plan': plan, 'judgments': reuse_judgments(case, corpus, pools), 'judgment_history': [], 'results': {}}
            for name in VARIANTS:
                started = perf_counter()
                result = {'evidence': [], 'metrics': None}
                try:
                    adapter = ControlledRetriever(active, name, embeddings)
                    found = await asyncio.wait_for(execute_plan(deepcopy(plan), adapter, graph_enabled=VARIANTS[name]['graph'],
                                                               definition_expansion=False, rerank_enabled=False), timeout=100)
                    result.update({key: found[key] for key in ('evidence', 'steps', 'paths', 'channels', 'warnings', 'stop_reason', 'reranked')})
                    if found['reranked']:
                        raise RuntimeError('对照意外执行了重排，本轮无效')
                except (RuntimeError, TimeoutError, ValueError, httpx.HTTPError) as exc:
                    result['error'] = str(exc) if isinstance(exc, RuntimeError) else type(exc).__name__
                result['seconds'] = round(perf_counter() - started, 3)
                row['results'][name] = result
            record['rows'].append(row)
        if main.retriever is not active or corpus_fingerprint(active) != corpus:
            raise HTTPException(409, '运行过程中资料发生变化，请重新进行对照')
        record['embedding_requests'] = embeddings.calls
        record = rescore_comparison(record)
        save(record)
        return record


class ComparisonJudgment(BaseModel):
    section_id: str = Field(min_length=1, max_length=160)
    label: str = Field(pattern='^(direct|supporting|irrelevant|unjudged)$')
    origin: str = Field(default='user', pattern='^(user|assistant)$')
    note: str = Field(min_length=1, max_length=2000)
    quote: str = Field(default='', max_length=4000)


@router.patch('/{identity}/judgments/{case_id}')
def judge_comparison(identity: str, case_id: str, value: ComparisonJudgment):
    from backend.main import retriever
    record = get_comparison(identity)
    if corpus_fingerprint(retriever) != record['corpus_sha256']:
        raise HTTPException(409, '资料已变化，请重新进行对照')
    row = next((r for r in record['rows'] if r['case']['id'] == case_id), None)
    if not row:
        raise HTTPException(404, '题目不存在')
    section = next((s for s in retriever.sections if s['id'] == value.section_id), None)
    if not section or not value.note.strip() or (value.quote and value.quote not in section['content']):
        raise HTTPException(422, '请提供有效章节、判断依据和原文摘录')
    before = row['judgments'].get(value.section_id)
    if value.origin == 'assistant' and before and before.get('origin') == 'user':
        raise HTTPException(409, '辅助标注不能覆盖用户判断')
    mark = value.model_dump(exclude={'section_id'})
    row['judgments'][value.section_id] = mark
    row['judgment_history'].append({'section_id': value.section_id, 'before': before, 'after': mark, 'at': datetime.now(timezone.utc).isoformat()})
    record = rescore_comparison(record)
    # 三种方案使用同一个判断池，补标后同步重算；既有评测运行不被改写。
    pool = {'id': qrels_key(row['case'], record['corpus_sha256']), 'type': 'evaluation_qrels',
            'case_id': case_id, 'corpus_sha256': record['corpus_sha256'], 'judgments': row['judgments'],
            'updated_at': datetime.now(timezone.utc).isoformat(), 'source_run_id': identity}
    save(record, related=pool)
    return record
