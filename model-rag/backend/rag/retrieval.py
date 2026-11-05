"""三路召回与有限关系遍历；全部引用来自已保存记录。"""
import asyncio
import hashlib
import json
import re
from collections import deque
from contextlib import aclosing
from time import monotonic

import httpx
import jieba
import numpy as np
from rank_bm25 import BM25Okapi

from backend.infra import config
from backend.infra.database import KnowledgeDocument, KnowledgeModel, KnowledgeRelation, KnowledgeSection, KnowledgeSource, records
from backend.infra.knowledge import split_blocks
from backend.rag.index_quality import POLICY, exclusion_reason, python_units
from backend.rag.document_priority import prioritize, priority_day, POLICY as DOCUMENT_PRIORITY_POLICY

jieba.setLogLevel(40)


def tokens(text):
    words = re.findall(r"[a-z][a-z0-9_.-]*|\d+", text.lower())
    words += [w for w in jieba.cut(text) if re.search(r"[\u4e00-\u9fff]", w) and w.strip()]
    return words or ["_"]


def provider_timeout(endpoint, body):
    # 非流式回答要等整段正文；长回答有更长、仍有上限的读取窗口。
    return 120 if endpoint == 'chat/completions' and body.get('max_tokens', 0) > 1600 else 65


async def provider_post(base, key, endpoint, body):
    if not key:
        raise RuntimeError("服务密钥尚未配置")
    if endpoint == 'chat/completions' and body.get('stream') is True:
        # 核对同样使用已有SSE传输；外层有整体时限，取消时关闭正在读取的连接。
        from backend.infra.streaming import stream_completion
        pieces, reason, first = [], None, None
        started = monotonic()
        try:
            async with aclosing(stream_completion(base, key, body)) as events:
                async for item in events:
                    if item['text']:
                        first = first if first is not None else round((monotonic() - started) * 1000)
                        pieces.append(item['text'])
                    reason = item.get('finish_reason') or reason
        except asyncio.CancelledError as exc:
            exc.stream_diagnostics = {'first_text_ms': first, 'received_chars': sum(map(len, pieces))}
            raise
        return {'choices': [{'message': {'content': ''.join(pieces)}, 'finish_reason': reason}],
                'stream_diagnostics': {'first_text_ms': first, 'duration_ms': round((monotonic() - started) * 1000)}}
    async with httpx.AsyncClient(timeout=provider_timeout(endpoint, body)) as client:
        for attempt in range(3):
            response = await client.post(f"{base.rstrip('/')}/{endpoint}", headers={"Authorization": f"Bearer {key}"}, json=body)
            if response.status_code in (429, 502, 503) and attempt < 2:
                await asyncio.sleep(1 + attempt)
                continue
            if response.status_code >= 400:
                raise RuntimeError(f"模型服务返回 HTTP {response.status_code}")
            try:
                result = response.json()
            except ValueError as exc:
                raise RuntimeError("模型服务未返回有效 JSON") from exc
            if not isinstance(result, dict):
                raise RuntimeError("模型服务响应结构无效")
            return result
    raise RuntimeError("模型服务暂不可用")


class Retriever:
    def __init__(self):
        self.models = {m["id"]: m for m in records(KnowledgeModel)}
        all_documents = records(KnowledgeDocument)
        archived = {d['id'] for d in all_documents if d.get('status') == 'archived'}
        all_sections = records(KnowledgeSection)
        self.archived_section_ids = {s['id'] for s in all_sections if s.get('document_id') in archived}
        self.sections = sorted((s for s in all_sections if s['id'] not in self.archived_section_ids), key=lambda s: (self.models[s["model_id"]]["order"], s["order"]))
        self.sources = {s["id"]: s for s in records(KnowledgeSource)}
        self.relations = [edge for edge in records(KnowledgeRelation) if edge.get('status', 'accepted') == 'accepted']
        self.documents = [d for d in all_documents if d['status'] == 'imported']
        self._priority_time = None
        self.document_boost_enabled = True
        from backend.infra.public_sources import document_kinds
        self.document_kinds = document_kinds(self.documents)
        for model in self.models.values():
            model["document_count"] = sum(d["model_id"] == model["id"] for d in self.documents)
            model["searchable"] = model["ready"] or model["document_count"] > 0
        self.chunks, self.index_exclusions = [], []
        self.index_policy = POLICY
        page_parts = {}
        for section in self.sections:
            if section.get('page'):
                page_parts.setdefault((section.get('document_id'), section['page']), {})[section.get('part', 1)] = section
        for section in self.sections:
            if section["slug"] in ("followups", "figures", "sources"):
                continue
            siblings = page_parts.get((section.get('document_id'), section.get('page')), {})
            kind = self.document_kinds.get(section.get('document_id'))
            reason = exclusion_reason(section, kind, siblings.get(section.get('part', 0) - 1), bool(siblings) and section.get('part') == max(siblings))
            if reason:
                self.index_exclusions.append({'section_id': section['id'], 'reason': reason})
                continue
            units = python_units(section['content']) if kind == 'official_code' else None
            contents = [u['content'] for u in units] if units is not None else split_blocks(section['content'])
            for index, content in enumerate(contents):
                self.chunks.append({"id": f"{section['id']}:{index}", "model_id": section["model_id"], "section_id": section["id"], "section_slug": section["slug"], "title": f"{self.models[section['model_id']]['name']} · {section['title']}", "content": content, "parent_content": section["content"], "source_ids": section["source_ids"], "document_id": section.get("document_id"), "locator": section.get("locator"), "origin": section.get("origin", "model_card"), "cloud_allowed": section.get("cloud_allowed", True), "reader_url": section.get("reader_url", f"/models/{section['model_id']}?section={section['slug']}")})
                if section.get('document_id'):
                    self.chunks[-1].update({key: section.get(key) for key in ('chapter', 'page', 'pages', 'paragraph_start', 'paragraph_end', 'paragraph_basis')})
                    self.chunks[-1]['original_file_url'] = section.get('original_file_url') or f"/api/documents/{section['document_id']}/file?inline=true" + (f"#page={section['page']}" if section.get('page') else '')
                if units is not None:
                    self.chunks[-1].update(index_symbol=units[index]['symbol'], index_unit=units[index]['kind'])
                    if units[index]['symbol']:
                        self.chunks[-1]['title'] += ' · ' + units[index]['symbol']
        self.bm25 = BM25Okapi([tokens(c["title"] + "\n" + c["content"]) for c in self.chunks])
        self.texts = [c["title"] + "\n" + c["content"] for c in self.chunks]
        self.fingerprint = hashlib.sha256(json.dumps([config.EMBEDDING_MODEL, self.texts], ensure_ascii=False).encode()).hexdigest()
        from backend.infra.access import current_actor
        actor = current_actor.get()
        self.cache_path = (config.DATA_ROOT / 'indices' / (actor.id or 'public') / 'embeddings.npz') if actor is not None else config.DATA_ROOT / 'embeddings.npz'
        self.vectors = None
        self.vector_cache = {}
        self.chunk_keys = [hashlib.sha256((config.EMBEDDING_MODEL + "\0" + text).encode()).hexdigest() for text in self.texts]
        self.index_lock = asyncio.Lock()
        self._load_cache()

    @property
    def priority_time(self):
        return self._priority_time or priority_day()

    @priority_time.setter
    def priority_time(self, value):
        self._priority_time = value

    @property
    def document_priority_snapshot(self):
        return {'policy': DOCUMENT_PRIORITY_POLICY, 'at': self.priority_time.isoformat(),
                'documents': sorted((d['id'], d.get('retrieval_priority', 1.0), d.get('retrieval_added_at') or d.get('imported_at')) for d in self.documents)}

    def _load_cache(self):
        path = self.cache_path if self.cache_path.exists() else config.DATA_ROOT / 'embeddings.npz'
        if path.exists():
            try:
                with np.load(path, allow_pickle=False) as cache:
                    if "keys" in cache:
                        self.vector_cache = {str(key): vector.copy() for key, vector in zip(cache["keys"], cache["vectors"], strict=True)}
                    elif str(cache["fingerprint"]) == self.fingerprint and len(cache["vectors"]) == len(self.chunks):
                        self.vector_cache = dict(zip(self.chunk_keys, cache["vectors"].copy(), strict=True))
                    elif "fingerprint" in cache:
                        # 迁移首版缓存：新资料不影响已有两张模型卡的向量复用。
                        originals = [i for i, chunk in enumerate(self.chunks) if chunk["origin"] == "model_card"]
                        fingerprint = hashlib.sha256(json.dumps([config.EMBEDDING_MODEL, [self.texts[i] for i in originals]], ensure_ascii=False).encode()).hexdigest()
                        if str(cache["fingerprint"]) == fingerprint and len(cache["vectors"]) == len(originals):
                            self.vector_cache = dict(zip([self.chunk_keys[i] for i in originals], cache["vectors"].copy(), strict=True))
            except (ValueError, OSError, KeyError):
                self.vector_cache = {}
        allowed = set(self.chunk_keys)
        self.vector_cache = {key:value for key,value in self.vector_cache.items() if key in allowed}
        self._activate_vectors()

    def _activate_vectors(self):
        available = [self.vector_cache.get(key) if chunk["cloud_allowed"] else None for key, chunk in zip(self.chunk_keys, self.chunks)]
        sample = next((vector for vector in available if vector is not None), None)
        self.vector_mask = np.array([vector is not None for vector in available])
        self.vectors = np.stack([vector if vector is not None else np.zeros_like(sample) for vector in available]) if sample is not None else None

    def index_status(self):
        return {"indexed_chunks": int(self.vector_mask.sum()), "eligible_chunks": sum(c["cloud_allowed"] for c in self.chunks), "local_only_chunks": sum(not c["cloud_allowed"] for c in self.chunks),
                "index_policy": self.index_policy, "excluded_sections": len(self.index_exclusions)}

    async def embed(self, texts, query=False):
        if query:
            texts = ["Instruct: Retrieve deep learning passages that explain the model, its inputs, outputs, architecture or implementation.\nQuery:" + t for t in texts]
        result = await provider_post(config.EMBEDDING_BASE, config.EMBEDDING_KEY, "embeddings", {"model": config.EMBEDDING_MODEL, "input": texts, "encoding_format": "float"})
        data = sorted(result.get("data", []), key=lambda item: item["index"])
        if len(data) != len(texts):
            raise RuntimeError("嵌入服务返回条数不一致")
        vectors = np.asarray([item["embedding"] for item in data], dtype=np.float32)
        if vectors.ndim != 2 or not np.isfinite(vectors).all():
            raise RuntimeError("嵌入服务返回无效向量")
        return vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-8)

    async def build_index(self):
        async with self.index_lock:
            missing = [i for i, chunk in enumerate(self.chunks) if chunk["cloud_allowed"] and self.chunk_keys[i] not in self.vector_cache]
            additions = {}
            for offset in range(0, len(missing), 12):
                indices = missing[offset:offset + 12]
                vectors = await self.embed([self.texts[i] for i in indices])
                additions.update(zip([self.chunk_keys[i] for i in indices], vectors, strict=True))
            updated = {**self.vector_cache, **additions}
            keys = list(dict.fromkeys(self.chunk_keys[i] for i, chunk in enumerate(self.chunks) if chunk["cloud_allowed"] and self.chunk_keys[i] in updated))
            if not keys:
                return {"chunks": len(self.chunks), "cached": True, "added_chunks": 0, **self.index_status()}
            if len({updated[key].shape for key in keys}) != 1:
                raise RuntimeError("新增向量与缓存维度不一致，请检查嵌入模型配置。")
            vectors = np.stack([updated[key] for key in keys])
            self.cache_path.parent.mkdir(parents=True,exist_ok=True)
            temporary = self.cache_path.with_name('embeddings.tmp.npz')
            np.savez_compressed(temporary, keys=np.array(keys), vectors=vectors)
            temporary.replace(self.cache_path)
            self.vector_cache = updated
            self._activate_vectors()
            return {"chunks": len(self.chunks), "cached": not missing, "added_chunks": len(missing), "dimension": int(vectors.shape[1]), **self.index_status()}

    def identify_models(self, query, focus=None):
        # 同一位置优先完整名称，如 Vision Transformer 不额外命中内部的 Transformer。
        lower = query.lower()
        matches = [(match.start(), match.end(), model["id"])
                   for model in self.models.values() for alias in model["aliases"]
                   for match in re.finditer((r"(?<![a-z0-9])" + re.escape(alias.lower()) + r"(?![a-z0-9])")
                                           if alias.lower() in {"vit", "clip"} else re.escape(alias.lower()), lower)]
        explicit = {mid for start, end, mid in matches if not any(
            other != mid and left <= start and end <= right and right - left > end - start
            for left, right, other in matches)}
        found = [model["id"] for model in self.models.values() if model["id"] in explicit]
        if focus in self.models and focus not in found:
            found.append(focus)
        return found

    def paths(self, roots, max_hops=2):
        """只沿有来源的关系遍历，路径留存方向；不把邻接当成因果。"""
        adjacency = {}
        for edge in self.relations:
            adjacency.setdefault(edge["source"], []).append((edge["target"], edge))
            adjacency.setdefault(edge["target"], []).append((edge["source"], edge))
        paths = []
        for root in roots:
            queue = deque([(root, [], {root})])
            while queue and len(paths) < 18:
                node, path, visited = queue.popleft()
                if len(path) == max_hops:
                    continue
                for neighbor, edge in adjacency.get(node, []):
                    if neighbor in visited:
                        continue
                    extended = path + [edge]
                    paths.append({"root": root, "hops": len(extended), "edges": extended, "source_ids": sorted({s for item in extended for s in item["source_ids"]})})
                    queue.append((neighbor, extended, visited | {neighbor}))
        return paths

    async def search(self, query, focus=None, section_id=None, top_k=5, use_rerank=True, strict_focus=False, allow_remote=True, query_vector=None, allow_vectors=True, prefer_sources=(), source_kinds=(), excluded_source_kinds=(), retrieval_channels=None, source_boost=True, allowed_section_ids=None, document_boost=True):
        priority_time = self.priority_time
        warnings, channels = [], {}
        enabled = {'关键词', '实体/章节', '向量'} if retrieval_channels is None else set(retrieval_channels)
        if not enabled or not enabled <= {'关键词', '实体/章节', '向量'}:
            raise ValueError('检索渠道设置无效')
        matched = self.identify_models(query, focus)
        if strict_focus and focus:
            matched = [focus]
        eligible = {i for i, chunk in enumerate(self.chunks)
                    if (not strict_focus or not focus or chunk['model_id'] == focus)
                    and (allowed_section_ids is None or chunk['section_id'] in allowed_section_ids)
                    and (not source_kinds or self.document_kinds.get(chunk.get('document_id')) in source_kinds)
                    and self.document_kinds.get(chunk.get('document_id')) not in excluded_source_kinds}
        if allowed_section_ids is not None and not eligible:
            return {'evidence': [], 'paths': [], 'warnings': [], 'channels': {}, 'reranked': False}
        keyword_scores = self.bm25.get_scores(tokens(query))
        if '关键词' in enabled:
            channels["关键词"] = [int(i) for i in np.argsort(-keyword_scores) if i in eligible and keyword_scores[i] > 0][:16]
        entity_scores = []
        for i, chunk in enumerate(self.chunks):
            score = (2 if chunk["model_id"] in matched else 0) + (3 if section_id == chunk["section_id"] else 0)
            if score and i in eligible:
                entity_scores.append((i, score + float(max(keyword_scores[i], 0)) * 0.05))
        if '实体/章节' in enabled:
            channels["实体/章节"] = [i for i, _ in sorted(entity_scores, key=lambda item: -item[1])][:16]
        local_only_scope = any((not matched or c["model_id"] in matched) and not c["cloud_allowed"] for c in self.chunks)
        if '向量' in enabled and self.vectors is not None and not local_only_scope and allow_remote and allow_vectors:
            try:
                q = query_vector if query_vector is not None else (await self.embed([query], query=True))[0]
                if q.shape[0] != self.vectors.shape[1]:
                    raise RuntimeError("问题与文档向量维度不一致，需要重建索引")
                scores = self.vectors @ q
                channels["向量"] = [int(i) for i in np.argsort(-scores) if i in eligible and self.vector_mask[i] and scores[i] > 0.2][:16]
            except (RuntimeError, httpx.HTTPError):
                warnings.append("本轮向量检索暂不可用，已使用其他检索渠道。")
        elif '向量' in enabled:
            warnings.append("当前检索范围仅允许本地处理，本轮使用本地检索。" if local_only_scope or not allow_remote else "向量索引或服务尚未就绪，本轮使用关键词与实体/章节检索。")
        scores, hits = {}, {}
        for channel, ranking in channels.items():
            for rank, index in enumerate(ranking, 1):
                scores[index] = scores.get(index, 0) + 1 / (60 + rank)
                hits.setdefault(index, []).append(channel)
        order = sorted(scores, key=lambda i: -scores[i])
        preferred_kinds = set(prefer_sources) | set(source_kinds)
        if re.search(r'官方|源码|\.encode_|\.forward', query):
            preferred_kinds.add('official_code')
        if re.search(r'原论文|论文|paper', query, re.I):
            preferred_kinds.add('paper')
        if source_boost and preferred_kinds:
            source_order = []
            sections_seen = set()
            for i in sorted(eligible, key=lambda index: -scores.get(index, 0)):
                chunk = self.chunks[i]
                if matched and chunk['model_id'] not in matched:
                    continue
                kind = self.document_kinds.get(chunk.get('document_id'))
                if kind not in preferred_kinds or (kind == 'official_code' and '```' not in chunk['parent_content']):
                    continue
                if chunk['section_id'] not in sections_seen:
                    source_order.append(i)
                    sections_seen.add(chunk['section_id'])
            channels['原始来源'] = source_order[:8]
            for rank, i in enumerate(channels['原始来源'], 1):
                scores[i] = scores.get(i, 0) + 1 / (60 + rank)
                hits.setdefault(i, []).append('原始来源')
            order = list(dict.fromkeys([*channels['原始来源'], *order]))
        if focus and focus in self.models and not self.models[focus]["searchable"] and len(matched) <= 1:
            order = []
        # 当前两张卡很小，优先保持所问模型范围；比较问题允许多个对象。
        if matched:
            order = [i for i in order if self.chunks[i]["model_id"] in matched]
        ranking = [{'_index': i, 'section_id': self.chunks[i]['section_id'], 'document_id': self.chunks[i].get('document_id'),
                    'score': scores[i], 'retrieval_relevant': bool(keyword_scores[i] > 0 or i in channels.get('向量', []))} for i in order]
        if document_boost:
            ranking = prioritize(ranking, self.documents, priority_time,
                protected={self.chunks[i]['section_id'] for i in channels.get('原始来源', [])})
            order = [item['_index'] for item in ranking]
        ranking_by_index = {item['_index']: item for item in ranking}
        # 子片段用于匹配，重排与Top-K名额按父章节计；同章多个函数窗口不能重复占位。
        parents, distinct = set(), []
        for index in order:
            parent = self.chunks[index]['section_id']
            if parent not in parents:
                parents.add(parent); distinct.append(index)
        order = distinct[:18]
        reranked = False
        cloud_safe = all(self.chunks[i]["cloud_allowed"] for i in order)
        if use_rerank and order and config.RERANK_KEY and cloud_safe and allow_remote:
            try:
                result = await provider_post(config.RERANK_BASE, config.RERANK_KEY, "rerank", {"model": config.RERANK_MODEL, "query": query, "documents": [self.texts[i] for i in order], "top_n": len(order), "return_documents": False})
                ranking = result.get("results", [])
                valid = [item for item in ranking if isinstance(item.get("index"), int) and 0 <= item["index"] < len(order)]
                if valid:
                    rerank_scores = {order[item['index']]: item.get('relevance_score') for item in valid}
                    order = list(dict.fromkeys([order[item['index']] for item in valid] + order))
                    if document_boost:
                        ranking = prioritize([{**ranking_by_index[i], 'rerank_score': rerank_scores.get(i)} for i in order],
                                             self.documents, priority_time, score_key='rerank_score')
                        order = [item['_index'] for item in ranking]
                        ranking_by_index = {item['_index']: item for item in ranking}
                    reranked = True
            except (RuntimeError, httpx.HTTPError):
                warnings.append("本轮重排服务暂不可用，保留融合排序。")
        elif order and (not cloud_safe or not allow_remote):
            warnings.append("候选资料含本地资料，本轮未发送正文到云端重排。")
        selected = []
        for index in order:
            chunk = dict(self.chunks[index])
            chunk['retrieval_excerpt'] = chunk['content']
            # 检索子片段命中后补齐所属章节，避免公式与条件分离。
            chunk["content"] = chunk.pop("parent_content")
            chunk["channels"] = hits[index]
            chunk["score"] = round(scores[index], 5)
            chunk['retrieval_relevant'] = ranking_by_index[index]['retrieval_relevant']
            if 'retrieval_adjustment' in ranking_by_index[index]:
                chunk['retrieval_adjustment'] = ranking_by_index[index]['retrieval_adjustment']
            chunk["sources"] = [self.sources[s] for s in chunk["source_ids"] if s in self.sources]
            if any(item["section_id"] == chunk["section_id"] for item in selected):
                continue
            selected.append(chunk)
            if len(selected) == top_k:
                break
        graph_intent = any(word in query for word in ["关系", "联系", "相似", "共同", "区别", "差异", "比较", "对比", "关联", "组成", "模块", "结构", "采用", "哪些"])
        # 来源限定同样约束生成上下文：目录路径可能含教学卡，不能绕过仅官方正文筛选。
        graph_paths = self.paths(matched) if graph_intent and not source_kinds else []
        return {"evidence": selected, "paths": graph_paths, "warnings": warnings, "channels": {name: len(items) for name, items in channels.items()}, "reranked": reranked}
