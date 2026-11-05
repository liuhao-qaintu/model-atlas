"""执行有界检索计划：并列比较、依据驱动的顺序关系扩展、合并后只重排一次。"""
from time import perf_counter

import httpx

from backend.infra import config
from backend.rag.planning import MAX_QUERIES, mentions, scope_is_cloud_safe
from backend.rag.retrieval import provider_post
from backend.rag.graph import proven_edges
from backend.infra.public_sources import expand_definitions
from backend.rag.source_scope import strict_method_sections
from backend.rag.document_priority import prioritize, priority_day
import re

MAX_HOPS = 2
MAX_EVIDENCE = 5


def step_record(number, query, result, seen, kind="initial", depends_on=None, anchor=None):
    ids = [item["section_id"] for item in result["evidence"]]
    return {"id": f"s{number}", "kind": kind, "query": query["query"], "model_id": query["model_id"],
            "found_count": len(ids), "new_evidence_count": sum(sid not in seen for sid in ids), "evidence_ids": ids,
            "channels": result["channels"], "status": "found" if ids else "missing", "depends_on": depends_on,
            "anchor": anchor, "references": [], 'definition_expansions': result.get('definition_expansions', [])}


def next_edge(retriever, current, visited, evidence, question, model_ids=None, preferred_id=None):
    candidates = []
    for edge in proven_edges(retriever):
        if preferred_id and edge['id'] != preferred_id:
            continue
        if model_ids and edge['owner_model_id'] not in model_ids:
            continue
        if current not in (edge["source"], edge["target"]) or not edge["source_ids"] or not all(s in retriever.sources for s in edge["source_ids"]):
            continue
        neighbor = edge["target"] if current == edge["source"] else edge["source"]
        if neighbor in visited:
            continue
        current_name = edge["source_name"] if current == edge["source"] else edge["target_name"]
        neighbor_name = edge["target_name"] if current == edge["source"] else edge["source_name"]
        if neighbor in retriever.models and model_ids and neighbor not in model_ids:
            continue
        proofs = [item for proof in edge['evidence'] for item in evidence if item['section_id'] == proof['section_id'] and proof['quote'] in item['content']]
        if not proofs:
            continue
        priority = 10 * mentions(question, neighbor_name) + (3 if edge["relation"] in {"使用模块", "包含算子", "采用机制", "本卡教学实例"} else 0)
        candidates.append((priority, edge, neighbor, current_name, neighbor_name, proofs[0]))
    return max(candidates, key=lambda item: item[0]) if candidates else None


def merge_evidence(results):
    merged = {}
    for step, result in results:
        for rank, item in enumerate(result["evidence"], 1):
            sid = item["section_id"]
            if sid not in merged:
                merged[sid] = {**item, "step_ids": [], "score": 0, "channels": []}
            candidate = merged[sid]
            candidate['retrieval_relevant'] = candidate.get('retrieval_relevant', False) or item.get('retrieval_relevant', False)
            candidate["score"] += 1 / (60 + rank)
            candidate["step_ids"].append(step["id"])
            candidate["channels"] = list(dict.fromkeys(candidate["channels"] + item["channels"]))
    return sorted(merged.values(), key=lambda item: -item["score"])


async def final_ranking(query, candidates, warnings, allow_remote):
    if not candidates or not config.RERANK_KEY:
        return candidates, False
    if not allow_remote or any(not c.get("cloud_allowed", True) for c in candidates):
        warnings.append("本轮合并候选包含仅在本地处理的资料，未发送到云端重排。")
        return candidates, False
    try:
        response = await provider_post(config.RERANK_BASE, config.RERANK_KEY, "rerank", {"model": config.RERANK_MODEL, "query": query,
                                       "documents": [c["title"] + "\n" + c.get('retrieval_excerpt', c["content"])[:4500] for c in candidates], "top_n": len(candidates), "return_documents": False})
        indices = list(dict.fromkeys(item["index"] for item in response.get("results", []) if isinstance(item, dict) and type(item.get("index")) is int and 0 <= item["index"] < len(candidates)))
        if not indices:
            raise RuntimeError("无可用重排结果")
        indices += [i for i in range(len(candidates)) if i not in indices]
        scores = {item['index']: item.get('relevance_score') for item in response.get('results', [])
                  if isinstance(item, dict) and type(item.get('index')) is int and 0 <= item['index'] < len(candidates)}
        return [{**candidates[i], 'rerank_score': scores.get(i)} for i in indices], True
    except (RuntimeError, httpx.HTTPError, TypeError, KeyError):
        warnings.append("本轮合并候选的重排服务暂不可用，保留融合排序。")
        return candidates, False


def retain_evidence(candidates, steps, model_ids, document_kinds=None, prefer_official=False):
    """优先覆盖对象、每步和关系锚点，再填充相关片段；全轮最多五条。"""
    required = []
    for step in steps:
        for proof in (step.get("anchor") or {}).get("proofs", []):
            required.append(proof["section_id"])
        for definition in step.get('definition_expansions', []):
            required.extend([definition['trigger_section_id'], definition['definition_section_id']])
    for mid in model_ids:
        item = next((c for c in candidates if c['model_id'] == mid and (document_kinds or {}).get(c.get('document_id')) == 'official_code'), None) if prefer_official else None
        item = item or next((c for c in candidates if c["model_id"] == mid), None)
        if item:
            required.append(item["section_id"])
    for step in steps:
        if not any(sid in step["evidence_ids"] for sid in required):
            item = next((c for c in candidates if c["section_id"] in step["evidence_ids"]), None)
            if item:
                required.append(item["section_id"])
    available = {item['section_id'] for item in candidates}
    selected_ids = list(dict.fromkeys(sid for sid in required if sid in available))[:MAX_EVIDENCE]
    for item in candidates:
        if item["section_id"] not in selected_ids and len(selected_ids) < MAX_EVIDENCE:
            selected_ids.append(item["section_id"])
    return [{**item, "score": round(item["score"], 5)} for item in candidates if item["section_id"] in selected_ids]


def attach_references(steps, evidence):
    for step in steps:
        step["references"] = [{"number": index, "title": item["title"], "url": item["reader_url"]} for index, item in enumerate(evidence, 1) if item["section_id"] in step["evidence_ids"]]
        for proof in (step.get("anchor") or {}).get("proofs", []):
            proof["reference"] = next((index for index, item in enumerate(evidence, 1) if item["section_id"] == proof["section_id"]), None)
        if step["found_count"] and not step["references"]:
            step["status"] = "not_retained"


async def execute_plan(plan, retriever, *, graph_enabled=True, definition_expansion=True, rerank_enabled=True):
    started = perf_counter()
    warnings = list(plan["warnings"])
    if plan["clarification"]:
        return {"evidence": [], "paths": [], "warnings": warnings, "channels": {}, "reranked": False, "steps": [], "stop_reason": "需要明确追问对象", "duration_ms": 0}
    complex_query = plan["mode"] != "single"
    allowed_sections = strict_method_sections(plan.get('source_scope', {}), retriever)
    safe_scope = all(s.get('cloud_allowed', not bool(s.get('document_id'))) for s in retriever.sections if s['id'] in allowed_sections) if allowed_sections is not None else scope_is_cloud_safe(retriever, plan['model_ids'])
    allow_remote = plan.get("remote_allowed", True) and safe_scope
    method_filter = {} if allowed_sections is None else {'allowed_section_ids': allowed_sections}
    if allowed_sections == set():
        warnings.append('当前没有符合所限方法与版本的已核验正文，未放宽到整个类或其他版本。')
    results, seen, paths = [], set(), []
    vectors, allow_vectors = None, True
    # 并列比较的初始子问题可批量嵌入；关系扩展必须等前一步证据返回。
    if len(plan["queries"]) > 1 and retriever.vectors is not None and allow_remote and allowed_sections != set():
        try:
            vectors = await retriever.embed([q["query"] for q in plan["queries"]], query=True)
        except (RuntimeError, httpx.HTTPError):
            allow_vectors = False
            warnings.append("子问题批量向量化暂不可用，继续使用本地召回。")
    implementation_scope = plan.get('source_scope', {}).get('kind') == 'implementation'
    prefer_sources = ('official_code',) if implementation_scope or plan.get('relation_route') else ()
    for index, query in enumerate(plan["queries"]):
        query_filter = method_filter
        if query.get('document_id'):
            document_sections = {s['id'] for s in retriever.sections if s.get('document_id') == query['document_id']}
            query_filter = {'allowed_section_ids': document_sections if allowed_sections is None else allowed_sections & document_sections}
        result = await retriever.search(query["query"], query["model_id"], plan["section_id"], top_k=8 if complex_query else MAX_EVIDENCE,
                                        use_rerank=rerank_enabled and not complex_query, strict_focus=True, allow_remote=allow_remote,
                                        query_vector=vectors[index] if vectors is not None else None, allow_vectors=allow_vectors,
                                        prefer_sources=prefer_sources, source_kinds=plan.get('source_kinds', ()),
                                        excluded_source_kinds=plan.get('excluded_source_kinds', ()), document_boost=not complex_query, **query_filter)
        if definition_expansion and allowed_sections is None and (not plan.get('source_kinds') or 'official_code' in plan['source_kinds']):
            result = expand_definitions(retriever, result, 8 if complex_query else MAX_EVIDENCE)
        step = step_record(len(results) + 1, query, result, seen, kind="comparison" if plan["mode"] == "comparison" else "initial")
        results.append((step, result)); seen.update(step["evidence_ids"]); warnings.extend(result["warnings"])
    stop_reason = "单次检索完成" if plan["mode"] == "single" else "各对象检索完成" if plan["mode"] == "comparison" else "未找到由当前正文支持的后续关系，停止扩展"
    if plan["mode"] == "relation" and not graph_enabled:
        stop_reason = '本对照方案未启用关系扩展'
    if graph_enabled and plan["mode"] == "relation" and plan["model_id"]:
        root = current = search_model = plan["model_id"]
        visited, used_edges = {root}, []
        while len(results) < MAX_QUERIES and len(used_edges) < MAX_HOPS:
            previous_step, previous_result = results[-1]
            route = plan.get('relation_route', [])
            preferred = route[len(used_edges)]['edge_id'] if len(used_edges) < len(route) else None
            choice = next_edge(retriever, current, visited, previous_result["evidence"], plan["original_query"] + " " + plan["normalized_query"], plan['model_ids'], preferred)
            if not choice:
                break
            _, edge, neighbor, current_name, neighbor_name, proof = choice
            anchor = {"edge": edge, "from_node": current, "to_node": neighbor, "from_name": current_name, "to_name": neighbor_name,
                      "proofs": [{"section_id": proof["section_id"], "title": proof["title"], "url": proof["reader_url"]}]}
            if neighbor in retriever.models and neighbor in plan['model_ids']:
                search_model = neighbor
            query = {"model_id": search_model, "query": f"{retriever.models[search_model]['name']}：{current_name} {edge['relation']} {neighbor_name}。{plan['normalized_query']}"[:1400]}
            result = await retriever.search(query["query"], search_model, top_k=8, use_rerank=False, strict_focus=True, allow_remote=allow_remote,
                                            prefer_sources=prefer_sources, source_kinds=plan.get('source_kinds', ()),
                                            excluded_source_kinds=plan.get('excluded_source_kinds', ()), document_boost=False, **method_filter)
            if definition_expansion and allowed_sections is None and (not plan.get('source_kinds') or 'official_code' in plan['source_kinds']):
                result = expand_definitions(retriever, result, 8)
            result["evidence"] = [item for item in result["evidence"] if mentions(item["content"], neighbor_name)]
            kept_ids = {item['section_id'] for item in result['evidence']}
            result['definition_expansions'] = [item for item in result.get('definition_expansions', [])
                                               if {item['trigger_section_id'], item['definition_section_id']} <= kept_ids]
            step = step_record(len(results) + 1, query, result, seen, kind="graph_hop", depends_on=previous_step["id"], anchor=anchor)
            results.append((step, result)); seen.update(step["evidence_ids"]); warnings.extend(result["warnings"])
            if not result["evidence"]:
                warnings.append(f"关系目录包含 {neighbor_name}，但未找到对应正文片段，本轮不继续扩展。")
                stop_reason = "后续对象缺少正文依据"
                break
            used_edges.append(edge)
            paths.append({"root": root, "hops": len(used_edges), "edges": list(used_edges), "source_ids": sorted({sid for item in used_edges for sid in item["source_ids"]})})
            current = neighbor; visited.add(neighbor)
        if len(used_edges) == MAX_HOPS:
            stop_reason = "已达到两跳、三步的本轮上限"
        if plan.get('relation_route') and len(used_edges) < len(plan['relation_route']):
            warnings.append('当前检索正文未支持完整跨模型路径，已停止扩展；尚未覆盖的对象不能用其他模型资料替代。')
    steps = [step for step, _ in results]
    if complex_query:
        candidates = merge_evidence(results)
        candidates, reranked = await final_ranking(plan["normalized_query"], candidates, warnings, allow_remote) if rerank_enabled else (candidates, False)
        if getattr(retriever, 'document_boost_enabled', True):
            candidates = prioritize(candidates, getattr(retriever, 'documents', []), getattr(retriever, 'priority_time', priority_day()),
                                    score_key='rerank_score' if reranked else 'score')
        evidence = retain_evidence(candidates, steps, plan["model_ids"], retriever.document_kinds,
                                   implementation_scope or bool(re.search(r'官方|源码|Torchvision', plan['original_query'], re.I)))
    else:
        evidence, reranked = results[0][1]["evidence"], results[0][1]["reranked"]
        evidence = [{**item, "step_ids": [steps[0]["id"]]} for item in evidence]
    attach_references(steps, evidence)
    for step in steps:
        if step["status"] == "missing":
            warnings.append(f"子问题“{step['query']}”未找到可用正文，回答只能覆盖已有资料。")
    channels = {}
    for _, result in results:
        for channel, count in result["channels"].items():
            channels[channel] = channels.get(channel, 0) + count
    return {"evidence": evidence, "paths": [] if plan.get('document_id') or plan.get('document_ids') else paths if complex_query else results[0][1]["paths"], "warnings": list(dict.fromkeys(warnings)), "remote_allowed": allow_remote,
            "channels": channels, "reranked": reranked, "steps": steps, "stop_reason": stop_reason, "duration_ms": round((perf_counter() - started) * 1000)}
