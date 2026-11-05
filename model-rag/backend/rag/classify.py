"""文档级域先验 + section 分类 + 实体链接。

只产出“草稿”，不写图谱：调用方把结果放进审核队列，人工确认后才写入。
提示词按 2026-11-02 的 v1/v2/v3 样例收敛：
- 文档级域先验，section 继承，避免局部词把领域带偏；
- 每 section 最多 6 个实体，优先 Model/Dataset/Task/Metric；
- 实体链接先 exact/fuzzy，再逐条让模型判定；批量判定不稳定，不用。
"""
from __future__ import annotations

import asyncio
import json
import re
from typing import Any, Awaitable, Callable

from backend.infra import config
from backend.rag.retrieval import provider_post

TOP_LEVEL = [
    "视觉", "自然语言处理", "多模态", "语音", "时序与信号", "图学习",
    "强化学习", "生成模型", "训练与优化", "应用领域",
]

SUBDOMAINS: dict[str, list[str]] = {
    "视觉": ["图像分类", "目标检测", "图像分割", "图像生成", "视频理解", "三维视觉", "医学影像", "遥感图像", "工业视觉"],
    "自然语言处理": ["文本分类", "信息抽取", "机器翻译", "问答系统", "文本生成", "检索与排序", "对话系统", "知识图谱"],
    "多模态": ["图文检索", "视觉问答", "图文生成", "视频-语言"],
    "语音": ["语音识别", "语音合成", "语音增强"],
    "时序与信号": ["故障诊断", "异常检测", "预测性维护", "时间序列预测", "信号分类", "振动信号分析"],
    "图学习": ["节点分类", "链路预测", "图生成"],
    "强化学习": ["策略优化", "离线强化学习", "多智能体"],
    "生成模型": ["扩散模型", "生成对抗网络", "自回归生成", "变分自编码器"],
    "训练与优化": ["预训练", "微调", "蒸馏", "量化剪枝", "分布式训练", "数据增强"],
    "应用领域": ["医疗", "工业", "金融", "自动驾驶", "推荐系统", "教育", "科学发现"],
}

SECTION_ROLES = ["摘要", "背景", "方法", "实验", "结论", "局限", "其他"]
KNOWLEDGE_TYPES = ["概念", "模型", "组件", "数据集", "任务", "指标", "结论", "其他"]
ENTITY_TYPES = ["Model", "Component", "Dataset", "Task", "Metric", "Concept"]
LINK_DECISIONS = {"link_existing", "new_node", "ambiguous"}

SUBDOMAIN_TEXT = "\n".join(f"{top}：{'、'.join(items)}" for top, items in SUBDOMAINS.items())

CLASSIFY_SYSTEM = f"""你是知识入库分类器。给定论文片段，只输出严格 JSON，不要解释。
顶层领域：{'、'.join(TOP_LEVEL)}。
子领域参考（优先选；允许新词，但必须 domain_new=true）：
{SUBDOMAIN_TEXT}
章节角色：{'、'.join(SECTION_ROLES)}。
角色线索：we propose/this paper proposes/our method → 方法；related work/background/traditional methods → 背景；accuracy/table/results/experiments → 实验；conclusion/future work → 结论；limitation/shortcoming → 局限；references/acknowledgements → 其他。
知识类型：{'、'.join(KNOWLEDGE_TYPES)}。
实体类型：{'、'.join(ENTITY_TYPES)}。
规则：
- domain_path 必须是 [顶层, 子领域]，第二级不能重复顶层词。
- 子领域不在参考表时 domain_new=true。
- 实体最多 6 个，优先 Model/Dataset/Task/Metric；通用损失/层（cross-entropy、ReLU）除非核心贡献否则不抽。
- 不要编造片段里没有的实体。
输出字段：domain_path、domain_new、section_role、knowledge_type、entities(name,type,aliases)、confidence、summary。"""

PRIOR_SYSTEM = """你是文档领域判定器。根据标题和摘要判断文档的主领域，只输出严格 JSON。
顶层领域：视觉、自然语言处理、多模态、语音、时序与信号、图学习、强化学习、生成模型、训练与优化、应用领域。
输出：{"domain_path":["顶层","子领域"],"domain_new":true|false,"confidence":0-1,"reason":"一句话"}"""

JUDGE_SYSTEM = """你是实体链接判定器。判断实体是否对应候选模型，只输出严格 JSON。
规则：多个候选同名/同义 → ambiguous；唯一明确对应 → link_existing；无合适候选 → new_node。
输出：{"decision":"link_existing|new_node|ambiguous","model_id":null,"reason":"一句话"}"""


def _json_from(text: str) -> dict[str, Any]:
    match = re.search(r"\{.*\}", text or "", re.S)
    if not match:
        raise ValueError("模型未返回 JSON")
    data = json.loads(match.group(0))
    if not isinstance(data, dict):
        raise ValueError("模型返回的不是对象")
    return data


async def _chat(system: str, user: str, max_tokens: int = 800) -> dict[str, Any]:
    body = {
        "model": config.LLM_MODEL,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "temperature": 0,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }
    response = await asyncio.wait_for(
        provider_post(config.LLM_BASE, config.LLM_KEY, "chat/completions", body), timeout=65
    )
    choice = response["choices"][0]
    return _json_from(choice["message"]["content"])


def _clean_domain(raw: Any, fallback: list[str] | None = None) -> tuple[list[str], bool]:
    path = [str(item).strip() for item in raw] if isinstance(raw, list) else []
    path = [item for item in path if item]
    if len(path) < 2 or path[0] not in TOP_LEVEL:
        path = list(fallback or ["应用领域", "其他"])[:2]
    if len(path) == 1:
        path.append("其他")
    if path[1] == path[0]:
        path[1] = "其他"
    known = path[1] in SUBDOMAINS.get(path[0], [])
    return path[:2], not known


def normalize_classification(raw: dict[str, Any], prior: dict[str, Any] | None = None) -> dict[str, Any]:
    fallback = (prior or {}).get("domain_path")
    domain_path, domain_new = _clean_domain(raw.get("domain_path"), fallback)
    role = raw.get("section_role") if raw.get("section_role") in SECTION_ROLES else "其他"
    kind = raw.get("knowledge_type") if raw.get("knowledge_type") in KNOWLEDGE_TYPES else "其他"
    entities, seen = [], set()
    for item in raw.get("entities") or []:
        if isinstance(item, str):
            item = {"name": item, "type": "Concept", "aliases": []}
        if not isinstance(item, dict) or not str(item.get("name", "")).strip():
            continue
        name = str(item["name"]).strip()
        key = name.casefold()
        if key in seen:
            continue
        seen.add(key)
        etype = item.get("type") if item.get("type") in ENTITY_TYPES else "Concept"
        aliases = [str(alias).strip() for alias in (item.get("aliases") or []) if str(alias).strip()]
        entities.append({"name": name, "type": etype, "aliases": aliases})
        if len(entities) >= 6:
            break
    try:
        confidence = max(0.0, min(1.0, float(raw.get("confidence", 0))))
    except (TypeError, ValueError):
        confidence = 0.0
    return {
        "domain_path": domain_path,
        "domain_new": bool(raw.get("domain_new", True)) or domain_new,
        "section_role": role,
        "knowledge_type": kind,
        "entities": entities,
        "confidence": round(confidence, 3),
        "summary": str(raw.get("summary", ""))[:300],
    }


async def document_domain_prior(title: str, sample_text: str) -> dict[str, Any]:
    user = f"文档标题：{title}\n\n开头内容：\n{(sample_text or '')[:1800]}"
    raw = await _chat(PRIOR_SYSTEM, user, 300)
    domain_path, domain_new = _clean_domain(raw.get("domain_path"))
    try:
        confidence = max(0.0, min(1.0, float(raw.get("confidence", 0))))
    except (TypeError, ValueError):
        confidence = 0.0
    return {
        "domain_path": domain_path,
        "domain_new": bool(raw.get("domain_new", True)) or domain_new,
        "confidence": round(confidence, 3),
        "reason": str(raw.get("reason", ""))[:200],
    }


async def classify_section(section: dict[str, Any], prior: dict[str, Any]) -> dict[str, Any]:
    if not section.get("cloud_allowed", True):
        return {
            "domain_path": list(prior.get("domain_path") or ["应用领域", "其他"]),
            "domain_new": bool(prior.get("domain_new", True)),
            "section_role": "其他",
            "knowledge_type": "其他",
            "entities": [],
            "confidence": 0.0,
            "summary": "",
            "skipped": "cloud_not_allowed",
        }
    user = (
        f"文档领域先验：{json.dumps(prior.get('domain_path'), ensure_ascii=False)}\n"
        f"位置：{section.get('locator') or section.get('title') or ''}\n\n"
        f"{(section.get('content') or '')[:3000]}"
    )
    raw = await _chat(CLASSIFY_SYSTEM, user, 800)
    return normalize_classification(raw, prior)


def normalize_models(models: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "id": item.get("id"),
            "name": str(item.get("name", "")),
            "aliases": [str(alias) for alias in (item.get("aliases") or [])],
        }
        for item in models
        if item.get("id")
    ]


def exact_or_fuzzy_link(name: str, models: list[dict[str, Any]]) -> dict[str, Any] | None:
    norm = name.casefold().strip()
    if not norm:
        return None
    exact = []
    for model in models:
        names = [model["name"], *model["aliases"]]
        if norm in [item.casefold().strip() for item in names]:
            exact.append(model)
    if len(exact) == 1:
        return {"decision": "link_existing", "model_id": exact[0]["id"], "reason": "exact alias"}
    if len(exact) > 1:
        return {"decision": "ambiguous", "reason": "多个精确别名", "candidates": [m["id"] for m in exact[:4]]}
    fuzzy = []
    for model in models:
        names = [model["name"], *model["aliases"]]
        if any(norm in item.casefold() or item.casefold() in norm for item in names if len(item.strip()) > 2):
            fuzzy.append(model)
    if len(fuzzy) == 1:
        return {"decision": "link_existing", "model_id": fuzzy[0]["id"], "reason": "fuzzy alias"}
    if len(fuzzy) > 1:
        return {"decision": "ambiguous", "reason": "多个近似候选", "candidates": [m["id"] for m in fuzzy[:4]]}
    return None


async def judge_entity(name: str, etype: str, candidates: list[dict[str, Any]]) -> dict[str, Any]:
    if not candidates:
        return {"decision": "new_node", "reason": "无候选"}
    user = (
        f"实体：{name}（类型 {etype}）\n候选：{json.dumps(candidates, ensure_ascii=False)}\n"
        "判断并输出 JSON。"
    )
    raw = await _chat(JUDGE_SYSTEM, user, 200)
    decision = raw.get("decision")
    model_id = raw.get("model_id")
    ids = {item.get("model_id") for item in candidates}
    if decision not in LINK_DECISIONS:
        decision = "ambiguous"
    if decision == "link_existing" and model_id not in ids:
        decision = "ambiguous"
    return {"decision": decision, "model_id": model_id if decision == "link_existing" else None, "reason": str(raw.get("reason", ""))[:200]}


CandidateFn = Callable[[str], Awaitable[list[dict[str, Any]]]]


async def classify_document(
    sections: list[dict[str, Any]],
    models: list[dict[str, Any]],
    prior: dict[str, Any],
    candidates_fn: CandidateFn | None = None,
):
    """逐 section 产出分类草稿和实体链接；调用方负责发布事件与落审核队列。"""
    normalized = normalize_models(models)
    for section in sections:
        try:
            classification = await classify_section(section, prior)
        except Exception as exc:
            yield {
                "section_id": section.get("id"),
                "page": section.get("page"),
                "part": section.get("part"),
                "title": section.get("title") or section.get("locator"),
                "error": f"{type(exc).__name__}: {str(exc)[:200]}",
                "classification": None,
                "entity_links": [],
            }
            continue
        links = []
        for entity in classification.get("entities", []):
            link = exact_or_fuzzy_link(entity["name"], normalized)
            if link is None and candidates_fn is not None:
                try:
                    candidates = await candidates_fn(entity["name"])
                except Exception:
                    candidates = []
                link = await judge_entity(entity["name"], entity["type"], candidates)
            if link is None:
                link = {"decision": "new_node", "reason": "无候选"}
            links.append({"entity": entity["name"], "type": entity["type"], "link": link})
        yield {
            "section_id": section.get("id"),
            "page": section.get("page"),
            "part": section.get("part"),
            "title": section.get("title") or section.get("locator"),
            "classification": classification,
            "entity_links": links,
        }
