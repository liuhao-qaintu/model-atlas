"""从已保存关系构建可追溯的浏览视图，不请求外部模型。"""
from collections import defaultdict, deque
from urllib.parse import urlsplit

ENTITY_KINDS = {
    "basicblock": "模块", "conv1d": "算子", "conv2d": "算子",
    "residual": "机制", "tiny-signal-cnn": "教学实例",
    "paper-resnet": "论文", "resnet-family": "模型家族",
    "global-average-pooling": "机制", "linear-head": "算子",
    "lenet-tutorial": "教学实例", "paper-transformer": "论文",
    "transformer-encoder-layer": "模块", "transformer-decoder-layer": "模块",
    "multihead-attention": "模块", "positionwise-ffn": "模块",
    "tiny-seq2seq": "教学实例", "position-embedding": "机制", "vocabulary-projection": "输出层",
    "paper-vit": "论文", "paper-clip": "论文", "tiny-patch-vit": "教学实例",
    "vit-encoder-block": "模块", "patch-embedding": "模块", "class-token": "机制",
    "clip-visual-vit": "模型变体", "clip-text-transformer": "模块",
    "contrastive-alignment": "学习目标", "normalized-similarity": "匹配方式", "tiny-dual-encoder": "教学实例",
    "tiny-dense-ae": "教学实例", "dense-ae-encoder": "模块", "dense-ae-decoder": "模块",
    "latent-bottleneck": "机制", "input-reconstruction": "输出", "mse-reconstruction": "损失", "linear-map": "算子",
    "tiny-bi-lstm": "教学实例", "bilstm-core": "模块", "lstm-gates": "机制", "recurrent-state": "状态",
    "bidirectional-summary": "机制", "packed-sequence": "接口", "sequence-length": "长度信息",
    "forward-final-state": "状态", "backward-final-state": "状态",
    'detr-cnn-backbone': '视觉骨干', 'detr-transformer': '模块', 'detr-object-queries': '查询', 'object-detection': '任务',
    'llava-clip-vision': '视觉分支', 'llava-projector': '连接器', 'vicuna-language-model': '语言模块', 'visual-question-answering': '任务',
    'dinov2-vit-backbone': '视觉骨干', 'visual-self-distillation': '学习目标', 'visual-token-features': '表示',
    'dense-visual-prediction': '任务', 'downstream-task-head': '任务头',
    'selective-ssm': '机制', 'ssd-duality': '数学框架', 'complex-state': '机制', 'mimo-state': '内部结构',
    'sigmoid-pair-loss': '学习目标', 'caption-training': '训练目标', 'gram-anchoring': '正则',
}


def source_link_safe(source):
    url = source.get("url", "")
    if source.get("kind") in {"internal", "document"}:
        return url.startswith(("/models/", "/documents/")) and not url.startswith("//")
    parsed = urlsplit(url)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def proven_edges(retriever):
    """所有来源及指定摘录都存在才展示；摘录匹配不等于自动核验其真伪。"""
    sections = {section["id"]: section for section in retriever.sections}
    edges = []
    for relation in retriever.relations:
        if relation.get('status', 'accepted') != 'accepted':
            continue
        ids = relation.get("source_ids", [])
        if not ids or any(sid not in retriever.sources or not source_link_safe(retriever.sources[sid]) for sid in ids):
            continue
        owner = relation.get("owner_model_id")
        if owner not in retriever.models:
            continue
        proofs = []
        for support in relation.get("evidence", []):
            if support.get('section_id') in getattr(retriever, 'archived_section_ids', set()):
                continue
            section = sections.get(support.get("section_id"))
            quote = support.get("quote", "")
            if not section or section["model_id"] != owner or not quote.strip() or quote not in section["content"]:
                break
            proofs.append({"section_id": section["id"], "title": section["title"], "quote": quote,
                           "reader_url": section.get("reader_url", f"/models/{owner}?section={section['slug']}")})
        else:
            if proofs:
                edges.append({**relation, **{f'{side}_name': retriever.models[relation[side]]['name'] for side in ('source', 'target') if relation[side] in retriever.models}, "evidence": proofs, "sources": [retriever.sources[sid] for sid in ids],
                              "owner_name": retriever.models[owner]["name"]})
    return sorted(edges, key=lambda edge: edge["id"])


def build_graph(retriever, root=None, max_hops=2, kind=None):
    """按无向邻接探索，关系箭头始终保留原始方向，路径最多两跳。"""
    if max_hops not in {1, 2}:
        raise ValueError("探索深度只能为 1 或 2")
    all_edges = proven_edges(retriever)
    kinds = sorted({edge["kind"] for edge in all_edges})
    if kind and kind not in kinds:
        raise ValueError("关系类型不存在")
    nodes = {mid: {"id": mid, "name": model["name"], "kind": model["kind"], "model_id": mid,
                   "ready": model.get('searchable', model["ready"]), "reader_url": f"/models/{mid}", "description": model["description"]}
             for mid, model in retriever.models.items()}
    for edge in all_edges:
        for side in ("source", "target"):
            entity = edge[side]
            nodes.setdefault(entity, {"id": entity, "name": edge[f"{side}_name"],
                                     "kind": ENTITY_KINDS.get(entity, "实体"), "model_id": None,
                                     "ready": True, "reader_url": None, "description": "点击相关关系查看来源与章节摘录。"})
    if root and root not in nodes:
        raise LookupError("图谱节点不存在")
    eligible = [edge for edge in all_edges if not kind or edge["kind"] == kind]
    adjacency = defaultdict(list)
    for edge in eligible:
        adjacency[edge["source"]].append((edge["target"], edge))
        adjacency[edge["target"]].append((edge["source"], edge))
    distances, routes = {}, {}
    if root:
        distances[root] = 0
        routes[root] = []
        queue = deque([root])
        while queue:
            node = queue.popleft()
            if distances[node] == max_hops:
                continue
            for neighbor, edge in adjacency[node]:
                if neighbor not in distances:
                    distances[neighbor] = distances[node] + 1
                    routes[neighbor] = routes[node] + [{"edge_id": edge["id"], "from": node, "to": neighbor,
                                                      "reverse": edge["source"] != node}]
                    queue.append(neighbor)
        # 每条边至少有一端还可向外走；避免把边界节点之间的额外边算进限定深度。
        edges = [edge for edge in eligible if edge["source"] in distances and edge["target"] in distances
                 and min(distances[edge["source"]], distances[edge["target"]]) < max_hops]
        edges.sort(key=lambda edge: (min(distances[edge["source"]], distances[edge["target"]]),
                                     edge["source"] != root, edge["id"]))
        visible = set(distances)
    else:
        edges = eligible
        visible = set(nodes) if not kind else {entity for edge in edges for entity in (edge["source"], edge["target"])}
    incoming = defaultdict(list)
    for edge in all_edges:
        incoming[edge["target"]].append(edge)
    shared = []
    for entity, links in incoming.items():
        # 只归纳人工标注可比较的结构，不把一个家族或同一模型的重复边当成相似性。
        links = [edge for edge in links if edge.get("implementation") and (edge["kind"] == "共享结构" or edge.get("comparison_group"))]
        if entity not in visible or len({edge["owner_model_id"] for edge in links}) < 2:
            continue
        shared.append({"node_id": entity, "name": nodes[entity]["name"], "edge_ids": [edge["id"] for edge in links],
                       "implementations": [{"entity": edge["source_name"], "model_name": edge["owner_name"], "text": edge["implementation"],
                                            "reader_url": edge["evidence"][0]["reader_url"], "edge_id": edge["id"]} for edge in links]})
    visible_nodes = [{**node, "distance": distances.get(entity)} for entity, node in nodes.items() if entity in visible]
    return {"root": root, "max_hops": max_hops, "kind": kind, "nodes": visible_nodes, "edges": edges,
            "catalog": list(nodes.values()), "kinds": kinds, "shared": shared,
            "paths": [{"target": entity, "hops": distances[entity], "steps": steps} for entity, steps in routes.items() if steps],
            "stats": {"nodes": len(nodes), "edges": len(all_edges), "excluded_edges": len(retriever.relations) - len(all_edges)},
            "curation": "人工整理与已确认的自动提取关系；候选不参与检索，浏览路径不代表本轮问答执行了多跳。"}
