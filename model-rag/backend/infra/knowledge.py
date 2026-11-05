"""将已有 Markdown 模型卡转为稳定章节、来源与有证据的关系。"""
import hashlib
import re

from backend.infra.config import CONTENT_ROOT
from backend.infra.database import KnowledgeModel, KnowledgeRelation, KnowledgeSection, KnowledgeSource, upsert_records
from backend.infra.recent_knowledge import RECENT_TOPICS, recent_relations
from backend.infra.learning_guides import formula_guide

TOPICS = [
    {"id": "cnn-1d", "name": "1D CNN", "kind": "架构家族", "category": "卷积网络", "description": "面向序列与信号的卷积架构，提取局部特征并聚合为分类表示。", "aliases": ["1d cnn", "一维卷积", "一维cnn", "conv1d", "tinysignalcnn"], "file": "01_1D_CNN.md", "source_ids": ["S01", "S02", "E01"], "tags": ["信号", "卷积", "分类"]},
    {"id": "cnn-2d", "name": "2D CNN / LeNet", "kind": "架构与教学实例", "category": "卷积网络", "description": "面向图像的卷积架构，以 LeNet 教学实现说明结构与张量形状。", "aliases": ["2d cnn", "二维卷积", "lenet", "conv2d"], "file": "03_2D_CNN_LeNet.md", "source_ids": ["S03", "S05", "E03"], "tags": ["图像", "卷积"]},
    {"id": "resnet-18", "name": "ResNet-18", "kind": "具体模型", "category": "卷积网络", "description": "采用基础残差块的图像分类模型，提供特征提取与分类两类输出。", "aliases": ["resnet", "resnet18", "resnet-18", "残差网络"], "file": "02_ResNet18.md", "source_ids": ["S06", "S07", "S08", "E02"], "tags": ["残差", "图像", "特征"]},
    {"id": "autoencoder", "name": "Autoencoder", "kind": "方法家族", "category": "表示学习", "description": "通过编码与解码学习潜在表示，并以输入重构作为学习目标。", "aliases": ["autoencoder", "自编码器", "tinydenseautoencoder"], "file": "07_Autoencoder.md", "source_ids": ["S10", "S29", "S30", "S55", "E07"], "tags": ["重构", "表示"]},
    {"id": "lstm", "name": "LSTM", "kind": "网络结构", "category": "序列建模", "description": "通过门控和循环状态建模序列，支持双向编码与变长输入。", "aliases": ["lstm", "长短期记忆", "tinybilstm"], "file": "08_LSTM.md", "source_ids": ["S11", "S28", "S29", "E08"], "tags": ["序列", "门控"]},
    {"id": "transformer", "name": "Transformer", "kind": "架构家族", "category": "注意力", "description": "基于注意力与位置表示的序列架构，包含编码器和解码器。", "aliases": ["transformer", "tinyseq2seq"], "file": "04_Transformer.md", "source_ids": ["S13", "S14", "S21", "S22", "S23", "S24", "E04"], "tags": ["注意力", "序列"]},
    {"id": "vit", "name": "ViT", "kind": "模型家族", "category": "视觉与多模态", "description": "将图像分块映射为 token，通过 Transformer 编码与分类。", "aliases": ["vit", "vision transformer", "visiontransformer", "tinypatchvit"], "file": "05_ViT.md", "source_ids": ["S16", "S25", "S18", "E05"], "tags": ["图像", "注意力"]},
    {"id": "clip", "name": "CLIP", "kind": "模型与方法", "category": "视觉与多模态", "description": "通过图文对比学习对齐双编码器表示，并计算跨模态匹配分数。", "aliases": ["clip", "tinydualencoder"], "file": "06_CLIP.md", "source_ids": ["S18", "S19", "S26", "S27", "S31", "E06"], "tags": ["多模态", "对比学习"]},
    {"id": "detr", "name": "DETR", "kind": "论文应用实例", "category": "应用案例", "description": "组合 CNN 与 Transformer，将图像映射为目标类别和边界框集合。", "aliases": ["detr"], "file": "09_DETR.md", "source_ids": ["S32", "S35", "A01"], "tags": ["目标检测", "模型组合"]},
    {"id": "llava", "name": "LLaVA", "kind": "论文应用实例", "category": "应用案例", "description": "连接 CLIP 视觉分支、投影与语言模型，从图像和指令生成回答文本。", "aliases": ["llava", "llava原始模型"], "file": "10_LLaVA.md", "source_ids": ["S33", "S36", "A02"], "tags": ["多模态", "视觉问答"]},
    {"id": "dinov2", "name": "DINOv2", "kind": "论文应用实例", "category": "应用案例", "description": "通过 ViT 自监督预训练获取视觉表示，再连接分类或密集任务读出。", "aliases": ["dinov2", "dino v2"], "file": "11_DINOv2.md", "source_ids": ["S34", "S37", "A03"], "tags": ["自监督", "视觉表示"]},
]

SECTION_NAMES = {"概览": "overview", "概览与核心机制": "overview", "输入输出": "io", "输入输出与数据流": "io", "长度公式": "formula", "核心公式": "formula", "掩码与训练接口": "masks", "变长序列与方向": "masks", "最小代码": "code", "最小调用代码": "code", "与相关模型的联系": "relations", "模型联系与应用记录": "relations", "图示与资料状态": "figures", "图示与版本": "figures", "来源与定位": "sources", "可继续追问": "followups"}
SECTION_NAMES['应用流程与使用边界'] = 'applications'
SECTION_NAMES.update({'结构与原理': 'mechanism', '推导与排错': 'practice'})

# 新卡按章节绑定依据，不将整张卡的全部参考链接复制给每个章节。
SECTION_SOURCES = {
    'cnn-1d': {'mechanism': ['S01', 'E01'], 'practice': ['S01', 'S02', 'E01']},
    'resnet-18': {'mechanism': ['S06', 'S08', 'E02'], 'practice': ['S08', 'S44', 'S45', 'E02']},
    "cnn-2d": {"overview": ["S03", "S05"], "io": ["S03", "E03"], "formula": ["S03"], "code": ["E03", "S05"], "relations": ["E03", "S03", "S05"], "figures": ["S05"]},
    "transformer": {"overview": ["S13", "S21", "S22", "S24"], "io": ["S14", "S23", "S24", "E04"], "formula": ["S13"], "masks": ["S14", "S23", "E04"], "code": ["E04", "S14"], "relations": ["S21", "S22", "S24", "E04"], "figures": ["S13"]},
    "vit": {"overview": ["S16", "S25"], "io": ["E05", "S25"], "formula": ["S16", "S25"], "code": ["E05", "S25"], "relations": ["S25", "S18", "E05"], "figures": ["S16"]},
    "clip": {"overview": ["S18", "S26"], "io": ["S26", "S27", "S19", "E06"], "formula": ["S18", "S26", "S31"], "code": ["E06"], "relations": ["S18", "S26"], "figures": ["S18"]},
    "autoencoder": {"mechanism": ["S10", "S55", "E07"], "practice": ["S10", "S55", "E07"], "overview": ["S10", "E07"], "io": ["E07", "S30"], "formula": ["S10", "S30"], "code": ["E07"], "relations": ["E07", "S29", "S10"], "figures": ["S10"]},
    "lstm": {"overview": ["S11", "E08"], "io": ["S11", "E08"], "formula": ["S11"], "masks": ["S28", "E08"], "code": ["E08", "S11", "S28"], "relations": ["S11", "S29", "E08", "E01"], "figures": ["S12"]},
    "detr": {"overview": ["S32", "S52"], "mechanism": ["S32", "S46", "S47", "S53", "S54"], "io": ["S32", "S46", "S47", "S49", "A01"], "formula": ["S32", "S46", "S48", "S49"], "code": ["S46", "S49", "A01"], "practice": ["S48", "A01"], "relations": ["S32", "S47", "S54"], "applications": ["S32"], "figures": ["S32", "S47"]},
    "llava": {"overview": ["S33"], "io": ["S33"], "formula": ["S33"], "code": ["A02", "S33"], "relations": ["S33"], "applications": ["S33"]},
    "dinov2": {"overview": ["S34"], "io": ["S34"], "formula": ["S34"], "code": ["A03", "S34"], "relations": ["S34"], "applications": ["S34"]},
}


TOPICS.extend(RECENT_TOPICS)
SECTION_NAMES['架构图与论文插图'] = 'figures'
for topic, sid, aid in zip(RECENT_TOPICS, ('S38', 'S39', 'S40', 'S41', 'S42'), ('A04', 'A05', 'A06', 'A07', 'A08')):
    SECTION_SOURCES[topic['id']] = {slug: [sid] for slug in ('overview', 'io', 'formula', 'relations', 'applications', 'figures')}
    SECTION_SOURCES[topic['id']]['code'] = [sid, aid]
SECTION_SOURCES['mamba']['io'] = ['S38', 'S43']
SECTION_SOURCES['transformer'].update({'mechanism': ['S13', 'S14', 'S24', 'E04'], 'practice': ['S14', 'S24', 'S44', 'E04']})


def parse_sections(text):
    sections, current, fence = [], None, False
    for line in text.splitlines():
        if line.startswith("```"):
            fence = not fence
        if line.startswith("## ") and not fence:
            current = {"title": line[3:].strip(), "lines": []}
            sections.append(current)
        elif current is not None:
            current["lines"].append(line)
    return [{"title": s["title"], "content": "\n".join(s["lines"]).strip()} for s in sections]


def source_records():
    text = (CONTENT_ROOT / "首批资料清单.md").read_text(encoding="utf-8")
    sources = []
    pattern = r"^\| (S\d+) \| ([^|]+) \| \[([^\]]+)\]\((https?://[^)]+)\) \| ([^|]+) \|"
    for match in re.finditer(pattern, text, re.M):
        sid, topic, title, url, scope = match.groups()
        checked = sid in {"S03", "S05", "S10", "S11", "S13", "S14", "S16", "S18", "S19", "S21", "S22", "S23", "S24", "S25", "S26", "S27", "S28", "S29", "S30", "S31", "S32", "S33", "S34", 'S38', 'S39', 'S40', 'S41', 'S42', 'S44', 'S45'}
        sources.append({"id": sid, "topic": topic.strip(), "title": title, "url": url, "scope": scope.strip(), "status": "相关段落已核对" if checked or sid == 'S43' else "入口已核对", "checked_at": "2026-10-14" if sid == 'S43' else "2026-10-13" if checked or sid in {'S35', 'S36', 'S37'} else "2026-10-12", "kind": "external"})
        if sid in {'S07', 'S44', 'S45'}:
            sources[-1].update(checked_at='2026-10-16', status='固定版本接口及相关内容已核对')
        if sid in {'S46', 'S47', 'S48', 'S49', 'S53', 'S54'}:
            sources[-1].update(checked_at='2026-10-18', status='固定提交中相关函数已核对')
        if sid == 'S55':
            sources[-1].update(checked_at='2026-11-04', status='作者公开原文第2–3页及2.2–2.4节已核对')
        if sid in {'S50', 'S51', 'S52'}:
            sources[-1].update(checked_at='2026-10-18', status='作者介绍相关段落已核对' if sid == 'S52' else '实现入口已核对')
    for sid, model, title in [("E01", "cnn-1d", "TinySignalCNN 教学示例"), ("E02", "resnet-18", "ResNet-18 调用与形状示例"), ("E03", "cnn-2d", "LeNetTutorial 教学示例"), ("E04", "transformer", "TinySeq2Seq 教学示例"), ("E05", "vit", "TinyPatchViT 教学配置"), ("E06", "clip", "TinyDualEncoder 形状教学示例"), ("E07", "autoencoder", "TinyDenseAutoencoder 教学示例"), ("E08", "lstm", "TinyBiLSTM 教学示例")]:
        sources.append({"id": sid, "topic": model, "title": title, "url": f"/models/{model}?section=code", "scope": "教学示例，已通过前向形状核查", "status": "代码已核查", "checked_at": "2026-10-12" if sid in {"E01", "E02"} else "2026-10-13", "kind": "internal"})
    for sid, model in [('A01', 'detr'), ('A02', 'llava'), ('A03', 'dinov2'), ('A04', 'mamba'), ('A05', 'mamba2'), ('A06', 'mamba3'), ('A07', 'siglip2'), ('A08', 'dinov3')]:
        name = next(topic['name'] for topic in TOPICS if topic['id'] == model)
        sources.append({'id': sid, 'topic': model, 'title': name + ' 结构伪代码', 'url': f'/models/{model}?section=code',
                        'scope': '原创结构示意；不是作者可运行接口，未部署或训练任务模型。', 'status': '结构已核对', 'checked_at': '2026-10-13', 'kind': 'internal'})
        if model == 'detr':
            sources[-1].update(title='DETR数据流与输出教学例', scope='自编布局、输出还原和小矩阵匹配；局部运行核对通过，不是完整检测器或论文性能验证。', status='局部运算已核对', checked_at='2026-10-18')
    ids = [source['id'] for source in sources]
    if len(ids) != len(set(ids)):
        raise ValueError('资料来源编号重复，请先修正清单，避免覆盖其他来源。')
    return sources


def seed_knowledge():
    sources = source_records()
    models, sections = [], []
    for position, topic in enumerate(TOPICS):
        record = dict(topic)
        record["order"] = position
        filename = record.pop("file", None)
        record["ready"] = bool(filename)
        record["section_count"] = 0
        if filename:
            text = (CONTENT_ROOT / "模型卡" / filename).read_text(encoding="utf-8")
            record["content_file"] = filename
            parsed = parse_sections(text)
            if topic['id'] == 'resnet-18' and not any(SECTION_NAMES.get(s['title']) == 'formula' for s in parsed):
                parsed.insert(2, {'title': '核心公式', 'content': r'残差块可用 $y=F(x)+S(x)$ 说明相加部分：F 是残差分支，S 是 shortcut。两路形状要匹配；默认 Torchvision BasicBlock 在相加后还应用 ReLU，因此这个简式不是完整 forward。实际定义与分支见 [Torchvision BasicBlock](https://github.com/pytorch/vision/blob/v0.21.0/torchvision/models/resnet.py)。'})
            for section_position, section in enumerate(parsed):
                slug = SECTION_NAMES.get(section["title"], hashlib.sha1(section["title"].encode()).hexdigest()[:8])
                if slug == 'formula': section['content'] += formula_guide(topic['id'])
                section_sources = SECTION_SOURCES.get(topic["id"], {}).get(slug, topic.get("source_ids", []))
                if topic["id"] == "resnet-18" and slug == "relations":
                    section_sources = [*section_sources, "E01"]
                section.update({"id": f"{topic['id']}:{slug}", "slug": slug, "model_id": topic["id"], "order": section_position, "source_ids": section_sources})
                sections.append(section)
            record["section_count"] = len(parsed)
        models.append(record)
    relations = [
        {"id": "resnet-basicblock", "source": "resnet-18", "target": "basicblock", "source_name": "ResNet-18", "target_name": "BasicBlock", "relation": "使用模块", "source_ids": ["S08"], "locator": "resnet18 / BasicBlock"},
        {"id": "basicblock-conv2d", "source": "basicblock", "target": "conv2d", "source_name": "BasicBlock", "target_name": "二维卷积", "relation": "包含算子", "source_ids": ["S08"], "locator": "BasicBlock.__init__"},
        {"id": "basicblock-residual", "source": "basicblock", "target": "residual", "source_name": "BasicBlock", "target_name": "残差相加", "relation": "采用机制", "source_ids": ["S08"], "locator": "BasicBlock.forward"},
        {"id": "cnn1d-example", "source": "cnn-1d", "target": "tiny-signal-cnn", "source_name": "1D CNN", "target_name": "TinySignalCNN", "relation": "本卡教学实例", "source_ids": ["E01"], "locator": "TinySignalCNN"},
        {"id": "example-conv1d", "source": "tiny-signal-cnn", "target": "conv1d", "source_name": "TinySignalCNN", "target_name": "一维卷积", "relation": "包含算子", "source_ids": ["E01", "S01"], "locator": "TinySignalCNN.features"},
        {"id": "paper-resnet", "source": "paper-resnet", "target": "resnet-family", "source_name": "ResNet 原论文", "target_name": "ResNet 系列", "relation": "提出", "source_ids": ["S06"], "locator": "论文摘要；完整方法待整理"},
        {"id": "resnet-family", "source": "resnet-18", "target": "resnet-family", "source_name": "ResNet-18", "target_name": "ResNet 系列", "relation": "属于", "source_ids": ["S07", "S08"], "locator": "resnet18"},
    ]
    # 关系证据指向具体章节的原样摘录，避免把全卡参考链接当成逐条依据。
    details = {
        "resnet-basicblock": ("结构组成", "resnet-18", "overview", "Torchvision 的 ResNet-18 使用 BasicBlock，四个阶段各含两个块；块包含二维卷积和残差分支。", "框架实现中的具体组成关系。"),
        "basicblock-conv2d": ("结构组成", "resnet-18", "overview", "实现细节：BasicBlock 的 F 使用两个 3×3 卷积；S 需要对齐时采用 1×1 卷积与归一化。", "二维卷积不等于一维卷积，输入的空间轴不同。"),
        "basicblock-residual": ("结构组成", "resnet-18", "relations", "结构组成关系：ResNet-18 → BasicBlock → Conv2d / 残差相加等。", "相加需形状匹配，框架实现还有相加后的 ReLU。"),
        "cnn1d-example": ("教学实例", "cnn-1d", "code", "class TinySignalCNN(nn.Module):", "原创最小分类器，用于解释 1D CNN；并非每个 1D CNN 的统一实现。"),
        "example-conv1d": ("结构组成", "cnn-1d", "code", "nn.Conv1d(in_channels, 16, kernel_size=5, padding=2),", "本卡教学实例中的具体算子。"),
        "paper-resnet": ("论文背景", "resnet-18", "relations", "原论文提出残差学习框架，本卡将其归为 ResNet 系列；ResNet-18 是具体变体。", "仅核对摘要中的残差学习背景，完整方法与应用关系待整理。"),
        "resnet-family": ("模型归属", "resnet-18", "relations", "原论文提出残差学习框架，本卡将其归为 ResNet 系列；ResNet-18 是具体变体。", "区分模型系列与具体变体。"),
    }
    for edge in relations:
        kind, owner, slug, quote, scope = details[edge["id"]]
        edge.update({"kind": kind, "owner_model_id": owner, "scope": scope, "evidence": [{"section_id": f"{owner}:{slug}", "quote": quote}]})
        if edge["id"] in {"basicblock-conv2d", "basicblock-residual"}:
            edge["comparison_group"] = edge["target"]
            edge["implementation"] = "二维卷积：BasicBlock 的残差分支含两个 3×3 卷积；对齐分支按需使用 1×1 卷积。" if edge["target"] == "conv2d" else "BasicBlock：卷积分支与恒等或投影分支相加，随后 ReLU；形状必须匹配。"
    for eid, source, name, target, target_name, owner, quote, source_ids, implementation in [
        ("example-global-pool", "tiny-signal-cnn", "TinySignalCNN", "global-average-pooling", "全局平均聚合", "cnn-1d", "nn.AdaptiveAvgPool1d(1),", ["E01", "S02"], "AdaptiveAvgPool1d(1)：沿序列长度聚合，展平后得到 32 维特征。"),
        ("resnet-global-pool", "resnet-18", "ResNet-18", "global-average-pooling", "全局平均聚合", "resnet-18", "`nn.AdaptiveAvgPool2d((1, 1))`", ["S08"], "AdaptiveAvgPool2d((1, 1))：沿二维空间聚合，展平后得到 512 维特征。"),
        ("example-linear-head", "tiny-signal-cnn", "TinySignalCNN", "linear-head", "Linear 分类头", "cnn-1d", "self.classifier = nn.Linear(32, num_classes)", ["E01"], "Linear(32, num_classes)：本例 num_classes=4。"),
        ("resnet-linear-head", "resnet-18", "ResNet-18", "linear-head", "Linear 分类头", "resnet-18", "`nn.Linear(512 * block.expansion, num_classes)`", ["S08"], "BasicBlock expansion=1，分类头 Linear(512, num_classes)，默认 1000 类。"),
    ]:
        slug = "code" if owner == "cnn-1d" else "relations"
        relations.append({"id": eid, "source": source, "target": target, "source_name": name, "target_name": target_name, "relation": "采用聚合方式" if target == "global-average-pooling" else "使用分类头", "kind": "共享结构", "owner_model_id": owner, "source_ids": source_ids, "locator": "TinySignalCNN" if owner == "cnn-1d" else "ResNet.__init__ / ResNet._forward_impl · v0.21.0", "implementation": implementation, "scope": "只比较本卡的两个具体实现；共享模块不代表架构、权重或任务相同。", "evidence": [{"section_id": f"{owner}:{slug}", "quote": quote}]})
    for eid, source, source_name, target, target_name, verb, owner, slug, quote, ids, locator, implementation in [
        ("cnn2d-example", "cnn-2d", "2D CNN / LeNet", "lenet-tutorial", "LeNetTutorial", "本卡教学实例", "cnn-2d", "code", "class LeNetTutorial(nn.Module):", ["E03"], "LeNetTutorial", ""),
        ("lenet-conv2d", "lenet-tutorial", "LeNetTutorial", "conv2d", "二维卷积", "包含算子", "cnn-2d", "code", "nn.Conv2d(in_channels, 6, kernel_size=5, padding=2),", ["E03", "S03"], "LeNetTutorial.features", "两个 5×5 Conv2d，使用 Sigmoid 和平均池化；本例没有残差连接。"),
        ("lenet-linear-head", "lenet-tutorial", "LeNetTutorial", "linear-head", "Linear 分类头", "使用分类头", "cnn-2d", "code", "self.classifier = nn.Linear(84, num_classes)", ["E03"], "LeNetTutorial.classifier", "Linear(84, num_classes)：本例 num_classes=10。"),
        ("transformer-paper", "paper-transformer", "Transformer 原论文", "transformer", "Transformer", "提出", "transformer", "overview", "原论文提出以注意力为核心的 encoder-decoder，并展示机器翻译任务。", ["S13"], "Attention Is All You Need · §3", ""),
        ("transformer-encoder", "transformer", "Transformer", "transformer-encoder-layer", "TransformerEncoderLayer", "使用模块", "transformer", "overview", "PyTorch 的 TransformerEncoderLayer 包含自注意力和逐位置前馈网络", ["S21", "S24"], "Transformer / TransformerEncoderLayer · v2.6.0", ""),
        ("transformer-decoder", "transformer", "Transformer", "transformer-decoder-layer", "TransformerDecoderLayer", "使用模块", "transformer", "overview", "TransformerDecoderLayer 还包含面向编码器 memory 的注意力。", ["S22", "S24"], "Transformer / TransformerDecoderLayer · v2.6.0", ""),
        ("transformer-example", "transformer", "Transformer", "tiny-seq2seq", "TinySeq2Seq", "本卡教学实例", "transformer", "code", "class TinySeq2Seq(nn.Module):", ["E04"], "TinySeq2Seq", ""),
        ("encoder-attention", "transformer-encoder-layer", "TransformerEncoderLayer", "multihead-attention", "多头注意力", "使用模块", "transformer", "relations", "编码器与解码器都使用多头注意力", ["S21", "S24"], "TransformerEncoderLayer.self_attn", ""),
        ("decoder-attention", "transformer-decoder-layer", "TransformerDecoderLayer", "multihead-attention", "多头注意力", "使用模块", "transformer", "relations", "编码器与解码器都使用多头注意力", ["S22", "S24"], "TransformerDecoderLayer.self_attn / multihead_attn", ""),
        ("encoder-feedforward", "transformer-encoder-layer", "TransformerEncoderLayer", "positionwise-ffn", "逐位置前馈网络", "使用模块", "transformer", "overview", "TransformerEncoderLayer 包含自注意力和逐位置前馈网络", ["S21", "S24"], "TransformerEncoderLayer._ff_block", ""),
        ("decoder-feedforward", "transformer-decoder-layer", "TransformerDecoderLayer", "positionwise-ffn", "逐位置前馈网络", "使用模块", "transformer", "relations", "Transformer → TransformerEncoderLayer / TransformerDecoderLayer → 多头注意力、逐位置前馈网络、残差相加。", ["S22", "S24"], "TransformerDecoderLayer._ff_block", ""),
        ("encoder-residual", "transformer-encoder-layer", "TransformerEncoderLayer", "residual", "残差相加", "采用机制", "transformer", "overview", "两类层都使用残差相加和 LayerNorm，具体顺序取决于 norm_first；本例沿用默认 False。", ["S21", "S24"], "TransformerEncoderLayer.forward", "自注意力、前馈分支分别使用残差相加和 LayerNorm；本例 norm_first=False。"),
        ("decoder-residual", "transformer-decoder-layer", "TransformerDecoderLayer", "residual", "残差相加", "采用机制", "transformer", "overview", "两类层都使用残差相加和 LayerNorm，具体顺序取决于 norm_first；本例沿用默认 False。", ["S22", "S24"], "TransformerDecoderLayer.forward", "目标自注意力、交叉注意力和前馈分支分别使用残差相加、LayerNorm。"),
        ("seq-position", "tiny-seq2seq", "TinySeq2Seq", "position-embedding", "可学习位置表示", "使用模块", "transformer", "code", "self.positions = nn.Embedding(max_length, 32)", ["E04"], "TinySeq2Seq.add_position", ""),
        ("seq-vocabulary", "tiny-seq2seq", "TinySeq2Seq", "vocabulary-projection", "词表输出投影", "使用输出层", "transformer", "code", "self.vocabulary_projection = nn.Linear(32, tgt_vocab)", ["E04"], "TinySeq2Seq.vocabulary_projection", ""),
    ]:
        edge = {"id": eid, "source": source, "target": target, "source_name": source_name, "target_name": target_name, "relation": verb, "owner_model_id": owner, "kind": "论文背景" if source == "paper-transformer" else "教学实例" if verb == "本卡教学实例" else "结构组成", "source_ids": ids, "locator": locator, "scope": "仅对应本卡教学配置或固定版本框架实现，不推断其他变体、应用论文或性能。", "evidence": [{"section_id": f"{owner}:{slug}", "quote": quote}]}
        if implementation:
            edge.update({"implementation": implementation, "comparison_group": target})
        if target == "linear-head":
            edge["kind"] = "共享结构"
        relations.append(edge)
    # 经典视觉/多模态关系分别指向论文、固定实现或明确的教学配置。
    for eid, source, source_name, target, target_name, verb, owner, slug, quote, ids, locator, implementation in [
        ("vit-paper", "paper-vit", "ViT 原论文", "vit", "ViT", "提出", "vit", "overview", "ViT 将图像分块、映射为 token，再由 Transformer 编码器处理。", ["S16"], "ViT · §3.1 / Figure 1", ""),
        ("vit-example", "vit", "ViT", "tiny-patch-vit", "TinyPatchViT", "本卡教学实例", "vit", "code", "class TinyPatchViT(nn.Module):", ["E05"], "TinyPatchViT", ""),
        ("vit-encoder", "vit", "ViT", "vit-encoder-block", "Torchvision EncoderBlock", "使用模块", "vit", "overview", "它的 EncoderBlock 包含多头自注意力、MLP 与残差相加", ["S25"], "Encoder / EncoderBlock · v0.21.0", ""),
        ("vit-attention", "vit-encoder-block", "Torchvision EncoderBlock", "multihead-attention", "多头注意力", "使用模块", "vit", "relations", "ViT → Torchvision EncoderBlock → 多头注意力、逐位置前馈网络、残差相加。", ["S25"], "EncoderBlock.self_attention", ""),
        ("vit-feedforward", "vit-encoder-block", "Torchvision EncoderBlock", "positionwise-ffn", "逐位置前馈网络", "使用模块", "vit", "relations", "ViT → Torchvision EncoderBlock → 多头注意力、逐位置前馈网络、残差相加。", ["S25"], "MLPBlock", ""),
        ("vit-residual", "vit-encoder-block", "Torchvision EncoderBlock", "residual", "残差相加", "采用机制", "vit", "relations", "该固定实现使用 LayerNorm 后进入注意力或 MLP 分支，并在分支输出后相加", ["S25"], "EncoderBlock.forward", "LayerNorm 后进入自注意力 / MLP 分支，再残差相加；使用 GELU。"),
        ("vit-patches", "tiny-patch-vit", "TinyPatchViT", "patch-embedding", "图像分块投影", "使用模块", "vit", "overview", "使用 conv_proj 完成 patch 投影", ["S25", "E05"], "VisionTransformer.conv_proj / TinyPatchViT", "本例 Conv2d 的 kernel_size=stride=8；32×32 图像产生 16 个 64 维 token。"),
        ("vit-class-token", "tiny-patch-vit", "TinyPatchViT", "class-token", "分类 token", "采用机制", "vit", "code", "cls = self.backbone.class_token.expand(images.size(0), -1, -1)", ["E05", "S25"], "VisionTransformer.class_token", "本例在 16 个 patch 前加入 class token；最终首 token 64 维表示进入 10 类分类头。"),
        ("vit-conv2d", "tiny-patch-vit", "TinyPatchViT", "conv2d", "二维卷积", "包含算子", "vit", "relations", "TinyPatchViT 的 patch 投影使用 Conv2d(3,64,kernel_size=8,stride=8)", ["S25", "E05"], "VisionTransformer.conv_proj", "非重叠 patch 投影，8×8 卷积、步长 8；作用是形成视觉 token。"),
        ("vit-classifier", "tiny-patch-vit", "TinyPatchViT", "linear-head", "Linear 分类头", "使用分类头", "vit", "relations", "分类头为 Linear(64,10)", ["S25", "E05"], "VisionTransformer.heads", "Linear(64,10)：读取本例首 token 的 64 维表示；此配置分类头零初始化。"),
        ("clip-paper", "paper-clip", "CLIP 原论文", "clip", "CLIP", "提出", "clip", "overview", "CLIP 分别编码图像与文本，将二者投影到共享表示空间", ["S18"], "CLIP · §2.3 / Figure 3", ""),
        ("clip-vision", "clip", "CLIP", "clip-visual-vit", "CLIP 的 ViT 视觉编码器", "可选视觉主干", "clip", "overview", "官方 CLIP 可选 ViT 视觉编码器", ["S18", "S26"], "CLIP.__init__ / VisionTransformer · a1d0717", ""),
        ("clip-vit-family", "clip-visual-vit", "CLIP 的 ViT 视觉编码器", "vit", "ViT", "采用架构", "clip", "relations", "CLIP → CLIP 的 ViT 视觉编码器 → ViT 架构家族。", ["S18", "S26"], "CLIP · §2.4–2.5 / VisionTransformer", ""),
        ("clip-text", "clip", "CLIP", "clip-text-transformer", "CLIP 文本 Transformer", "使用模块", "clip", "overview", "文本侧使用带因果遮罩的 Transformer", ["S26"], "CLIP.transformer / build_attention_mask", ""),
        ("clip-text-attention", "clip-text-transformer", "CLIP 文本 Transformer", "multihead-attention", "多头注意力", "使用模块", "clip", "relations", "CLIP 的文本 Transformer 使用多头自注意力与残差相加。", ["S26"], "ResidualAttentionBlock.attn", ""),
        ("clip-text-residual", "clip-text-transformer", "CLIP 文本 Transformer", "residual", "残差相加", "采用机制", "clip", "relations", "它采用 LayerNorm 后进入分支，注意力还有因果遮罩", ["S26"], "ResidualAttentionBlock.forward", "官方文本侧 LayerNorm 后进入注意力或 MLP 分支，再残差相加；注意力使用因果遮罩。"),
        ("clip-alignment", "clip", "CLIP", "contrastive-alignment", "图文对比学习", "采用学习目标", "clip", "formula", "对角线作为正对，两个方向的交叉熵取平均。", ["S18"], "CLIP · §2.3 / Figure 3", ""),
        ("clip-similarity", "clip", "CLIP", "normalized-similarity", "归一化余弦匹配", "采用匹配方式", "clip", "relations", "CLIP 的核心匹配包含投影后归一化与缩放余弦相似度。", ["S26"], "CLIP.forward", ""),
        ("clip-example", "clip", "CLIP", "tiny-dual-encoder", "TinyDualEncoder", "本卡教学实例", "clip", "code", "class TinyDualEncoder(nn.Module):", ["E06"], "TinyDualEncoder；仅示意双编码器接口", ""),
        ("clip-class-token", "clip-visual-vit", "CLIP 的 ViT 视觉编码器", "class-token", "分类 token", "采用机制", "clip", "relations", "该视觉主干有 patch 投影与 class token，但输出目标是共享图文表示", ["S26"], "VisionTransformer.class_embedding / proj", "读取视觉首 token 并投影到图文共享维数；不是输出固定类别分数。"),
        ("clip-patches", "clip-visual-vit", "CLIP 的 ViT 视觉编码器", "patch-embedding", "图像分块投影", "使用模块", "clip", "relations", "该视觉主干有 patch 投影与 class token", ["S26"], "VisionTransformer.conv1", "官方 ViT 视觉主干使用非重叠 Conv2d 投影；尺寸、patch 大小与宽度随具体配置变化。"),
        ("clip-visual-conv2d", "clip-visual-vit", "CLIP 的 ViT 视觉编码器", "conv2d", "二维卷积", "包含算子", "clip", "relations", "该视觉主干有 patch 投影与 class token", ["S26"], "VisionTransformer.conv1 · kernel_size=stride=patch_size", "官方视觉 ViT 中的 Conv2d 形成 patch token；没有把 CNN 分类器作为这个变体的主干。"),
    ]:
        kind = "论文背景" if source.startswith("paper-") else "教学实例" if verb == "本卡教学实例" else "模型归属" if verb == "采用架构" else "结构组成"
        edge = {"id": eid, "source": source, "target": target, "source_name": source_name, "target_name": target_name, "relation": verb, "owner_model_id": owner, "kind": kind, "source_ids": ids, "locator": locator, "scope": "论文机制、固定实现和教学配置按来源区分；不推断共享权重、真实任务表现或未核对的应用。", "evidence": [{"section_id": f"{owner}:{slug}", "quote": quote}]}
        if implementation:
            edge.update({"implementation": implementation, "comparison_group": target})
        if target == "linear-head":
            edge["kind"] = "共享结构"
        relations.append(edge)
    for eid, source, source_name, target, target_name, verb, owner, slug, quote, ids, locator, implementation in [
        ("ae-example", "autoencoder", "Autoencoder", "tiny-dense-ae", "TinyDenseAutoencoder", "本卡教学实例", "autoencoder", "code", "class TinyDenseAutoencoder(nn.Module):", ["E07"], "TinyDenseAutoencoder", ""),
        ("ae-encoder", "autoencoder", "Autoencoder", "dense-ae-encoder", "DenseEncoder", "使用模块", "autoencoder", "overview", "本例 TinyDenseAutoencoder 使用 DenseEncoder 与 DenseDecoder。", ["E07"], "TinyDenseAutoencoder.encoder", ""),
        ("ae-decoder", "autoencoder", "Autoencoder", "dense-ae-decoder", "DenseDecoder", "使用模块", "autoencoder", "overview", "本例 TinyDenseAutoencoder 使用 DenseEncoder 与 DenseDecoder。", ["E07"], "TinyDenseAutoencoder.decoder", ""),
        ("ae-bottleneck", "dense-ae-encoder", "DenseEncoder", "latent-bottleneck", "潜在瓶颈", "采用机制", "autoencoder", "relations", "Autoencoder → DenseEncoder → 潜在瓶颈", ["E07", "S10"], "编码器末层 / §14.1", ""),
        ("ae-reconstruction", "dense-ae-decoder", "DenseDecoder", "input-reconstruction", "输入重构", "输出重构值", "autoencoder", "relations", "Autoencoder → DenseDecoder → 输入重构", ["E07"], "解码器输出 32 维", ""),
        ("ae-loss", "autoencoder", "Autoencoder", "mse-reconstruction", "均方重构误差", "采用教学损失", "autoencoder", "code", "loss = F.mse_loss(reconstruction, values)", ["E07", "S30"], "F.mse_loss · mean", ""),
        ("ae-encoder-linear", "dense-ae-encoder", "DenseEncoder", "linear-map", "线性映射", "包含算子", "autoencoder", "code", "nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 4),", ["E07", "S29"], "TinyDenseAutoencoder.encoder", "32→16→4，两个 Linear 之间有 ReLU；最终输出潜在表示，不是类别分数。"),
        ("ae-decoder-linear", "dense-ae-decoder", "DenseDecoder", "linear-map", "线性映射", "包含算子", "autoencoder", "code", "nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, 32),", ["E07", "S29"], "TinyDenseAutoencoder.decoder", "4→16→32，两个 Linear 之间有 ReLU；最终重构输入，末层不使用 Sigmoid 或 softmax。"),
        ("lstm-example", "lstm", "LSTM", "tiny-bi-lstm", "TinyBiLSTM", "本卡教学实例", "lstm", "code", "class TinyBiLSTM(nn.Module):", ["E08"], "TinyBiLSTM", ""),
        ("lstm-core", "lstm", "LSTM", "bilstm-core", "双向 LSTM 主干", "使用模块", "lstm", "overview", "本例 TinyBiLSTM 包含一个双向 LSTM 主干和 Linear 分类头", ["E08", "S11"], "TinyBiLSTM.core", ""),
        ("lstm-gates", "bilstm-core", "双向 LSTM 主干", "lstm-gates", "门控更新", "采用机制", "lstm", "relations", "LSTM → 双向 LSTM 主干 → 隐藏/记忆状态与门控更新。", ["S11"], "LSTM 文档公式 · 无投影", ""),
        ("lstm-states", "bilstm-core", "双向 LSTM 主干", "recurrent-state", "隐藏/记忆状态", "输出状态", "lstm", "relations", "LSTM → 双向 LSTM 主干 → 隐藏/记忆状态与门控更新。", ["S11", "E08"], "output / h_n / c_n", ""),
        ("lstm-final-summary", "tiny-bi-lstm", "TinyBiLSTM", "bidirectional-summary", "双向最终隐藏状态", "采用聚合方式", "lstm", "code", "features = torch.cat([h_n[-2], h_n[-1]], dim=-1)", ["E08", "S11"], "TinyBiLSTM.forward", ""),
        ("lstm-classifier", "tiny-bi-lstm", "TinyBiLSTM", "linear-head", "Linear 分类头", "使用分类头", "lstm", "code", "self.classifier = nn.Linear(16, 3)", ["E08"], "TinyBiLSTM.classifier", "Linear(16,3)：将整条序列的双向最终表示转换为三类 logits；不是逐位置词表分数。"),
        ("lstm-affine", "tiny-bi-lstm", "TinyBiLSTM", "linear-map", "线性映射", "包含算子", "lstm", "code", "self.classifier = nn.Linear(16, 3)", ["E08", "S29"], "TinyBiLSTM.classifier", "16→3 的 Linear 输出序列类别分数，与自编码器重构/潜在层的输出目标不同。"),
        ("lstm-packed", "tiny-bi-lstm", "TinyBiLSTM", "packed-sequence", "PackedSequence", "使用变长接口", "lstm", "code", "packed = pack_padded_sequence(values, lengths.cpu(), batch_first=True, enforce_sorted=False)", ["E08", "S28"], "pack_padded_sequence", ""),
        ("packed-lengths", "packed-sequence", "PackedSequence", "sequence-length", "有效序列长度", "使用长度信息", "lstm", "masks", "lengths 为 CPU 整数向量", ["S28", "E08"], "pack_padded_sequence lengths", ""),
        ("lstm-forward-state", "bilstm-core", "双向 LSTM 主干", "forward-final-state", "前向最终状态", "使用方向", "lstm", "io", "最后一层的前向和反向最终状态为 h_n[-2] 与 h_n[-1]", ["S11", "E08"], "h_n[-2] · 两层双向教学配置", ""),
        ("lstm-backward-state", "bilstm-core", "双向 LSTM 主干", "backward-final-state", "反向最终状态", "使用方向", "lstm", "io", "最后一层的前向和反向最终状态为 h_n[-2] 与 h_n[-1]", ["S11", "E08"], "h_n[-1] · 两层双向教学配置", ""),
        ("lstm-cnn-layout", "lstm", "LSTM", "cnn-1d", "1D CNN", "可比较序列布局", "lstm", "relations", "本库 1D CNN 使用 `[B,C,L]` 布局，E08 使用 `[B,L,F]`。", ["E01", "E08"], "教学实例输入布局；特征/通道语义需一致", ""),
    ]:
        kind = "教学实例" if verb == "本卡教学实例" else "输入布局" if verb == "可比较序列布局" else "结构组成"
        edge = {"id": eid, "source": source, "target": target, "source_name": source_name, "target_name": target_name, "relation": verb, "owner_model_id": owner, "kind": kind, "source_ids": ids, "locator": locator, "scope": "仅支持固定接口或本卡教学配置；潜在表示、重构值、序列状态与类别分数按用途区分。", "evidence": [{"section_id": f"{owner}:{slug}", "quote": quote}]}
        if implementation:
            edge.update({"implementation": implementation, "comparison_group": target})
        if target == "linear-head":
            edge["kind"] = "共享结构"
        relations.append(edge)
    for eid, source, source_name, target, target_name, owner, slug, quote, verb in [
        ('detr-cnn', 'detr', 'DETR', 'detr-cnn-backbone', 'DETR 的 CNN 骨干', 'detr', 'overview', 'DETR先用CNN提取图像特征，再由Transformer编码器/解码器完成集合预测。', '使用视觉骨干'),
        ('detr-transformer', 'detr', 'DETR', 'detr-transformer', 'DETR 编码器/解码器', 'detr', 'overview', 'DETR的Transformer编码器/解码器处理图像特征与object queries，不是翻译任务。', '使用检测模块'),
        ('detr-transformer-family', 'detr-transformer', 'DETR 编码器/解码器', 'transformer', 'Transformer', 'detr', 'relations', 'DETR的编码器/解码器沿用Transformer架构，但解码端并行预测目标集合。', '采用架构'),
        ('detr-queries', 'detr-transformer', 'DETR 编码器/解码器', 'detr-object-queries', 'Object queries', 'detr', 'mechanism', 'object queries是可学习的目标槽位查询，不是自然语言问题，也不是从真实目标数量直接生成的查询。', '使用查询'),
        ('detr-task', 'detr', 'DETR', 'object-detection', '目标检测', 'detr', 'applications', '目标检测输出类别和边界框，保留每个目标的位置。', '用于任务'),
        ('llava-vision', 'llava', 'LLaVA', 'llava-clip-vision', 'LLaVA 的 CLIP 视觉分支', 'llava', 'overview', 'LLaVA将CLIP视觉编码器、可训练投影层和Vicuna语言模型连接起来，根据图像与语言指令生成文本回答。', '使用视觉分支'),
        ('llava-clip-source', 'llava-clip-vision', 'LLaVA 的 CLIP 视觉分支', 'clip', 'CLIP', 'llava', 'relations', 'CLIP是视觉编码器的预训练来源，LLaVA不使用CLIP文本塔产生回答', '来自预训练模型'),
        ('llava-projector', 'llava', 'LLaVA', 'llava-projector', 'LLaVA 原始投影层', 'llava', 'io', '原始LLaVA用线性投影把视觉token映射到Vicuna的词嵌入空间。', '使用连接器'),
        ('llava-linear', 'llava-projector', 'LLaVA 原始投影层', 'linear-map', '线性映射', 'llava', 'relations', '原始连接器是线性投影层，用于视觉token到语言嵌入的维度对齐', '进行维度对齐'),
        ('llava-language', 'llava', 'LLaVA', 'vicuna-language-model', 'Vicuna 语言模型', 'llava', 'relations', '文本回答由 Vicuna 语言模型生成。', '使用语言模块'),
        ('llava-task', 'llava', 'LLaVA', 'visual-question-answering', '视觉问答', 'llava', 'applications', '图像与语言指令进入模型，回答以文本形式输出', '用于任务'),
        ('dinov2-backbone', 'dinov2', 'DINOv2', 'dinov2-vit-backbone', 'DINOv2 的 ViT 骨干', 'dinov2', 'overview', 'DINOv2采用ViT作为纯视觉骨干，通过图像自监督预训练获得可迁移特征。', '使用视觉骨干'),
        ('dinov2-vit-family', 'dinov2-vit-backbone', 'DINOv2 的 ViT 骨干', 'vit', 'ViT', 'dinov2', 'relations', 'DINOv2的视觉骨干属于ViT架构家族，训练目标与普通有监督图像分类不同。', '采用架构'),
        ('dinov2-training', 'dinov2', 'DINOv2', 'visual-self-distillation', '视觉自监督蒸馏', 'dinov2', 'formula', '基础自监督过程中，student 通过目标函数更新，teacher 采用过往 student 参数的指数移动平均。', '采用学习目标'),
        ('dinov2-features', 'dinov2', 'DINOv2', 'visual-token-features', '全局与局部视觉特征', 'dinov2', 'io', 'DINOv2骨干给出视觉特征，不会仅凭一个特征向量自动得到任意任务的完整答案。', '输出表示'),
        ('dinov2-dense-task', 'dinov2', 'DINOv2', 'dense-visual-prediction', '分割与深度读出', 'dinov2', 'applications', '下游头与读出流程负责产生像素级分割或深度结果', '连接下游任务'),
        ('dinov2-head', 'dinov2', 'DINOv2', 'downstream-task-head', '下游任务头', 'dinov2', 'relations', 'DINOv2作为视觉表示骨干，任务头负责把表示映射为目标任务输出。', '连接任务读出'),
    ]:
        sid = {'detr': 'S32', 'llava': 'S33', 'dinov2': 'S34'}[owner]
        relations.append({'id': eid, 'source': source, 'source_name': source_name, 'target': target, 'target_name': target_name,
                          'relation': verb, 'kind': '论文应用', 'owner_model_id': owner, 'source_ids': [sid],
                          'locator': '固定原论文方法与当前整理章节', 'scope': '仅解释已核对的组成、输入输出和任务连接；不推断共享权重、部署能力或当前SOTA。',
                          'evidence': [{'section_id': f'{owner}:{slug}', 'quote': quote}]})
    relations.extend(recent_relations())
    upsert_records(KnowledgeModel, models)
    upsert_records(KnowledgeSection, sections)
    upsert_records(KnowledgeSource, sources)
    from backend.infra.public_sources import enrich_relations
    upsert_records(KnowledgeRelation, enrich_relations(relations))


def split_blocks(text, max_chars=900):
    """段落优先；代码、块公式完整保留，超长保护块允许超过软目标。"""
    blocks, buf, fence, math = [], [], None, False
    for line in text.splitlines():
        marker = re.match(r"^\s*(`{3,}|~{3,})(.*)$", line)
        if marker:
            if fence is None:
                fence = marker[1]
            elif marker[1][0] == fence[0] and len(marker[1]) >= len(fence) and not marker[2].strip():
                fence = None
        if line.strip() == "$$" and not fence:
            math = not math
        if not line.strip() and not fence and not math:
            if buf:
                blocks.append("\n".join(buf))
                buf = []
        else:
            buf.append(line)
    if buf:
        blocks.append("\n".join(buf))
    chunks, current = [], ""
    for block in blocks:
        if current and len(current) + len(block) + 2 > max_chars:
            chunks.append(current)
            current = ""
        current = f"{current}\n\n{block}" if current else block
    if current:
        chunks.append(current)
    return chunks
