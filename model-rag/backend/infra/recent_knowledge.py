"""固定论文版本的近期方法目录；正文来自对应模型卡。"""
RECENT_TOPICS = [
    {'id': 'mamba', 'name': 'Mamba', 'aliases': ['mamba'], 'file': '12_Mamba.md', 'category': '状态空间模型', 'kind': '序列架构', 'description': '通过输入相关的状态更新选择保留序列信息，骨干输出仍是逐位置表示。', 'tags': ['序列', '选择性状态空间'], 'source_ids': ['S38', 'S43', 'A04']},
    {'id': 'mamba2', 'name': 'Mamba-2', 'aliases': ['mamba-2', 'mamba2', 'mamba 2'], 'file': '13_Mamba2.md', 'category': '状态空间模型', 'kind': '序列架构', 'description': '利用结构化状态空间对偶连接选择性SSM与结构化注意力计算。', 'tags': ['SSD', '序列'], 'source_ids': ['S39', 'S43', 'A05']},
    {'id': 'mamba3', 'name': 'Mamba-3', 'aliases': ['mamba-3', 'mamba3', 'mamba 3'], 'file': '14_Mamba3.md', 'category': '状态空间模型', 'kind': '序列架构', 'description': '2026年方法：结合更丰富的离散化递推、复值状态与MIMO设计。', 'tags': ['2026', '状态跟踪'], 'source_ids': ['S40', 'S43', 'A06']},
    {'id': 'siglip2', 'name': 'SigLIP 2', 'aliases': ['siglip 2', 'siglip2', 'siglip-2'], 'file': '15_SigLIP2.md', 'category': '视觉与多模态', 'kind': '图文编码方法', 'description': '2025年图文编码器：结合sigmoid对齐、字幕训练和视觉自监督。', 'tags': ['2025', '多模态', '图文对齐'], 'source_ids': ['S41', 'S50', 'A07']},
    {'id': 'dinov3', 'name': 'DINOv3', 'aliases': ['dinov3', 'dino v3'], 'file': '16_DINOv3.md', 'category': '视觉与多模态', 'kind': '视觉表示方法', 'description': '2025年视觉自监督方法，以Gram anchoring稳定密集特征，并提供多种蒸馏骨干。', 'tags': ['2025', '自监督', '密集特征'], 'source_ids': ['S42', 'S51', 'A08']},
]

PAPER_RECIPES = [
    {'key': 'mamba-paper', 'model': 'mamba', 'title': 'Mamba 原始论文 · v2 · 选择性状态空间与架构', 'version': '2312.00752v2', 'url': 'https://arxiv.org/pdf/2312.00752v2', 'filename': 'Mamba_2312.00752v2.pdf', 'pages': '2-3,5-8', 'type': 'paper'},
    {'key': 'mamba2-paper', 'model': 'mamba2', 'title': 'Mamba-2 原始论文 · v1 · SSD与模块设计', 'version': '2405.21060v1', 'url': 'https://arxiv.org/pdf/2405.21060v1', 'filename': 'Mamba2_2405.21060v1.pdf', 'pages': '3-4,16-18,22-24', 'type': 'paper'},
    {'key': 'mamba3-paper', 'model': 'mamba3', 'title': 'Mamba-3 原始论文 · v1 · 递推、复值状态与MIMO', 'version': '2603.15569v1', 'url': 'https://arxiv.org/pdf/2603.15569v1', 'filename': 'Mamba3_2603.15569v1.pdf', 'pages': '2-11', 'type': 'paper'},
    {'key': 'siglip2-paper', 'model': 'siglip2', 'title': 'SigLIP 2 原始论文 · v1 · 训练流程与图文接口', 'version': '2502.14786v1', 'url': 'https://arxiv.org/pdf/2502.14786v1', 'filename': 'SigLIP2_2502.14786v1.pdf', 'pages': '1-4', 'type': 'paper'},
    {'key': 'dinov3-paper', 'model': 'dinov3', 'title': 'DINOv3 原始论文 · v1 · 视觉自监督与Gram anchoring', 'version': '2508.10104v1', 'url': 'https://arxiv.org/pdf/2508.10104v1', 'filename': 'DINOv3_2508.10104v1.pdf', 'pages': '9-16,31', 'type': 'paper'},
]


def recent_relations():
    rows = [
        ('mamba-selective', 'mamba', 'Mamba', 'selective-ssm', '选择性状态空间', 'mamba', 'overview', 'Mamba使用选择性状态空间模型，将部分状态空间参数变成当前输入的函数。', '使用机制'),
        ('mamba-conv', 'mamba', 'Mamba', 'conv1d', '一维卷积', 'mamba', 'relations', '原始Mamba块包含短一维卷积、选择性SSM、门控和输入输出投影。', '包含算子'),
        ('mamba2-evolution', 'mamba2', 'Mamba-2', 'mamba', 'Mamba', 'mamba2', 'relations', 'Mamba-2在Mamba的选择性SSM基础上改进模块与算法。', '改进方法'),
        ('mamba2-ssd', 'mamba2', 'Mamba-2', 'ssd-duality', '结构化状态空间对偶', 'mamba2', 'overview', 'Mamba-2利用结构化状态空间对偶（SSD）连接SSM与特定结构的注意力计算。', '采用框架'),
        ('mamba3-evolution', 'mamba3', 'Mamba-3', 'mamba2', 'Mamba-2', 'mamba3', 'relations', 'Mamba-3沿状态空间序列模型路线改进Mamba-2，但不是简单增大状态维数。', '改进方法'),
        ('mamba3-complex', 'mamba3', 'Mamba-3', 'complex-state', '复值状态更新', 'mamba3', 'overview', 'Mamba-3结合更丰富的离散化递推、复值状态更新和多输入多输出（MIMO）设计。', '采用机制'),
        ('mamba3-mimo', 'mamba3', 'Mamba-3', 'mimo-state', 'MIMO状态空间', 'mamba3', 'io', 'MIMO指状态空间层内部的多输入多输出通道，不表示必须输入图像和文本两种模态。', '使用内部结构'),
        ('siglip2-vit', 'siglip2', 'SigLIP 2', 'vit', 'ViT', 'siglip2', 'relations', 'SigLIP 2的视觉编码器采用ViT，文本编码器产生与图像对齐的表示。', '使用视觉架构'),
        ('siglip2-sigmoid', 'siglip2', 'SigLIP 2', 'sigmoid-pair-loss', '图文sigmoid损失', 'siglip2', 'formula', 'sigmoid对每个图文配对分别判断正负，不要求对所有候选文本做softmax归一化。', '采用学习目标'),
        ('siglip2-caption', 'siglip2', 'SigLIP 2', 'caption-training', '字幕训练', 'siglip2', 'applications', '字幕解码器是训练时增加的学习目标，不意味着任意发布的双编码器接口都能直接生成回答。', '使用训练目标'),
        ('dinov3-evolution', 'dinov3', 'DINOv3', 'dinov2', 'DINOv2', 'dinov3', 'relations', 'DINOv3沿DINOv2的视觉自监督路线扩展训练，并用Gram anchoring改善密集特征。', '改进方法'),
        ('dinov3-vit', 'dinov3', 'DINOv3', 'vit', 'ViT', 'dinov3', 'relations', '主要大教师采用ViT；蒸馏后的DINOv3家族也包含ConvNeXt，不能说所有变体都是ViT。', '使用教师架构'),
        ('dinov3-gram', 'dinov3', 'DINOv3', 'gram-anchoring', 'Gram anchoring', 'dinov3', 'overview', 'Gram anchoring约束patch特征之间的相似关系，改善长训练中密集特征退化。', '采用正则'),
        ('dinov3-head', 'dinov3', 'DINOv3', 'downstream-task-head', '下游任务头', 'dinov3', 'io', '骨干输出视觉特征，分类、分割和深度等任务仍需对应读出或任务头。', '连接任务读出'),
    ]
    sources = {'mamba': 'S38', 'mamba2': 'S39', 'mamba3': 'S40', 'siglip2': 'S41', 'dinov3': 'S42'}
    return [{'id': eid, 'source': source, 'source_name': name, 'target': target, 'target_name': target_name,
             'owner_model_id': owner, 'relation': verb, 'kind': '方法演进' if verb == '改进方法' else '结构组成',
             'source_ids': [sources[owner]], 'locator': '固定版本原论文方法与模型卡章节',
             'scope': '按固定版本解释组成和演进；不推断通用性能、共享权重或已部署推理。',
             'evidence': [{'section_id': f'{owner}:{slug}', 'quote': quote}]} for eid, source, name, target, target_name, owner, slug, quote, verb in rows]
