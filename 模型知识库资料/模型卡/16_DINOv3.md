# DINOv3：稳定密集视觉表示

模型ID：dinov3。固定依据2508.10104v1，2025-08-13。基础视觉预训练、模型家族和后续文本对齐分开理解。

## 概览与核心机制

DINOv3先进行视觉自监督预训练，并增加后续正则、分辨率适配和模型蒸馏。Gram anchoring约束patch特征之间的相似关系，改善长训练中密集特征退化。[S42：§3.2、§4](https://arxiv.org/pdf/2508.10104v1#page=12)

## 输入输出与数据流

基础视觉骨干接收图像，不要求配对文本监督。以B、P、D表示批次、patch数量和表示维数：全局表示可写作`[B,D]`，局部表示可写作`[B,P,D]`，具体结构取决于变体。骨干输出视觉特征，分类、分割和深度等任务仍需对应读出或任务头。

核心视觉模型与另行对齐文本的dino.txt不是同一个输入输出接口；基础DINOv3不会因为存在文本对齐研究就直接生成自然语言回答。[S42：§5.3](https://arxiv.org/pdf/2508.10104v1#page=16)

## 核心公式

用每行一个patch的归一化表示矩阵F解释Gram关系：

$$
G(F)=FF^{\top}.
$$

它描述patch之间的成对相似度。Gram anchoring让student的这种相似关系接近固定Gram teacher，不是强迫所有单个向量逐元素相同。完整损失、teacher刷新与高分辨率对齐须看原论文，不把该说明式当成完整训练代码。[S42：§4.2–4.3](https://arxiv.org/pdf/2508.10104v1#page=12)

## 最小代码

原创结构伪代码，不是作者可以直接运行的API。

```text
global_features, patch_features = visual_backbone(images)
global_prediction = global_task_head(global_features)
dense_prediction = dense_task_head(patch_features, spatial_layout)
```

仅执行第一行只能得到特征；后两行的头、预处理与目标任务必须明确配置。

## 与相关模型的联系

DINOv3沿DINOv2的视觉自监督路线扩展训练，并用Gram anchoring改善密集特征。主要大教师采用ViT；蒸馏后的DINOv3家族也包含ConvNeXt，不能说所有变体都是ViT。基本视觉训练与后续dino.txt文本对齐分开处理。[S42：§3.2、§5.2、§7.2](https://arxiv.org/pdf/2508.10104v1#page=31)

## 应用流程与使用边界

图像→视觉骨干→全局或密集特征→目标任务读出。论文涉及分类、对应关系、分割、深度与视觉几何等设置；有的使用非参数读出，有的训练任务头。冻结骨干不意味着所有应用完全不需要任务设置或学习。

## 架构图与论文插图

Fig.12展示一个teacher为多个student共享推理的蒸馏流程。它是训练调度图，不是普通图像推理网络的逐层展开。读图时区分teacher推理、student更新与同步屏障。[原图第16页](https://arxiv.org/pdf/2508.10104v1#page=16)

## 来源与定位

- S42：[固定论文v1](https://arxiv.org/pdf/2508.10104v1)。
- S51：[作者DINOv3仓库](https://github.com/facebookresearch/dinov3)，入口与模型使用资料。
- A08：原创结构伪代码，未执行任务模型。

## 可继续追问

“Gram约束的是单个特征还是成对关系？”“所有变体都是ViT吗？”“基础视觉输入是否需要文本？”“能直接输出深度图或文字吗？”
