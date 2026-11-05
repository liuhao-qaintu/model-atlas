# DETR：从图像到无序目标集合

模型 ID：detr。本文解释2020年原始DETR，以论文2005.12872v3和作者仓库固定提交29901c5为依据。文中的小张量和数字是为讲解编写的例子；它们不是论文实验结果。作者代码的归档提交晚于论文发表，不把提交日期当成模型发表日期。

## 概览与核心机制

### 它解决的是什么问题

假设一张照片里有两只鸟。图像分类回答“这张图属于鸟类”，目标检测还要回答“每只鸟分别在哪里”。DETR把输出写成若干条“类别＋边界框”记录，记录的先后顺序没有任务含义：先写左边的鸟还是先写右边的鸟，都应描述同一组目标。这里的“集合”指这种无序的输出目标，不表示网络会直接返回Python的set。[原论文§3.1](https://arxiv.org/pdf/2005.12872v3#page=5)

DETR先用CNN提取图像特征，再由Transformer编码器/解码器完成集合预测。它预留固定数量的检测槽位，每个槽位都给出类别分数和一个框；没有对应真实目标的槽位应学习输出no-object。作者的说明文章用“图像到集合”介绍这一设计。[作者介绍](https://ai.meta.com/blog/end-to-end-object-detection-with-transformers/)

### 为什么不能只接一个图像分类头

整图分类头通常把空间特征汇总后给出一组类别分数，难以直接表示多个目标各自的位置。DETR保留CNN的二维特征图，让不同图像位置之间交换信息，再让多个槽位分别读取全图信息。原论文采用ResNet-50/101作为骨干；不能因此说原始实验使用了本库的ResNet-18。[原论文§3.2、§4](https://arxiv.org/pdf/2005.12872v3#page=6)

### 先记住三个角色

| 模块 | 要解决的问题 | 输出的含义 |
|---|---|---|
| CNN骨干 | 图像各处有哪些可用的视觉模式 | 带空间位置的特征图 |
| Transformer | 图像各位置如何联系；各检测槽位该读取什么 | 带全图上下文的目标表示 |
| 预测头＋集合训练目标 | 表示对应什么类别和框；如何避免大家都报同一目标 | 固定数量的候选记录，训练时与真值一一匹配 |

DETR的Transformer编码器/解码器处理图像特征与object queries，不是翻译任务。理解DETR的关键还包括一对一集合匹配，不能仅概括为“把CNN换成Transformer”。[原论文Fig.2及§3](https://arxiv.org/pdf/2005.12872v3#page=7)

## 结构与原理

### 1. CNN保留的是空间特征

CNN末端的每个位置对应一个视觉特征向量。`1×1 Conv`把每个位置的通道数从C映射到d，主要用于匹配Transformer隐藏维数；它不会在这一步把整张图平均为一个向量。展平只是把二维位置排成序列，原位置与序列索引仍可对应。[作者Backbone和投影层](https://github.com/facebookresearch/detr/blob/29901c51d7fe8712168b8d0d64351170bc0f83e0/models/backbone.py#L58-L109)、[DETR.__init__](https://github.com/facebookresearch/detr/blob/29901c51d7fe8712168b8d0d64351170bc0f83e0/models/detr.py#L30-L38)

### 2. 图像位置与padding不是一回事

位置表示告诉注意力某个特征来自图像哪个位置；padding mask则标出为了组成同一批次而补齐的无效位置。前者提供坐标线索，后者防止把补齐区域当成正常图像内容。原始实现的图像mask中，True表示padding；不能把它理解成“这个位置有目标”。[位置编码实现](https://github.com/facebookresearch/detr/blob/29901c51d7fe8712168b8d0d64351170bc0f83e0/models/position_encoding.py#L25-L43)、[前向接口](https://github.com/facebookresearch/detr/blob/29901c51d7fe8712168b8d0d64351170bc0f83e0/models/detr.py#L40-L60)

### 3. 编码器让图像位置交换信息

编码器的自注意力以图像特征序列为输入：一个位置可以读取其他位置的内容。作者默认的post-norm编码层在Q、K中加入空间位置表示，而V仍是特征内容；随后还有残差、归一化和前馈网络。不要把它简化成“只做一次注意力”或把位置向量当作检测框坐标。[TransformerEncoderLayer.forward_post](https://github.com/facebookresearch/detr/blob/29901c51d7fe8712168b8d0d64351170bc0f83e0/models/transformer.py#L134-L147)

### 4. object queries是可学习的槽位身份

object queries是可学习的目标槽位查询，不是自然语言问题，也不是从真实目标数量直接生成的查询。可以把它们暂时理解为N张身份不同、等待填内容的检测记录；这个比喻不表示“第1张永远负责鸟”或“第2张固定负责左上角”。实际用途由训练形成。[原论文§3.2](https://arxiv.org/pdf/2005.12872v3#page=7)

作者代码把可学习的`query_embed`复制到批次维度，另创建全零的初始`tgt`内容。查询身份通过`query_pos`进入注意力；解码器并不是把N个类别标签或N个真实框直接喂进去。即使初始内容是零，不同query身份仍能在交叉注意力中读取不同的图像信息。[Transformer.forward](https://github.com/facebookresearch/detr/blob/29901c51d7fe8712168b8d0d64351170bc0f83e0/models/transformer.py#L42-L53)

### 5. 解码器有两种注意力

自注意力让检测槽位之间交换信息；交叉注意力让槽位从编码后的图像中读取信息。后者的Q来自槽位内容加query身份，K来自图像memory加空间位置，V来自memory。它们使用注意力这一共同运算，但读取对象不同。[TransformerDecoderLayer.forward_post](https://github.com/facebookresearch/detr/blob/29901c51d7fe8712168b8d0d64351170bc0f83e0/models/transformer.py#L192-L213)

原始DETR不按“先预测第1个目标，再把它作为输入预测第2个目标”的自回归顺序工作；同一解码层并行处理N个槽位。网络仍有多层解码器，不能把“并行预测”理解为没有逐层计算。[原论文§3.2](https://arxiv.org/pdf/2005.12872v3#page=7)

### 6. 两个预测头回答不同问题

分类头把每个槽位表示映射为类别logits，最后一类是no-object；边框头是三层MLP，末端sigmoid输出归一化的中心与宽高。头的参数在不同槽位之间共享，并不是给100个槽位各训练100套独立分类器。[DETR输出与MLP](https://github.com/facebookresearch/detr/blob/29901c51d7fe8712168b8d0d64351170bc0f83e0/models/detr.py#L30-L66)

## 输入输出与数据流

B是批量，C是CNN输出通道，H、W是特征图高宽，d是隐藏维数，N是检测槽位数。K暂指连续编号的真实类别数量；下面会单独说明COCO实现的例外。

| 阶段 | 形状 | 应怎样理解 |
|---|---|---|
| 批次图像 | `[B,3,H0,W0]` | 图像可能经过缩放；变尺寸批次还需padding mask |
| CNN最后特征 | `[B,C,H,W]` | H、W是特征图尺寸，不是原图像素尺寸 |
| 1×1通道投影 | `[B,d,H,W]` | 每个空间位置变为d维向量 |
| 作者Transformer内展平 | `[H*W,B,d]` | 此处采用序列在前的实现布局，不是batch_first |
| query身份 / 初始内容 | `[N,B,d]` | query可学习，初始tgt为零 |
| 各层解码输出 | `[L_dec,B,N,d]` | L_dec表示解码层数；最终读取最后一层 |
| 末层pred_logits | `[B,N,K+1]` | 未归一化类别分数，包含no-object |
| 末层pred_boxes | `[B,N,4]` | `cx,cy,w,h`，相对于每张有效图像尺寸归一化 |

这些形状对应作者`DETR.forward`与`Transformer.forward`的运算。若使用batch_first接口，轴顺序可能不同，必须对照实际API。[输出和展平源码](https://github.com/facebookresearch/detr/blob/29901c51d7fe8712168b8d0d64351170bc0f83e0/models/transformer.py#L42-L53)、[框坐标转换](https://github.com/facebookresearch/detr/blob/29901c51d7fe8712168b8d0d64351170bc0f83e0/util/box_ops.py#L9-L12)

### 用一个小例子理解“展平不等于丢掉位置”

以下教学例只检查布局，不执行完整检测器。把C=8、d=16和小特征图代入，是为方便观察；不是论文网络配置。

```python
import torch
from torch import nn

torch.manual_seed(0)
batch, height, width = 2, 2, 3
feature_map = torch.arange(96, dtype=torch.float32).reshape(batch, 8, height, width)
projection = nn.Conv2d(8, 16, kernel_size=1)
projected = projection(feature_map)
sequence = projected.flatten(2).permute(2, 0, 1)

query_identity = nn.Embedding(3, 16).weight
queries = query_identity[:, None, :].expand(-1, batch, -1)
initial_content = torch.zeros_like(queries)

assert sequence.shape == (6, 2, 16)
assert queries.shape == initial_content.shape == (3, 2, 16)
assert torch.allclose(sequence[5, 0], projected[0, :, 1, 2])
assert initial_content.count_nonzero().item() == 0
```

第5个序列位置仍可对回第1行第2列；位置编码随后帮助注意力使用这种空间信息。`permute`只是改轴顺序，不会把特征自动变成类别或坐标。相关布局来自作者实现，上面的数值与断言是本卡自编演示。

### N、K与图像尺寸分别决定什么

原论文附录A.4的实验使用N=100。N是可容纳的预测槽位数，不是每张图必须有100个目标；超过槽位容量的目标也不可能仅靠该固定输出同时表示。H、W决定图像memory长度，N决定检测输出条数，二者不要混用。[论文附录A.4](https://arxiv.org/pdf/2005.12872v3#page=23)

`K+1`适用于连续类别编号的解释。作者COCO构建代码按最大类别ID加1配置`num_classes=91`，再加no-object得到92维logits；COCO有80个有效类别，不代表这份实现输出81维。自定义数据需核对标签映射，不能只改一处数字。[作者build中的类别编号说明](https://github.com/facebookresearch/detr/blob/29901c51d7fe8712168b8d0d64351170bc0f83e0/models/detr.py#L271-L294)

## 核心公式

### 为什么要先匹配，再计算训练损失

两只鸟的真值记录没有固定排序，预测也没有预先指定的“第1只鸟槽位”。如果直接把第1条预测永远监督成第1条真值，交换真值列表顺序就可能改变学习目标。DETR先在所有预测与真值之间寻找总代价较小的一对一分配，再根据该分配监督网络。[原论文§3.1，式1–2](https://arxiv.org/pdf/2005.12872v3#page=5)

### 一对候选的匹配代价怎样读

设真值有M个、预测有N个且M≤N；`c_i`是真值类别，`b_i`是真值框，`p_j(c_i)`是第j个预测对该类别的概率。下面用只列出真实目标的等价分配方式说明匹配：

$$
C_{ji}=-p_j(c_i)+\lambda_1\lVert \hat b_j-b_i\rVert_1-\lambda_g\,\mathrm{GIoU}(\hat b_j,b_i)
$$

$$
\hat\sigma=\arg\min_{\sigma\text{为一对一分配}}\sum_{i=1}^{M} C_{\sigma(i),i}
$$

分类概率越高，负概率这一项越小；框越接近，L1越小；GIoU越大，其负值越小。匈牙利算法求的是整组分配的总成本，不是让每个目标各自独立挑最低的一行。作者matcher在实际真值中省略no-object，只将未匹配预测交给后续分类监督。[HungarianMatcher.forward](https://github.com/facebookresearch/detr/blob/29901c51d7fe8712168b8d0d64351170bc0f83e0/models/matcher.py#L31-L72)

这里的匹配分类项是负概率，不是`-log(p)`。正式分类训练用交叉熵，两者不能混写。matcher也在无梯度模式下选择离散分配；训练梯度来自随后依据该分配计算的分类和边框损失。[匹配代价](https://github.com/facebookresearch/detr/blob/29901c51d7fe8712168b8d0d64351170bc0f83e0/models/matcher.py#L57-L68)、[分类损失](https://github.com/facebookresearch/detr/blob/29901c51d7fe8712168b8d0d64351170bc0f83e0/models/detr.py#L99-L112)

### L1与GIoU分别补充什么

L1直接比较归一化的`cx,cy,w,h`四个量的差；IoU比较两框交集面积占并集面积的比例。设A、B是角点坐标定义的两个框，C是包住二者的最小外接矩形，则

$$
\mathrm{GIoU}(A,B)=\frac{|A\cap B|}{|A\cup B|}-\frac{|C|-|A\cup B|}{|C|}
$$

教学数值：A=[0,0,2,2]、B=[1,1,3,3]，两框面积各4，交集1、并集7、外接矩形9。因此IoU=1/7，GIoU=1/7−2/9=−5/63。GIoU可以为负，不是概率。两框没有交集时IoU都为0，而GIoU还能利用外接区域区分部分几何差异。[作者GIoU定义与计算](https://github.com/facebookresearch/detr/blob/29901c51d7fe8712168b8d0d64351170bc0f83e0/util/box_ops.py#L20-L55)

正式边框训练采用加权L1与`1−GIoU`，且只对匹配到真实目标的预测计算；未匹配槽位仍参与no-object分类。`-GIoU`与`1−GIoU`相差常数，在固定匹配对数量时不改变最优分配，但不能据此把记录的损失数值视为相同。[loss_boxes与loss_labels](https://github.com/facebookresearch/detr/blob/29901c51d7fe8712168b8d0d64351170bc0f83e0/models/detr.py#L99-L147)

作者默认配置对no-object分类赋较小权重0.1，以处理大量空槽位；可选辅助损失还会监督中间解码层。本文上面的代价式不包含完整训练的归一化、辅助项和超参数调度，完整定义需结合criterion及配置阅读。[criterion](https://github.com/facebookresearch/detr/blob/29901c51d7fe8712168b8d0d64351170bc0f83e0/models/detr.py#L193-L228)、[训练参数](https://github.com/facebookresearch/detr/blob/29901c51d7fe8712168b8d0d64351170bc0f83e0/main.py#L65-L79)

## 最小代码

### 先沿作者forward找每一步

| 作者代码位置 | 这一操作在做什么 | 容易误读之处 |
|---|---|---|
| `backbone(samples)` | 返回空间特征和位置表示 | 不是整图分类logits |
| `input_proj(src)` | 1×1卷积对齐隐藏维数 | 不把空间位置全局平均 |
| `transformer(..., query_embed.weight, pos)` | 编码图像，解码目标槽位 | query不是文本，也不是输入真值框 |
| `class_embed(hs)` | 每层、每槽位输出类别logits | 这一步还没有softmax |
| `bbox_embed(hs).sigmoid()` | 输出归一化中心与宽高 | 四个值不是四个类别概率 |
| `outputs_class[-1]` | 取最后一层解码结果 | `-1`这里选层，不是选no-object类别 |

表格根据[DETR.forward](https://github.com/facebookresearch/detr/blob/29901c51d7fe8712168b8d0d64351170bc0f83e0/models/detr.py#L54-L66)逐项整理。论文中的FFN有时指预测网络，Transformer层内部也有前馈网络；两者位置与作用不同，不要仅凭缩写相同合并为一个模块。

### 可运行：把预测结果还原为像素框

以下仅用人工设定的3个槽位，复现输出解释中的softmax、坐标转换和缩放。真实类别在本例中为0、1，最后索引2代表no-object；这些假数据不是运行DETR得到的检测结果。

```python
import torch

logits = torch.tensor([[[6., 0., -2.], [0., 5., -2.], [-2., -2., 6.]]])
normalized_boxes = torch.tensor([[[.5, .5, .4, .2],
                                  [.2, .3, .2, .4],
                                  [.5, .5, .1, .1]]])

probabilities = logits.softmax(dim=-1)
scores, labels = probabilities[..., :-1].max(dim=-1)
center = normalized_boxes[..., :2]
size = normalized_boxes[..., 2:]
corners = torch.cat((center - size / 2, center + size / 2), dim=-1)
image_height, image_width = 100, 200
pixel_boxes = corners * torch.tensor([image_width, image_height,
                                     image_width, image_height])

# 仅为本例选择要显示的候选；不是NMS或训练规则。
show = scores[0] > 0.7
assert labels.shape == scores.shape == (1, 3)
assert torch.equal(labels[0, show], torch.tensor([0, 1]))
assert torch.allclose(pixel_boxes[0, 0], torch.tensor([60., 40., 140., 60.]))
assert show.tolist() == [True, True, False]
print(pixel_boxes[0, show])
```

1. `softmax(dim=-1)`沿类别轴归一化，并包含no-object；不能在槽位轴上归一化。
2. `[..., :-1]`只排除最后的no-object列，再在真实类别中取最大分数。作者PostProcess仍输出所有N个候选，**不等于先按包含no-object的argmax删除槽位**。
3. 中心宽高转成`x_min,y_min,x_max,y_max`；x坐标乘图像宽，y坐标乘图像高，不能颠倒。
4. 本例0.7只是显示阈值，不是论文规定的通用阈值。阈值筛选也不等于按框重叠去重的NMS。

这些输出语义对应[作者PostProcess](https://github.com/facebookresearch/detr/blob/29901c51d7fe8712168b8d0d64351170bc0f83e0/models/detr.py#L230-L256)和[框转换函数](https://github.com/facebookresearch/detr/blob/29901c51d7fe8712168b8d0d64351170bc0f83e0/util/box_ops.py#L9-L12)。真实推理还需使用权重对应的图像预处理，并传入希望还原到的有效图像尺寸；不能直接把padding后的批次尺寸当原图尺寸。

## 推导与排错

### 两只目标为什么不能各自选最低成本

下面的表格是人为构造的匹配成本，不是实际图像上的分类概率或距离。

| 预测槽位 | 真值A | 真值B |
|---|---:|---:|
| q0 | 0.10 | 0.20 |
| q1 | 0.11 | 9.00 |
| q2 | 2.00 | 3.00 |

若每个真值独立选最低成本，A、B都会选q0，违反一对一。全局最优是q1→A、q0→B，总成本0.31；q2无匹配，后续应监督为no-object。匈牙利匹配要解决的正是这种整组分配问题。[算法在作者代码中的调用](https://github.com/facebookresearch/detr/blob/29901c51d7fe8712168b8d0d64351170bc0f83e0/models/matcher.py#L67-L72)

以下教学例需要NumPy与SciPy，仅运行小矩阵分配：

```python
import numpy as np
from scipy.optimize import linear_sum_assignment

cost = np.array([[.10, .20], [.11, 9.00], [2.00, 3.00]])
prediction_ids, target_ids = linear_sum_assignment(cost)
pairs = list(zip(prediction_ids.tolist(), target_ids.tolist()))
assert pairs == [(0, 1), (1, 0)]
assert np.isclose(cost[prediction_ids, target_ids].sum(), .31)

target_labels = np.array([0, 1])  # A、B的类别
slot_labels = np.full(3, 2)      # 2代表本例no-object
slot_labels[prediction_ids] = target_labels[target_ids]
assert slot_labels.tolist() == [1, 0, 2]

# 交换真值记录顺序，映回原索引后应得到相同分配。
permutation = np.array([1, 0])
rows, columns = linear_sum_assignment(cost[:, permutation])
assert dict(zip(rows.tolist(), permutation[columns].tolist())) == dict(pairs)
```

交换真值列表顺序并不改变应检测的对象集合；代码检查的是这组数据的分配一致性。真实问题若存在成本完全相同的最优解，算法可能选不同的同价分配，不能要求每次索引完全相同。

### 按症状检查，不靠猜

| 现象或误读 | 先检查什么 |
|---|---|
| 框位置或宽高不对 | 是否把cxcywh当成xyxy；是否把H、W缩放颠倒 |
| 输出总有100条 | 这是槽位数；检查类别分数和显示筛选，不把每条都当真实目标 |
| 自定义两类却仍看到92维 | 是否还在用作者COCO分类头、标签ID或旧权重 |
| 把no-object当成某种背景物体框 | 它表示该槽位没有分配到检测目标，不是整片背景的分割结果 |
| 以为query直接给出初始框 | 核对query embedding、零tgt与交叉注意力，它们是特征表示 |
| 以为有集合训练就绝不重复 | 一对一监督鼓励去重，不是对任意输入零重复的保证 |

## 与相关模型的联系

DETR 编码器/解码器以 CNN 空间序列为输入。DETR的编码器/解码器沿用Transformer架构，但解码端并行预测目标集合。翻译Transformer与它共享注意力、前馈和残差等构件，任务输入、掩码与解码策略却不同，不能直接照搬翻译接口。[原论文§3.2及附录](https://arxiv.org/pdf/2005.12872v3#page=7)

CNN骨干在这里保留空间特征图，而不是使用本库ResNet-18的整图分类logits。作者归档代码允许按骨干名称构建不同网络，这不等于原论文对每一种可构建网络都做过实验。[骨干构建](https://github.com/facebookresearch/detr/blob/29901c51d7fe8712168b8d0d64351170bc0f83e0/models/backbone.py#L83-L95)

原始DETR不采用预设锚框，也不依赖常规NMS来去重；一对一匹配是这一设计的重要组成部分。后续名字中包含DETR的模型可能修改查询、注意力或训练方式，不能把本卡的原始设计自动套给所有变体。[原论文§1、§3](https://arxiv.org/pdf/2005.12872v3)

## 应用流程与使用边界

目标检测输出类别和边界框，保留每个目标的位置。它适合表达“图里有什么、在哪里”，不直接生成自然语言回答，也不把每个像素分成实例。全景分割扩展需要额外的分割分支，不能把基础检测框自动当成分割掩码。[原论文§4.4](https://arxiv.org/pdf/2005.12872v3#page=14)

理解一条实际应用流程时，应分别确认图像预处理、权重与类别映射、模型输出、坐标还原、显示规则；同一个分数阈值不能保证不同数据上的相同效果。若用于设备外观缺陷等自定义场景，还需相应标注与适配，本卡不声称已经验证了该场景。

原论文讨论了相对较长的训练调度和小目标表现方面的局限；这些是固定版本的实验背景，不构成对现今全部DETR变体的排名判断。[原论文§4](https://arxiv.org/pdf/2005.12872v3#page=8)

## 架构图与论文插图

### 按图2从左到右读

本页沿用原论文Fig.2，展示的是网络数据流；匹配训练没有全部画在这张结构图里。[原图及图注，物理第7页](https://arxiv.org/pdf/2005.12872v3#page=7)

| 图中位置 | 应读懂的内容 |
|---|---|
| 左侧CNN与立体特征块 | 图像先变为空间特征；不是先压成整图类别 |
| 编码器前的特征序列与位置分支 | 空间位置被展平成token；位置表示补充原二维位置线索 |
| 中间encoder | 图像位置彼此交换信息，输出图像memory |
| decoder下方彩色小方块 | 不同object query身份；颜色只是示意，不代表人工类别标签 |
| encoder到decoder的箭头 | decoder通过交叉注意力读取图像memory |
| 右侧并排FFN与class/box | 每个槽位经共享预测头产生类别和框；图画了多个位置，不代表参数互不共享 |
| no object与右侧鸟的框 | 槽位可以为空；有效目标才对应图像中的检测记录 |

### 图里没画全的部分

图中的“位置”有两类：图像token的二维位置表示，以及检测槽位的query身份。它们服务于不同对象，不能合并成“预设锚框”。训练时还需用类别、框与真值构造匹配和损失，推理时不需要先拿到真值。[原论文§3.1–3.2](https://arxiv.org/pdf/2005.12872v3#page=5)、[query与memory的实现](https://github.com/facebookresearch/detr/blob/29901c51d7fe8712168b8d0d64351170bc0f83e0/models/transformer.py#L42-L53)

可以按“左侧输入输出 → 中间注意力对象 → 右侧输出语义”的顺序回看图，再对照本卡代码把归一化框还原为像素框。这样能把结构图和具体运算对应起来，而不只记住模块名称。

## 来源与定位

本卡按资料提炼并用自己的语言解释，不复制整篇文章；链接旁标明的是相应论据的位置，教学数值不当作文献结论。

- S32：[原论文2005.12872v3](https://arxiv.org/pdf/2005.12872v3)：§3.1集合匹配、§3.2结构、§4实验与边界、Fig.2及附录A.4。当前原始PDF索引页为5–8，完整PDF可打开；其他页的解释已整理进本卡，不等于全部原页已索引。
- S35：[作者仓库](https://github.com/facebookresearch/detr)：原始方法的参考实现入口；本次代码阅读固定到29901c51d7fe8712168b8d0d64351170bc0f83e0。
- S46：[模型、损失与后处理](https://github.com/facebookresearch/detr/blob/29901c51d7fe8712168b8d0d64351170bc0f83e0/models/detr.py)：DETR.forward、SetCriterion、PostProcess及build。
- S47：[Transformer实现](https://github.com/facebookresearch/detr/blob/29901c51d7fe8712168b8d0d64351170bc0f83e0/models/transformer.py)：展平、query/tgt、编码器与解码器注意力。
- S48：[匹配实现](https://github.com/facebookresearch/detr/blob/29901c51d7fe8712168b8d0d64351170bc0f83e0/models/matcher.py)：负概率、L1、负GIoU和linear_sum_assignment。
- S49：[框运算](https://github.com/facebookresearch/detr/blob/29901c51d7fe8712168b8d0d64351170bc0f83e0/util/box_ops.py)：cxcywh转换、IoU与GIoU。
- S52：[作者说明文章](https://ai.meta.com/blog/end-to-end-object-detection-with-transformers/)：用“图像到集合”解释方法动机。
- S53：[位置编码](https://github.com/facebookresearch/detr/blob/29901c51d7fe8712168b8d0d64351170bc0f83e0/models/position_encoding.py)、S54：[骨干实现](https://github.com/facebookresearch/detr/blob/29901c51d7fe8712168b8d0d64351170bc0f83e0/models/backbone.py)：分别核对空间位置、padding和特征读出。
- A01：本卡自编布局、输出还原和匹配例子，仅验证这些局部运算；不是完整检测模型的推理或训练。

## 可继续追问

“query身份和零tgt为什么分开？”“self-attention和cross-attention各自读什么？”“N=100与K+1有什么区别？”“COCO为何输出92维？”“为什么不能每个真值独立挑一个最低成本？”“PostProcess是不是直接按no-object的argmax过滤？”“图2的彩色方块是类别还是槽位？”
