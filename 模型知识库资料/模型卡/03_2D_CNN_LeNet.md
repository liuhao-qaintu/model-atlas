# 2D CNN / LeNet：从二维输入到类别分数

模型 ID：cnn-2d。类型：架构家族与具体教学实例。整理日期：2026-10-13。

2D CNN 不是一个固定模型。本卡用参考 D2L LeNet 章节、重新编写的 LeNetTutorial 解释数据流；它采用教材的简化配置，不声称完全复现原始 LeNet-5。算子文档固定 PyTorch 2.6，与本地验证环境对应。

## 概览

Conv2d 在高度、宽度两个空间轴上提取局部特征，输入通道是另一维。灰度图像可以只有一个通道，RGB 图像通常有三个通道；通道数量本身不能证明存在多种模态。[S03：Conv2d](https://docs.pytorch.org/docs/2.6/generated/torch.nn.Conv2d.html#shape)

D2L 的 LeNet 教学结构使用卷积、Sigmoid、平均池化和全连接层。本卡使用显式输入输出维数，便于逐层检查。具体历史变体应查原论文，不能把这段教学实现等同于全部 LeNet 配置。[S05：教材结构与实现](https://d2l.ai/chapter_convolutional-neural-networks/lenet.html)

## 输入输出与数据流

E03 教学实例输入是浮点张量 `[B,C,H,W]`，本例为 `[2,1,28,28]`；输出 `[2,10]` 是十个类别的 logits，没有 softmax。随机初始化、随机输入只验证程序与形状，不具备实际识别能力。

| 步骤 | 本例输出形状 |
|---|---|
| Conv2d，1→6，k=5，p=2 | `[2,6,28,28]` |
| Sigmoid、AvgPool2d，k=2，s=2 | `[2,6,14,14]` |
| Conv2d，6→16，k=5，p=0 | `[2,16,10,10]` |
| Sigmoid、AvgPool2d，k=2，s=2 | `[2,16,5,5]` |
| 展平 | `[2,400]` |
| Linear，400→120，Sigmoid | `[2,120]` |
| Linear，120→84，Sigmoid | `[2,84]` |
| Linear 分类头，84→10 | `[2,10]` |

400 来自 `16×5×5`。本例分类前的表示是 84 维；卷积特征展平后是 400 维，两者位置不同。当前代码固定 28×28 的空间尺寸；换成 RGB 时须修改输入通道，换分辨率时须同时检查卷积形状与全连接层维数。不能把整个模型的输出形状直接套用 Conv2d 公式。[E03：可运行代码](/models/cnn-2d?section=code)

## 核心公式

Conv2d 的高度、宽度分别按同样形式计算。[S03：Shape](https://docs.pytorch.org/docs/2.6/generated/torch.nn.Conv2d.html#shape)

$$
H_{out}=\left\lfloor\frac{H_{in}+2p_h-d_h(k_h-1)-1}{s_h}+1\right\rfloor
$$

宽度将 H 与下标 h 替换为 W 与 w。p 是 padding，d 是 dilation，k 是 kernel size，s 是 stride。例如本例第二个卷积 `14→10`，因为 k=5、p=0、d=1、s=1；输出通道 16 由 out_channels 指定。

## 最小代码

E03 是独立编写的教学代码，结构参考 S05。依赖已有 PyTorch，不安装 D2L，不下载数据或权重，也不训练。features 返回展平的卷积特征，representation 返回分类头之前的表示。

```python
import torch
from torch import nn


class LeNetTutorial(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 6, kernel_size=5, padding=2),
            nn.Sigmoid(),
            nn.AvgPool2d(2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.Sigmoid(),
            nn.AvgPool2d(2, stride=2),
            nn.Flatten(start_dim=1),
        )
        self.representation = nn.Sequential(
            nn.Linear(400, 120), nn.Sigmoid(),
            nn.Linear(120, 84), nn.Sigmoid(),
        )
        self.classifier = nn.Linear(84, num_classes)

    def forward(self, x):
        return self.classifier(self.representation(self.features(x)))


torch.manual_seed(7)
model = LeNetTutorial().eval()
x = torch.randn(2, 1, 28, 28)
with torch.no_grad():
    flat_features = model.features(x)
    representation = model.representation(flat_features)
    logits = model(x)
assert flat_features.shape == (2, 400)
assert representation.shape == (2, 84)
assert logits.shape == (2, 10)
print("features:", tuple(flat_features.shape))
print("representation:", tuple(representation.shape))
print("logits:", tuple(logits.shape))
```

## 与相关模型的联系

本卡关系是 2D CNN → LeNetTutorial → 二维卷积、平均池化、Linear 分类头。只有本卡实例必须按这段代码运行，不能将其全部模块推广到所有 2D CNN。[E03：实现](/models/cnn-2d?section=code)

LeNetTutorial 与 ResNet-18 的 BasicBlock 都包含二维卷积。LeNetTutorial 没有 BasicBlock，也没有残差相加；共享 Conv2d 不代表结构或参数相同。输入通道、卷积核、归一化与连接方式须分别检查。

LeNetTutorial 的最后一层是 Linear(84, num_classes)，本例类别数为 10。它与 TinySignalCNN、ResNet-18 都可以使用 Linear 分类头，但头的输入维数不同。[E03：分类头](/models/cnn-2d?section=code)

## 图示与资料状态

当前通过逐层形状表解释结构，没有下载或插入图片。可在 S05 原网页查看教材图和对应说明。未整理声称某个 SOTA 模型或故障诊断论文采用本卡实例的应用关系。

## 来源与定位

- S03：[PyTorch 2.6 Conv2d](https://docs.pytorch.org/docs/2.6/generated/torch.nn.Conv2d.html)，输入输出轴、参数与 Shape。
- S05：[D2L 1.0.3 LeNet 章节](https://d2l.ai/chapter_convolutional-neural-networks/lenet.html)，教材结构及简化实现边界。
- E03：[LeNetTutorial 教学示例](/models/cnn-2d?section=code)，本卡代码及实际前向形状。未使用模型性能数值。

## 可继续追问

“400 和 84 分别是什么？”“RGB 输入改哪里？”“为什么换成 32×32 会遇到维数问题？”“和 ResNet-18 共享了什么算子？”“AvgPool2d(2) 与全局平均聚合有什么区别？”
