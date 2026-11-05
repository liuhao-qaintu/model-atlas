# 1D CNN：一维卷积网络

模型 ID：cnn-1d。类型：架构家族。整理日期：2026-10-12。

本卡使用一个自行编写的最小分类器解释数据流，不把它视为某篇论文的模型。关键算子依据 S01/S02，代码是教学示例。

## 概览

本卡的一维卷积沿序列位置提取局部特征。输入通道表示每个样本并行提供的信号维度；这些通道是否来自不同模态，取决于实际数据定义，不能只凭通道数判断。

任务或具体实现变化时，末端可以返回特征或任务输出。因此，“1D CNN 输出什么”需要结合分类头、输入数据及实现回答。

## 结构与原理

### 一个卷积核到底计算了什么

先看一个能手算的单通道例子：输入 `[1,2,4,8,16]`，卷积核 `[1,0,-1]`，不加偏置，kernel_size=3、stride=1、padding=0、dilation=1。第一个窗口得到 `1×1+2×0+4×(-1)=-3`；后两个窗口得到 -6、-12，输出长度为3。这里的负数是局部模式响应，不是负的“概率”。PyTorch Conv1d 实际使用互相关，不先翻转卷积核；不能拿翻转后的数学卷积结果对照这段代码。[Conv1d运算约定](https://docs.pytorch.org/docs/2.8/generated/torch.nn.Conv1d.html)

```python
import torch
from torch import nn

local_filter = nn.Conv1d(1, 1, kernel_size=3, bias=False)
with torch.no_grad():
    local_filter.weight.copy_(torch.tensor([[[1., 0., -1.]]]))
    local_response = local_filter(torch.tensor([[[1., 2., 4., 8., 16.]]]))
assert torch.allclose(local_response, torch.tensor([[[-3., -6., -12.]]]))
```

### 为什么有多个输入通道和输出通道

`groups=1` 时，每个输出通道都有一组覆盖全部输入通道的核，对同一位置的各通道局部窗口加权求和，再加该输出通道的偏置。多个输出通道就是多组可学习滤波器，并非简单复制输入。它们是否学到边缘、频率变化等模式取决于数据和训练，不能给随机滤波器指定现实物理含义。

权重形状是 `[C_out,C_in/groups,k]`。含偏置的参数量可从该形状推导为 `C_out×(C_in/groups)×k+C_out`；不含偏置则去掉最后一项。示例 `C_in=4,C_out=6,k=3,groups=2,bias=True`：权重36个、偏置6个，共42个。两个分组分别连接2个输入和3个输出通道；同一层不会把两组通道混合。输入、输出通道数都必须能被groups整除。这不是把序列长度分成两半。[权重与groups定义](https://docs.pytorch.org/docs/2.8/generated/torch.nn.Conv1d.html)

### 局部特征如何成为分类分数

本卡 TinySignalCNN 的卷积和激活先保留序列位置；MaxPool缩短长度；AdaptiveAvgPool1d(1)把每个通道的时间轴聚合成一个值，得到32维特征；最后Linear(32,4)才映射到本例定义的4个任务分数。卷积输出通道数32不是类别数4。全局平均会压缩显式位置信息，所以如果任务需要输出事件发生在哪个位置，不能直接照搬这个全局分类头。[本卡教学代码](/models/cnn-1d?section=code)

本例是可核对运算的教学实例，不是训练完成的故障诊断模型。特征、类别分数与概率的区别取决于任务含义和实际计算；特征即使偶然落在[0,1]或和为1，也不会因此拥有类别概率的含义。

## 输入输出

本例输入为浮点张量 `[B, C, L]`：批量 B、通道 C、长度 L。本例固定 C=1，使用 L=1024；其他长度在满足池化要求时也可前向运行。

本例输出 `[B, K]` 类别分数，K=4。它是 logits，没有经过 softmax，不能直接称为概率。4 个类别没有绑定到真实故障类别；随机初始化网络也没有实际诊断能力。

| 步骤 | 示例形状 |
|---|---|
| 输入 | `[2, 1, 1024]` |
| Conv1d，1→16，k=5，p=2，s=1 | `[2, 16, 1024]` |
| ReLU | `[2, 16, 1024]` |
| MaxPool1d，k=2 | `[2, 16, 512]` |
| Conv1d，16→32，k=3，p=1，s=1 | `[2, 32, 512]` |
| ReLU、AdaptiveAvgPool1d(1) | `[2, 32, 1]` |
| Flatten | `[2, 32]` |
| Linear，32→4 | `[2, 4]` |

## 长度公式

依据 [S01：Conv1d 的 Shape 说明](https://docs.pytorch.org/docs/2.8/generated/torch.nn.Conv1d.html)：

$$
L_{out}=\left\lfloor\frac{L_{in}+2p-d(k-1)-1}{s}+1\right\rfloor
$$

其中 p 是 padding，d 是 dilation，k 是 kernel size，s 是 stride。这是卷积层长度公式，不能直接用来解释整个分类器的输出维度。

## 最小代码

原创教学示例，使用已有 PyTorch；不读取实验数据，不下载权重，不训练。

```python
import torch
from torch import nn


class TinySignalCNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(start_dim=1),
        )
        self.classifier = nn.Linear(32, num_classes)

    def forward(self, x):
        return self.classifier(self.features(x))


torch.manual_seed(7)
model = TinySignalCNN().eval()
x = torch.randn(2, 1, 1024)
with torch.no_grad():
    features = model.features(x)
    logits = model(x)
assert features.shape == (2, 32)
assert logits.shape == (2, 4)
print("features:", tuple(features.shape))
print("logits:", tuple(logits.shape))
```

这里的 eval 与 no_grad 仅服务于前向示例：前者切换模块模式，后者避免记录梯度；不意味着随机权重变成已训练权重。

## 推导与排错

### 改变输入长度，沿每一步重新计算

仍使用本卡TinySignalCNN，输入改为 `[2,1,513]`。第一层k=5、p=2、s=1、d=1，长度 `floor((513+4-4-1)/1+1)=513`。MaxPool1d(2)默认stride=2、不加padding、ceil_mode=False，得到256；第二层k=3、p=1保长，仍是256。自适应平均池化后变为1，展平为 `[2,32]`，Linear得到 `[2,4]`。不要把513硬套进原1024长度示例，也不要把卷积长度公式当成类别数公式。

### 感受野为什么不等于卷积核大小

对本例这样的串联层，令R为一个位置覆盖的输入跨度，J为相邻位置在原输入上的间隔，初值均为1。使用 `R_new=R_old+(k-1)×d×J_old`、`J_new=J_old×s` 推导：

| 位置 | R | J | 含义 |
|---|---|---|---|
| 第一层卷积后 | 5 | 1 | 单点看5个原输入位置的跨度 |
| 池化后 | 6 | 2 | 池化合并相邻两个卷积响应 |
| 第二层卷积后 | 10 | 2 | 单点覆盖输入跨度10，而非3 |

这是全局池化之前的理论跨度推导，边界可能涉及补零；经过全局平均后会汇总所有保留的位置，不能把最终32维表示的感受野仍叫10。也不能用这个理论跨度保证模型实际上学会了长程依赖。

### 三个常见错误怎么核对

1. **通道数报错**：先打印输入shape，确认是 `[B,C,L]`。数据若为 `[B,L,C]`，应按轴含义使用permute(0,2,1)，不能只改shape数字。多通道情况下，reshape不会替你交换“时间”和“传感器”语义。
2. **改长度后Linear维度错**：检查是否先展平整个 `[B,C,L]`，从而把长度写进了Linear输入。本卡先自适应池化，所以合法的可变长度不会改变32维头输入；这不等于能接收任意短的序列，本例长度1无法通过MaxPool1d(2)。
3. **把32通道当32类**：确认读取的是features还是classifier输出；本例前者 `[B,32]`，后者 `[B,4]`。更换Linear输出维数仅改变接口，不代表新任务已训练好。

先运行“代码实现”中的TinySignalCNN定义，再运行下段。代码只核对形状、参数量和轴含义，不训练、不下载权重。

```python
odd_model = TinySignalCNN().eval()
with torch.no_grad():
    odd_features = odd_model.features(torch.randn(2, 1, 513))
    odd_logits = odd_model(torch.randn(2, 1, 513))
assert odd_features.shape == (2, 32)
assert odd_logits.shape == (2, 4)
assert sum(p.numel() for p in odd_model.parameters()) == 1796
grouped = nn.Conv1d(4, 6, kernel_size=3, groups=2, bias=True)
assert tuple(grouped.weight.shape) == (6, 2, 3)
assert sum(p.numel() for p in grouped.parameters()) == 42
time_major_data = torch.arange(24).reshape(2, 4, 3)
assert not torch.equal(time_major_data.permute(0, 2, 1), time_major_data.reshape(2, 3, 4))
```

这些数值由本卡配置推导，可在验证环境运行上述代码核对。更改kernel、stride、dilation、分组或池化后应重新计算，不推广到所有CNN。

## 与相关模型的联系

- 与 2D CNN 都可以包含卷积、激活、池化与分类头，但空间轴、参数和输入约定不同。
- 本教学实例包含 Conv1d、MaxPool1d、AdaptiveAvgPool1d、Linear。
- 是否用于某项故障诊断研究，需要在具体论文应用记录中给出来源，不凭架构名称推定。

这些是算子/实现级联系。后续图谱应把教学实例与架构家族分开，避免把示例的所有模块误认为每个 1D CNN 都必须使用。

## 图示与资料状态

当前以形状表解释结构，尚未绑定图片。不是从某张论文图推导出的实现。论文应用暂缺，不生成虚构论文或引用。

## 来源与定位

- S01：[PyTorch 2.8 Conv1d](https://docs.pytorch.org/docs/2.8/generated/torch.nn.Conv1d.html)，参数与 Shape。
- S02：[PyTorch 2.8 AdaptiveAvgPool1d](https://docs.pytorch.org/docs/2.8/generated/torch.nn.AdaptiveAvgPool1d.html)，输出大小与 Shape。
- E01：本卡原创示例 `TinySignalCNN`，已在 验证环境 的 PyTorch 2.6.0+cpu 中通过前向及形状核查；另检查长度 128 和 513 的输入。代码示例可在相应PyTorch环境中核对。

## 可继续追问

“C 与 L 分别是什么？”“多通道怎样修改输入？”“为什么输出不是概率？”“如何只取特征？”“与 2D CNN 有什么共同模块？”
