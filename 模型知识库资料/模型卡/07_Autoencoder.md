# Autoencoder：潜在表示与重构输出

模型 ID：autoencoder。整理日期：2026-10-13。

本卡介绍基础确定性自编码器，E07 为自行编写的全连接教学例。随机初始化，只核查接口，不训练模型或验证异常检测能力。

## 概览与核心机制

Autoencoder 由编码器与解码器组成：先把输入映射为潜在表示，再从表示重构输入。训练目标可来自输入本身，不需要把类别标签当成重构目标。教材第 14 章讨论了欠完备表示和正则化；模型是否学到有用表示取决于目标、约束与数据，不能仅凭存在一个瓶颈就保证效果。[S10：第 14 章与 §14.1](https://www.deeplearningbook.org/contents/autoencoders.html)

本例 TinyDenseAutoencoder 使用 DenseEncoder 与 DenseDecoder。编码器为 Linear(32,16)、ReLU、Linear(16,4)；解码器为 Linear(4,16)、ReLU、Linear(16,32)。这是欠完备的教学配置，不是所有 Autoencoder 的统一结构。[E07：代码](/models/autoencoder?section=code)

DenseEncoder / DenseDecoder 是本卡对 encoder / decoder 顺序模块的阅读名称，不是需要导入的 PyTorch 类名。

## 结构与原理

### 重构输入，为什么还要学习表示？

编码器把输入 $x$ 映射成表示 $z$，解码器再产生重构 $\hat{x}$。重构是训练信号；希望得到的是能保留数据结构的表示，并不是让任意输入都原样通过就算成功。容量过大时模型可能学会复制而没有有用的表示，因此需要结合瓶颈、正则化或具体训练准则理解，不能把“重构误差小”直接当成分类或诊断能力。潜在维数小于输入是欠完备的一种配置；正则化自编码器不一律要求这种维数关系。[S10：§14.1–14.2](https://www.deeplearningbook.org/contents/autoencoders.html)

### 去噪 AE 改变了什么？

基础 AE 将 $x$ 送入编码器，重构目标也是 $x$。去噪 AE 先用破坏过程得到 $\tilde{x}$，送入编码器的是 $\tilde{x}$，目标仍为原始 $x$：

$$
\hat{x}_{\mathrm{AE}}=g_\phi(f_\theta(x)),\qquad
\hat{x}_{\mathrm{DAE}}=g_\phi(f_\theta(\tilde{x}))
$$

这里统一用 $z$ 表示潜在表示、$\hat{x}$ 表示重构。Vincent 等原论文用 $y$ 表示隐藏表示、$z$ 表示重构，不能把论文里的 $z$ 误当成本卡的潜在向量。该论文实验随机选取一定比例的输入分量置零；它是文中的破坏设置，不是所有去噪 AE 必须采用的唯一噪声。[S55：§2.2–2.3，PDF 第 2–3 页](https://www.cs.toronto.edu/~larocheh/publications/icml-2008-denoising-autoencoders.pdf#page=2)

### 损失和结构要带着条件读

本卡 E07 使用实数特征、无末层 Sigmoid、mean MSE；论文 §2.2 的基础模型使用 Sigmoid 映射，并讨论平方误差和重构交叉熵。二者共享编码—解码与重构思想，但数据解释、输出约束和损失配置要分别核对。去噪训练的核心差别是“受破坏输入→原始目标”，不是把整个家族规定为某一组维数或某一种损失。[E07：本卡代码](/models/autoencoder?section=code)、[S55：§2.2–2.3](https://www.cs.toronto.edu/~larocheh/publications/icml-2008-denoising-autoencoders.pdf#page=2)

## 输入输出与数据流

| 位置 | 本例形状 / 含义 |
|---|---|
| 浮点输入 | `[2,32]`，每个样本 32 个数值特征 |
| 编码器中间层 | `[2,16]` |
| 潜在表示 z | `[2,4]`，4 为表示维数，不是类别数 |
| 解码器中间层 | `[2,16]` |
| 重构值 | `[2,32]`，与输入维数对齐 |
| mean MSE | 标量，平均所有样本和特征的平方误差 |
| 每样本重构误差 | `[2]`，沿 32 个特征取平均 |

本例最后一层不使用 Sigmoid / softmax，重构值可为任意实数，不是概率。输入也不是必须位于 [0,1] 的图像像素；若改数据类型、归一化或输出约束，需要相应选择损失。这里计算误差仅展示接口，不做梯度更新。[E07：数据与前向](/models/autoencoder?section=code)、[S30：MSELoss](https://github.com/pytorch/pytorch/blob/v2.6.0/torch/nn/modules/loss.py)

## 核心公式

编码和重构分别写作：

$$
z=f_\theta(x),\qquad \hat{x}=g_\phi(z)
$$

本例采用 mean MSE，B 为样本数、F=32 为输入特征数：

$$
\mathcal{L}=\frac{1}{BF}\sum_{i=1}^{B}\sum_{j=1}^{F}(\hat{x}_{ij}-x_{ij})^2
$$

这是一个可用的重构目标，不是所有自编码器的唯一损失；本例没有分类头、VAE 的分布参数或 KL 项，也没有去噪输入。[S10：§14.1–14.2](https://www.deeplearningbook.org/contents/autoencoders.html)、[S30：MSELoss 的 mean](https://github.com/pytorch/pytorch/blob/v2.6.0/torch/nn/modules/loss.py)

## 最小代码

E07 使用已有 PyTorch，输入为随机数值特征。只执行前向与误差计算，不读取任务数据，不优化权重。

```python
import torch
from torch import nn
from torch.nn import functional as F


class TinyDenseAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 4),
        )
        self.decoder = nn.Sequential(
            nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, 32),
        )

    def forward(self, values):
        if values.ndim != 2 or values.size(1) != 32:
            raise ValueError("本教学配置要求 [B,32] 浮点特征")
        latent = self.encoder(values)
        reconstruction = self.decoder(latent)
        return latent, reconstruction


torch.manual_seed(7)
model = TinyDenseAutoencoder().eval()
values = torch.randn(2, 32)
with torch.no_grad():
    latent, reconstruction = model(values)
    loss = F.mse_loss(reconstruction, values)
    per_sample_error = (reconstruction - values).square().mean(dim=1)
assert latent.shape == (2, 4)
assert reconstruction.shape == values.shape
assert loss.ndim == 0
assert per_sample_error.shape == (2,)
assert torch.allclose(loss, per_sample_error.mean())
print("latent:", tuple(latent.shape))
print("reconstruction:", tuple(reconstruction.shape))
print("mean MSE shape:", tuple(loss.shape))
```

## 推导与排错

| 容易混淆的问题 | 先核对什么 |
|---|---|
| 4 维潜在表示是不是 4 个类别？ | E07 的 4 是编码维数，没有分类头；重构仍为 32 维。 |
| AE 是否只能用 MSE？ | MSE 是 E07 的选择；论文 §2.2 还讨论重构交叉熵及其数据解释。 |
| 去噪目标是否也是加噪后的值？ | 论文 §2.3 的输入是受破坏版本，目标是原始输入；不能把二者交换。 |
| 去噪表示是否一定更适合我的诊断任务？ | 先看任务、数据和实验，不能从重构训练直接推出业务效果。 |
| 原论文的 z 为什么与本卡含义不同？ | 逐个来源读符号：原论文 z 是重构，本卡 z 是潜在表示。 |

这些是阅读核对项，不是额外训练或性能验证。[E07：代码](/models/autoencoder?section=code)、[S55：§2.2–2.4](https://www.cs.toronto.edu/~larocheh/publications/icml-2008-denoising-autoencoders.pdf#page=2)

## 与相关模型的联系

Autoencoder → DenseEncoder → 潜在瓶颈；Autoencoder → DenseDecoder → 输入重构。这里的编码器和解码器是全连接模块，没有注意力、交叉注意力或逐 token 词表输出，不能因名称相同就等同于 Transformer 编码器和解码器。[E07：结构](/models/autoencoder?section=code)

DenseEncoder 使用线性映射压缩到 4 维；DenseDecoder 使用线性映射恢复到 32 维。分类模型也可以使用 Linear，但输出用途不同：本例重构输入，TinyBiLSTM 的 Linear 分类头输出类别分数。共享算子不代表任务、参数或已训练权重相同。[E07：代码](/models/autoencoder?section=code)、[S29：Linear](https://github.com/pytorch/pytorch/blob/v2.6.0/torch/nn/modules/linear.py)、[LSTM 教学分类头](/models/lstm?section=code)

重构误差可作为后续分析的一个数值，但本例没有提供正常/异常分布或业务阈值，因此不直接输出故障类别、异常概率或可用的告警结果。VAE、去噪与稀疏自编码器有不同约束，本卡只介绍基础实例。[S10：第 14 章](https://www.deeplearningbook.org/contents/autoencoders.html)

## 图示与资料状态

教材 Figure 14.1 给出编码与重构关系。本轮只链接资料位置，不复制或插入图片；本例结构也可按 32 → 16 → 4 → 16 → 32 的特征维数阅读。[教材原文](https://www.deeplearningbook.org/contents/autoencoders.html)

## 来源与定位

- S10：[Deep Learning 第 14 章](https://www.deeplearningbook.org/contents/autoencoders.html)，概览、§14.1–14.2；支持编码/解码、重构目标及约束范围。
- S29：[PyTorch v2.6.0 linear.py](https://github.com/pytorch/pytorch/blob/v2.6.0/torch/nn/modules/linear.py)，Linear；输入输出维数与仿射映射。
- S30：[PyTorch v2.6.0 loss.py](https://github.com/pytorch/pytorch/blob/v2.6.0/torch/nn/modules/loss.py)，MSELoss；mean 归约。
- E07：[TinyDenseAutoencoder 教学示例](/models/autoencoder?section=code)，在 验证环境 核查潜在/重构形状、误差归约和尺寸边界。

- S55：[Vincent 等，ICML 2008 原论文](https://www.cs.toronto.edu/~larocheh/publications/icml-2008-denoising-autoencoders.pdf)，§2.2–2.4，基础 AE、去噪输入与重构目标、逐层训练。

## 可继续追问

“4 是类别数吗？”“重构值为什么不是概率？”“mean MSE 是对哪些元素平均？”“编码器与 Transformer 的编码器一样吗？”“为什么算出重构误差也不能直接判定故障？”
