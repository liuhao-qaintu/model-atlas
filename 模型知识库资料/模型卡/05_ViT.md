# ViT：从图像分块到分类 token

模型 ID：vit。整理日期：2026-10-13。

本卡解释经典 ViT 与固定版本 Torchvision 实现。E05 是小尺寸教学配置，随机初始化，只验证输入输出；不能用于识别真实图像。

## 概览与核心机制

ViT 将图像分块、映射为 token，再由 Transformer 编码器处理。原论文方法见 §3.1；它用于图像分类，而不是输入一段目标文本的 encoder-decoder。[S16：方法](https://arxiv.org/pdf/2010.11929#page=3)

Torchvision v0.21.0 的 VisionTransformer 使用 conv_proj 完成 patch 投影，加入 class_token 与位置表示，最后读取首 token 的特征并经过分类头。它的 EncoderBlock 包含多头自注意力、MLP 与残差相加；这里的具体类不是 nn.TransformerEncoderLayer。[S25：VisionTransformer / EncoderBlock](https://github.com/pytorch/vision/blob/v0.21.0/torchvision/models/vision_transformer.py)

## 输入输出与数据流

E05 使用 32×32 RGB 浮点输入、8×8 patch、隐藏维数 64、2 层、4 个注意力头、MLP 中间维数 128、10 类。以下形状是本例配置，不是 ViT 家族的统一规格。

| 位置 | 本例形状 / 含义 |
|---|---|
| 图像输入 | `[2,3,32,32]`；B、C、H、W |
| patch 投影 | `[2,64,4,4]` |
| 16 个 patch token | `[2,16,64]` |
| 加入 class token | `[2,17,64]` |
| 编码器输出 | `[2,17,64]` |
| 首 token 表示 | `[2,64]`，用于分类 |
| 分类头输出 | `[2,10]`，logits |

16 是 patch 数量，17 是包含 class token 的序列长度，64 是特征维数，10 是类别数。分类头直接返回未归一化 logits，没有应用 softmax，不能直接当成概率或识别置信度；本例分类头零初始化，分数也没有识别意义。更换图像尺寸不能只改输入张量：还要检查 patch 数量和位置表示。E05 显式要求 `[B,3,32,32]`。[E05：代码](/models/vit?section=code)

S25 的 vit_b_16 配置有 12 层、12 头、隐藏维数 768；其无权重默认配置使用 224×224 输入、16×16 patch 和 1000 类，包含 197 个 token。E05 的 17 个 token 不能当成该模型规格。[S25：vit_b_16 / VisionTransformer](https://github.com/pytorch/vision/blob/v0.21.0/torchvision/models/vision_transformer.py)

## 核心公式

输入尺寸可被 patch 大小 P 整除时，patch 数量与首层 token 数为：

$$
N=\frac{H}{P}\frac{W}{P},\qquad L=N+1
$$

每个展平 patch 的原始维数为 P²C，再线性投影到 D。Torchvision 的普通 conv_proj 使用 kernel_size=stride=P 的 Conv2d，实现非重叠分块及投影。它使用卷积算子不意味着整体架构变成了 LeNet。[S16：§3.1](https://arxiv.org/pdf/2010.11929#page=3)、[S25：conv_proj](https://github.com/pytorch/vision/blob/v0.21.0/torchvision/models/vision_transformer.py)

## 最小代码

E05 使用已有 Torchvision 0.21.0 与 PyTorch 2.6.0，显式配置 VisionTransformer，并展示中间形状。没有下载权重；Torchvision 此配置的分类头初始化为零，因此本例 logits 为零也符合实现，不能解释为识别成功。inspect_features 在本例 eval 模式中核查数据流。

```python
import torch
from torch import nn
from torchvision.models.vision_transformer import VisionTransformer


class TinyPatchViT(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = VisionTransformer(
            image_size=32, patch_size=8, num_layers=2,
            num_heads=4, hidden_dim=64, mlp_dim=128,
            num_classes=10, dropout=0.0, attention_dropout=0.0,
        )

    def check_input(self, images):
        if images.ndim != 4 or tuple(images.shape[1:]) != (3, 32, 32):
            raise ValueError("本教学配置要求 [B,3,32,32]")

    def inspect_features(self, images):
        self.check_input(images)
        patches = self.backbone.conv_proj(images).flatten(2).transpose(1, 2)
        cls = self.backbone.class_token.expand(images.size(0), -1, -1)
        tokens = torch.cat([cls, patches], dim=1)
        encoded = self.backbone.encoder(tokens)
        return patches, encoded, encoded[:, 0]

    def forward(self, images):
        self.check_input(images)
        return self.backbone(images)


torch.manual_seed(7)
model = TinyPatchViT().eval()
images = torch.randn(2, 3, 32, 32)
with torch.no_grad():
    patches, encoded, cls_features = model.inspect_features(images)
    logits = model(images)
assert patches.shape == (2, 16, 64)
assert encoded.shape == (2, 17, 64)
assert cls_features.shape == (2, 64)
assert logits.shape == (2, 10)
print("patches:", tuple(patches.shape))
print("encoded:", tuple(encoded.shape))
print("class features:", tuple(cls_features.shape))
print("logits:", tuple(logits.shape))
```

## 与相关模型的联系

ViT → Torchvision EncoderBlock → 多头注意力、逐位置前馈网络、残差相加。该固定实现使用 LayerNorm 后进入注意力或 MLP 分支，并在分支输出后相加；与本库 norm_first=False 的 Transformer 教学配置顺序不同。[S25：EncoderBlock.forward](https://github.com/pytorch/vision/blob/v0.21.0/torchvision/models/vision_transformer.py)

TinyPatchViT 的 patch 投影使用 Conv2d(3,64,kernel_size=8,stride=8)，分类头为 Linear(64,10)。前者生成 token，后者读取 class token 后分类，分别可以与 LeNet / BasicBlock 的卷积、其他分类头比较。[S25：VisionTransformer.__init__](https://github.com/pytorch/vision/blob/v0.21.0/torchvision/models/vision_transformer.py)、[E05：教学配置](/models/vit?section=code)

CLIP 原论文采用过 ViT 视觉编码器。CLIP 的 ViT 变体把视觉表示投影到图文共享空间，不等于直接使用 Torchvision 的 1000 类分类头。具体实现和输入输出请继续阅读 CLIP 卡。[S18：§2.4–2.5](https://arxiv.org/pdf/2103.00020#page=5)、[CLIP 输入输出](/models/clip?section=io)

## 图示与资料状态

可参考原论文第 3 页 Figure 1。本轮只提供原始位置链接，不复制或插入图像。教学代码的中间张量表用于解释数据流。[原论文图示](https://arxiv.org/pdf/2010.11929#page=3)

## 来源与定位

- S16：[ViT 原论文](https://arxiv.org/pdf/2010.11929)，§3.1 与 Figure 1；核对 patch、class token 与编码器。
- S25：[Torchvision v0.21.0 源码](https://github.com/pytorch/vision/blob/v0.21.0/torchvision/models/vision_transformer.py)，VisionTransformer、EncoderBlock、vit_b_16；与 验证环境 对应。
- S18：[CLIP 原论文](https://arxiv.org/pdf/2103.00020)，§2.4–2.5；仅支持 CLIP 使用 ViT 的关系，不推断故障诊断应用。
- E05：[TinyPatchViT 教学示例](/models/vit?section=code)，已在 验证环境 验证前向形状与尺寸边界；不训练、不评测预测准确率。

## 可继续追问

“为什么是 17 而不是 16 个 token？”“64 和 10 分别是什么？”“patch 投影为什么可以用卷积？”“ViT 是否需要解码器？”“与 Transformer 的残差顺序有什么区别？”“CLIP 是否直接使用 ViT 的分类头？”
