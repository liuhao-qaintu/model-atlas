# CLIP：图文双编码器与匹配分数

模型 ID：clip。整理日期：2026-10-13。

本卡解释经典 CLIP。官方实现引用固定 commit a1d0717；E06 是简化的双编码器教学例，只展示接口、归一化和匹配矩阵，不是预训练 CLIP。

## 概览与核心机制

CLIP 分别编码图像与文本，将二者投影到共享表示空间，并利用成对数据进行对比学习。它返回图文匹配分数；候选文本可以表示类别，也可以表示描述。[S18：§2.3 / Figure 3](https://arxiv.org/pdf/2103.00020#page=4)

官方 CLIP 可选 ViT 视觉编码器；文本侧使用带因果遮罩的 Transformer。另一组视觉变体采用 ModifiedResNet，不能把它说成直接采用本库的 ResNet-18。这里记录模型组成和用途，不比较模型准确率。[S26：CLIP.__init__ / build_attention_mask](https://github.com/openai/CLIP/blob/a1d071733d7111c9c014f024669f959182114e33/clip/model.py)、[S18：§2.4–2.5](https://arxiv.org/pdf/2103.00020#page=5)

## 输入输出与数据流

| 位置 | 官方接口的约定 |
|---|---|
| 图像 | `[B,3,H,W]`，通过所选模型对应的 preprocess；分辨率随配置确定 |
| 文本 | tokenize 返回整数 `[K,77]`，77 是官方默认 context_length |
| 图像表示 | `[B,D]`，encode_image 返回投影后的表示 |
| 文本表示 | `[K,D]`，encode_text 读取 EOT 位置并投影 |
| 图像对文本分数 | `[B,K]`，logits_per_image |
| 文本对图像分数 | `[K,B]`，上面的转置 |

B 是图像数量，K 是候选文本数量，D 是选定模型的共享维数；B 与 K 在推理时可以不同。encode_image / encode_text 单独返回的表示不保证已经单位归一化，官方 forward 才在计算匹配前进行归一化。[S26：encode_image / encode_text / forward](https://github.com/openai/CLIP/blob/a1d071733d7111c9c014f024669f959182114e33/clip/model.py)、[S27：tokenize](https://github.com/openai/CLIP/blob/a1d071733d7111c9c014f024669f959182114e33/clip/clip.py)

官方用法是先由 clip.load 返回模型和 preprocess，再用 clip.tokenize 处理候选文本。加载权重才能得到预训练模型的能力；本轮不执行下载。原始整数 token ID 不是自然语言编码器已经得到的语义向量。[S19：官方使用示例](https://github.com/openai/CLIP#usage)

E06 为形状教学使用图像 `[2,3,32,32]`、3 条整数序列 `[3,6]`、投影维数 16，输出 `[2,3]` 与 `[3,2]`。这里没有官方 BPE、77 位 context、EOT 选择或 Transformer 文本主干。随机分数没有图文识别能力。[E06：代码](/models/clip?section=code)

## 核心公式

### 条件与易错点

从相似度同时依赖图像和文本特征的计算图推导：若两侧参数均可训练，单向交叉熵同样可以向两个编码器传播梯度，不能说单向损失只训练其中一个编码器。双向损失分别约束图像检索文本与文本检索图像两个归一化方向；是否冻结某侧取决于训练设置。仅凭该目标不能承诺必然更鲁棒或提升某项实验准确率。

先分别投影并做 L2 归一化，再使用可学习的分数缩放：

$$
\hat v_i=\frac{v_i}{\|v_i\|_2},\quad
\hat t_j=\frac{t_j}{\|t_j\|_2},\quad
S_{ij}=\exp(s)\hat v_i^\top\hat t_j
$$

S 的形状是 B×K，s 对应官方 logit_scale；它不是样本数或特征维数。对候选文本方向做 softmax，只得到当前候选集合内的相对分布，不能当成开放世界的识别正确概率。[S26：forward](https://github.com/openai/CLIP/blob/a1d071733d7111c9c014f024669f959182114e33/clip/model.py)

训练时若有 N 个按同一顺序对齐的图文对，匹配矩阵为 N×N，对角线作为正对，两个方向的交叉熵取平均。这个目标说明为什么需要配对关系，不意味着本例完成了训练。[S18：§2.3 / Figure 3](https://arxiv.org/pdf/2103.00020#page=5)

### PyTorch 交叉熵与方向

以下是依据配对目标和 PyTorch 2.6 接口编写的教学计算，不是 OpenAI CLIP 仓库提供的训练函数。官方 `CLIP.forward` 返回匹配矩阵，不直接计算交叉熵损失。[S26：forward](https://github.com/openai/CLIP/blob/a1d071733d7111c9c014f024669f959182114e33/clip/model.py)

规定 `S[i,j]` 是第 i 张图像与第 j 条文本的 logit，且同索引的 N 个图文对正确对齐：每行固定一张图像，在所有文本中选正确项；`S.T` 每行固定一条文本，在所有图像中选正确项。二维 `F.cross_entropy` 使用 `[样本数,类别数]` 输入和 `[样本数]` 的整数目标，候选类别位于维度 1。函数没有 `axis` 或 `dim` 参数，也不需要先做 softmax。[S31：接口及 Shape](https://github.com/pytorch/pytorch/blob/v2.6.0/torch/nn/functional.py#L3404-L3501)

```python
import torch
from torch.nn import functional as F

# 自编数值例：3个配对，不是预训练模型的真实得分。
scores = torch.tensor([[3.0, 1.0, 0.0],
                       [0.0, 4.0, 2.0],
                       [1.0, 0.0, 2.0]])
labels = torch.arange(scores.shape[0])
loss_image = F.cross_entropy(scores, labels)    # 图像 → 文本
loss_text = F.cross_entropy(scores.T, labels)   # 文本 → 图像
loss = (loss_image + loss_text) / 2
```

这里两个调用的类别轴均为维度 1，方向通过输入矩阵转置改变。不要将原论文伪代码中的 `axis=0/1` 直接改写成 PyTorch 的 `dim=0/1`；论文伪代码未给出该交叉熵函数的定义，不能仅凭 `axis` 的数值猜测它的接口语义。上面的明确矩阵约定与 PyTorch 接口才是本例方向的依据。

`labels` 表示配对位置，不是人工类别标签，也不是 embedding 的维数。若文本次序改变，目标标签须随配对位置调整，不能继续无条件使用对角线；N×K 的推理匹配可以不是方阵，本例 N×N 的双向训练目标另有“批次内一一配对”的条件。

### 文本重排后两组标签不同

从上述按索引一一配对的 S 出发，仅把文本列按 `perm=[2,0,1]` 重排，记新矩阵 `S_new=S[:,perm]`。新列 0、1、2 分别是原文本 2、0、1；图像行仍保持原顺序。

- 图像→文本：原图像 0、1、2 分别要选新列 1、2、0，所以目标是 `[1,2,0]`，即 `argsort(perm)`。
- 文本→图像：`S_new.T` 的行已经按原文本 2、0、1 排列，各行应选择原图像 2、0、1，所以目标是 `[2,0,1]`，即 `perm`。

两个方向的查询顺序不同，不能都使用 `[1,2,0]`，也不能继续都使用 `arange(N)`。这是由明确配对约定推导的教学例，而非论文伪代码轴号的推断。

```python
perm = torch.tensor([2, 0, 1])
scores_new = scores[:, perm]
labels_image = torch.argsort(perm)             # [1,2,0]
labels_text = perm                            # [2,0,1]
loss_image_new = F.cross_entropy(scores_new, labels_image)
loss_text_new = F.cross_entropy(scores_new.T, labels_text)
loss_new = (loss_image_new + loss_text_new) / 2
assert torch.allclose(loss_image_new, loss_image)
assert torch.allclose(loss_text_new, loss_text)
```

## 最小代码

E06 独立编写，只演示双编码器与匹配计算。视觉分支是小 CNN，文本分支为 Embedding 加有效 token 均值；它们都不是官方 CLIP 主干。0 是本例 padding；空文本会报错，避免把空序列伪装成有效输入。两条图像与三条文本说明匹配矩阵不必是方阵。

```python
import math
import torch
from torch import nn
from torch.nn import functional as F


class TinyDualEncoder(nn.Module):
    def __init__(self, vocab_size=40, shared_dim=16):
        super().__init__()
        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )
        self.text_embedding = nn.Embedding(vocab_size, 12, padding_idx=0)
        self.image_projection = nn.Linear(8, shared_dim, bias=False)
        self.text_projection = nn.Linear(12, shared_dim, bias=False)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))

    def encode_image(self, images):
        return F.normalize(self.image_projection(self.image_encoder(images)), dim=-1)

    def encode_text(self, token_ids):
        valid = token_ids.ne(0)
        if not valid.any(dim=1).all():
            raise ValueError("本例每条文本至少要有一个非 padding token")
        embedded = self.text_embedding(token_ids)
        pooled = (embedded * valid.unsqueeze(-1)).sum(1) / valid.sum(1, keepdim=True)
        return F.normalize(self.text_projection(pooled), dim=-1)

    def forward(self, images, token_ids):
        visual = self.encode_image(images)
        textual = self.encode_text(token_ids)
        logits = self.logit_scale.exp() * visual @ textual.T
        return visual, textual, logits, logits.T


torch.manual_seed(7)
model = TinyDualEncoder().eval()
images = torch.randn(2, 3, 32, 32)
token_ids = torch.tensor([[1, 2, 3, 0, 0, 0], [4, 5, 6, 7, 0, 0],
                          [8, 9, 10, 11, 12, 0]])
with torch.no_grad():
    visual, textual, scores_image, scores_text = model(images, token_ids)
assert visual.shape == (2, 16)
assert textual.shape == (3, 16)
assert scores_image.shape == (2, 3)
assert scores_text.shape == (3, 2)
assert torch.allclose(scores_text, scores_image.T)
print("visual / textual:", tuple(visual.shape), tuple(textual.shape))
print("scores:", tuple(scores_image.shape), tuple(scores_text.shape))
```

本例的 encode_* 已归一化，与官方 encode_* 单独调用的约定不同；接口区别已明确，不能直接拿两段代码互换。

## 与相关模型的联系

CLIP → CLIP 的 ViT 视觉编码器 → ViT 架构家族。该视觉主干有 patch 投影与 class token，但输出目标是共享图文表示；它不是调用 Torchvision 的分类模型。这是已核对的应用组成关系。[S18：§2.4–2.5](https://arxiv.org/pdf/2103.00020#page=5)、[S26：VisionTransformer](https://github.com/openai/CLIP/blob/a1d071733d7111c9c014f024669f959182114e33/clip/model.py)

CLIP 的文本 Transformer 使用多头自注意力与残差相加。它采用 LayerNorm 后进入分支，注意力还有因果遮罩；并不等于本库 TinySeq2Seq 的 encoder-decoder 翻译接口。官方 encode_text 读取 EOT 位置，而 E06 的均值聚合仅供形状教学。[S26：ResidualAttentionBlock / encode_text](https://github.com/openai/CLIP/blob/a1d071733d7111c9c014f024669f959182114e33/clip/model.py)

CLIP 的核心匹配包含投影后归一化与缩放余弦相似度。多模态体现为图像与文本两种输入及对齐目标，不是因为输入图片有 RGB 三个通道。[S26：forward](https://github.com/openai/CLIP/blob/a1d071733d7111c9c014f024669f959182114e33/clip/model.py)

## 图示与资料状态

论文第 2 页 Figure 1 和第 5 页 Figure 3 可用于理解图文对齐。本轮只链接原图位置，不插入图像。候选文本写法会影响匹配含义，未整理某个故障类别的提示词或验证真实诊断能力。[原论文](https://arxiv.org/pdf/2103.00020#page=2)

## 来源与定位

- S18：[CLIP 原论文](https://arxiv.org/pdf/2103.00020)，§2.3–2.5、Figure 3；核对对比学习、视觉变体与配对目标。
- S19：[OpenAI CLIP 仓库](https://github.com/openai/CLIP)，Usage；用于定位官方加载与接口入口。
- S26：[固定 commit model.py](https://github.com/openai/CLIP/blob/a1d071733d7111c9c014f024669f959182114e33/clip/model.py)，CLIP、VisionTransformer、ResidualAttentionBlock、forward；不执行该远程代码。
- S27：[同 commit clip.py](https://github.com/openai/CLIP/blob/a1d071733d7111c9c014f024669f959182114e33/clip/clip.py)，load、tokenize 与 context_length=77。
- S31：[PyTorch v2.6.0 cross_entropy](https://github.com/pytorch/pytorch/blob/v2.6.0/torch/nn/functional.py#L3404-L3501)，接口及输入/目标形状；损失数值例为自行编写。
- E06：[TinyDualEncoder 教学示例](/models/clip?section=code)，仅使用本地已有 PyTorch；已验证归一化、匹配矩阵、padding 与空文本边界。

## 可继续追问

“B、K、D 分别是什么？”“两张图与三条文本为什么输出 2×3？”“encode_text 与 forward 的归一化约定一样吗？”“CLIP 的 ViT 与图像分类 ViT 输出有什么不同？”“为什么 RGB 不代表多模态？”“为什么这个例子没有真正识别能力？”
