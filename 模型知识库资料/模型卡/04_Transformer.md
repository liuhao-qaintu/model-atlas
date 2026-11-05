# Transformer：区分序列、隐藏特征与词表输出

模型 ID：transformer。类型：架构家族。整理日期：2026-10-13。

本卡围绕经典 encoder-decoder 结构及 PyTorch 2.6 nn.Transformer。E04 是加上词嵌入、可学习位置和输出层的最小教学封装，不是从零复现论文，也不代表所有 Transformer 或现代大模型的完整实现。

## 概览与核心机制

原论文提出以注意力为核心的 encoder-decoder，并展示机器翻译任务。编码器处理源序列，解码器同时接收目标前缀和编码器表示。注意力、前馈网络与残差连接承担不同作用。[S13：原论文 §3](https://arxiv.org/pdf/1706.03762#page=3)

PyTorch 的 TransformerEncoderLayer 包含自注意力和逐位置前馈网络；TransformerDecoderLayer 还包含面向编码器 memory 的注意力。两类层都使用残差相加和 LayerNorm，具体顺序取决于 norm_first；本例沿用默认 False。[S21：编码器层](https://docs.pytorch.org/docs/2.6/generated/torch.nn.TransformerEncoderLayer.html)、[S22：解码器层](https://docs.pytorch.org/docs/2.6/generated/torch.nn.TransformerDecoderLayer.html)、[S24：固定版本源码](https://github.com/pytorch/pytorch/blob/v2.6.0/torch/nn/modules/transformer.py)

## 结构与原理

### 从一次注意力计算理解Q、K、V

Q代表“当前要找什么”，K代表“各位置可被匹配的特征”，V是实际被加权汇总的内容。这是理解计算的比喻，实际三者是由输入和可学习投影得到的张量，不是人工填写的自然语言问题。点积给出匹配分数，softmax把一行中的可用键位置归一化，最后用这些权重对V求和。[原论文§3.2](https://arxiv.org/pdf/1706.03762#page=4)

本例只用一个查询、两个键，且单头维数为2：Q=`[1,0]`，K的两行为`[1,0]`、`[0,1]`，V的两行为`[2,0]`、`[0,4]`。QK转置的结果为 `[1,0]`；除以sqrt(2)得到约 `[0.7071,0]`；softmax权重约 `[0.6698,0.3302]`；汇总V得到约 `[1.3395,1.3210]`。这是特征向量，不是两个类别的概率；注意力权重归一化的是键位置。

```python
import math
import torch

q = torch.tensor([[[1., 0.]]])
k = torch.tensor([[[1., 0.], [0., 1.]]])
v = torch.tensor([[[2., 0.], [0., 4.]]])
scaled_scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
attention_weights = scaled_scores.softmax(dim=-1)
attention_result = attention_weights @ v
assert torch.allclose(attention_result, torch.tensor([[[1.3395231, 1.3209538]]]), atol=1e-6)
blocked_scores = scaled_scores.masked_fill(torch.tensor([[[False, True]]]), -torch.inf)
assert torch.allclose(blocked_scores.softmax(-1) @ v, torch.tensor([[[2., 0.]]]))
```

### 多头和前馈网络各自做什么

本卡E04中D=32、nhead=4，每头维数8；不是四个头各自保留32维，再把维数永久扩成128。不同头通过各自投影学习不同的匹配方式，拼接后再做输出投影回D。D必须能被头数整除，但头数越多不保证效果越好。[MultiheadAttention接口](https://github.com/pytorch/pytorch/blob/v2.6.0/torch/nn/modules/activation.py)

注意力在不同序列位置之间传递内容；前馈网络对每个位置应用相同的 `Linear(32,64) → ReLU → Linear(64,32)`，借助中间维数和非线性变换该位置的表示。64是本例dim_feedforward，不是词表大小，也不改变最终D=32。残差连接把子层输出与输入结合；LayerNorm是另一个不同的操作。本例norm_first=False，在残差相加后做相应归一化；不能把原论文或其他变体的顺序直接套过来。[固定版本EncoderLayer、DecoderLayer](https://github.com/pytorch/pytorch/blob/v2.6.0/torch/nn/modules/transformer.py)

### 自注意力、交叉注意力为什么长度不同

自注意力的Q/K/V来自同一序列。解码器交叉注意力的Q来自目标侧，K/V来自编码器memory。本例S=8、T=6时，单头交叉注意力分数矩阵为 `[B,6,8]`；每个目标查询可以在8个源位置中取信息，输出仍对应6个查询位置。因此输出长度跟目标侧T对齐，不会因为读取了8个源位置就变成8。

这些机制说明经典encoder-decoder；不把它等同于所有decoder-only语言模型，也不将“多头”误称为“多模态”。

## 输入输出与数据流

先区分接口：nn.Transformer 接收浮点特征；E04 教学封装接收整数 token ID，再通过 Embedding 转为特征。batch_first=True 时使用 `[B,S,D]` 和 `[B,T,D]`，S 是源长度，T 是目标前缀长度，D 是隐藏维数。[S14：Shape](https://docs.pytorch.org/docs/2.6/generated/torch.nn.Transformer.html#torch.nn.Transformer.forward)

| 本例步骤 | 形状 / 含义 |
|---|---|
| 源 token ID | `[2,8]`，整数，词表大小 31 |
| 目标前缀 token ID | `[2,6]`，整数，词表大小 23 |
| 源嵌入与位置 | `[2,8,32]` |
| 目标嵌入与位置 | `[2,6,32]` |
| Transformer 返回的解码特征 | `[2,6,32]` |
| 词表输出投影 | `[2,6,23]`，每个目标位置的 logits |

输出长度跟目标输入 T 对齐，不是源长度 S，也不是自动翻译出一句话。D=32、4 个头使每头维数为 8；前馈层中间维数 64 不会把最终隐藏维数变成 64。[S14：输出约定](https://docs.pytorch.org/docs/2.6/generated/torch.nn.Transformer.html)、[S23：MultiheadAttention](https://github.com/pytorch/pytorch/blob/v2.6.0/torch/nn/modules/activation.py)、[E04：本例代码](/models/transformer?section=code)

### 自定义解码器后，哪些结论需要重新核对

上表说明本卡默认解码器与TinySeq2Seq封装。PyTorch 2.6.0构造函数若收到`custom_decoder`，会将它赋给`self.decoder`；`Transformer.forward`检查src/tgt的批量及特征维数后，调用`self.decoder(...)`并直接`return output`，没有再次检查输出最后一维是否等于d_model。只看这个return，能确认返回调用结果，不能确认替换模块内部的运算、输出形状或任务含义。[S24：构造与forward](https://github.com/pytorch/pytorch/blob/v2.6.0/torch/nn/modules/transformer.py#L144-L289)

下面是刻意简化的反例：把目标特征前两维定义为两个教学符号的分数，在自定义解码器内归一化；输入特征维数4，返回最后一维却是2。它不读取memory，不是翻译模型，未训练，也不是给默认解码器追加结构的建议。只有看到了这段明确的符号定义，才知道归一化对应哪两个候选；任意特征经过softmax并不会自动变成词表概率。

```python
class TwoSymbolDecoder(nn.Module):
    def forward(self, tgt, memory, **kwargs):
        # 本例人为定义：前两维对应符号0、1；不代表学到语言能力。
        symbol_scores = tgt[..., :2]
        return symbol_scores.softmax(dim=-1)


custom_core = nn.Transformer(
    d_model=4, nhead=2, num_encoder_layers=1, dim_feedforward=8,
    dropout=0.0, batch_first=True, custom_decoder=TwoSymbolDecoder(),
).eval()
custom_src = torch.zeros(1, 3, 4)
custom_tgt = torch.tensor([[[0., 1., 2., 3.], [2., 0., 1., 3.]]])
with torch.no_grad():
    custom_probabilities = custom_core(custom_src, custom_tgt)
assert custom_probabilities.shape == (1, 2, 2)
assert torch.allclose(custom_probabilities, custom_tgt[..., :2].softmax(-1))
assert torch.allclose(custom_probabilities.sum(-1), torch.ones(1, 2))
```

先运行“代码实现”中的导入，再运行此例。默认实现、替换模块及本例约定应分别说明；当用户未提供自定义实现时，应指出未知，不能沿用默认形状作保证，也不能把“未归一化特征”一概叫作logits。

## 核心公式

缩放点积注意力按 Q 与 K 的匹配加权 V，M 表示额外的加性遮罩。[S13：§3.2.1](https://arxiv.org/pdf/1706.03762#page=4)

$$
\operatorname{Attention}(Q,K,V)=\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}+M\right)V
$$

单头可以用 `[B,Tq,dk]`、`[B,Tk,dk]`、`[B,Tk,dv]` 表示 Q/K/V，权重为 `[B,Tq,Tk]`。本例每头 dk=8，不是总隐藏维数 32。自注意力的 Q/K/V 来自同一序列；解码器交叉注意力用目标侧产生 Q、编码器 memory 产生 K/V。[S13：§3.2.2–3.2.3](https://arxiv.org/pdf/1706.03762#page=5)

逐位置前馈网络对每个位置使用同一组参数，但不意味着所有位置的特征相同。原论文还使用正弦/余弦位置编码；E04 为方便阅读使用可学习的位置 Embedding，配置与论文不同。[S13：§3.3、§3.5](https://arxiv.org/pdf/1706.03762#page=5)

## 掩码与训练接口

### 条件与易错点

区分参数名：src_mask、tgt_mask约束位置之间的注意关系，典型形状分别为 `[S,S]`、`[T,T]`；src_key_padding_mask、tgt_key_padding_mask按样本标记无效键位置，典型形状为 `[B,S]`、`[B,T]`。不能将src_mask直接称为专用padding参数。因果掩码对应自回归解码的未来信息限制，不是所有Transformer编码器都必须采用的设置。

在这里使用的 nn.Transformer 接口中，布尔 mask 的 True 表示禁止注意；不能把该语义不加核对地搬到其他注意力 API。tgt_mask 为 `[T,T]`；右上角 True 阻止目标位置看到后续位置。padding mask 为 `[B,S]` 或 `[B,T]`，用于屏蔽补齐的键，不替代因果遮罩。[S14：forward 的 mask 说明](https://docs.pytorch.org/docs/2.6/generated/torch.nn.Transformer.html#torch.nn.Transformer.forward)

训练时目标前缀与监督标签通常错开一位；标签中的 padding 也需要在损失中忽略。仅添加 causal mask 并不等于完成自回归训练或推理；实际生成还需逐步追加 token、停止条件和解码策略。本卡不实现训练或生成循环。[E04：教学接口](/models/transformer?section=code)

### 两个布尔掩码叠加时，先分清三个索引

仅讨论本卡PyTorch 2.6.0的布尔掩码：二维`tgt_mask[i,j]`的i是query位置，j是key位置，这个矩阵在批次间共享；`tgt_key_padding_mask[b,j]`的b才是样本编号，j仍是key位置。位置均从0开始。前者限制位置对，后者对一个样本的所有query屏蔽指定key。不能把`tgt_mask`的第一个索引当成batch，也不能因为屏蔽了padding键，就认定该位置作为query的输出必为0。[S14：forward形状](https://docs.pytorch.org/docs/2.6/generated/torch.nn.Transformer.html#torch.nn.Transformer.forward)

把两项“禁止”合到一个三维教学表示中，就是$B[b,i,j]=A[i,j]\lor P[b,j]$。这里只适用于上述两个布尔掩码；A为False仅表示它没禁止这一对，P仍可能禁止。因果限制通常也编码进A，限制未来键；`tgt_is_causal`是需要正确填写的因果提示，不能代替核对实际mask。浮点加性mask及其他API的布尔语义需要另行核对。[S23：MultiheadAttention.forward与merge_masks](https://github.com/pytorch/pytorch/blob/v2.6.0/torch/nn/modules/activation.py#L1381-L1448)

下例固定Q/K为0，使可用键均分权重，便于直接看出屏蔽位置。B=2、T=5，只有第1个样本的key4被padding掩码禁止：第0个样本的query2仍有5个可用键，第1个样本只有4个。再叠加因果mask后，两者的query2都只保留key0/1/2。这些是指定数值下的可运行演示，不代表真实训练时权重均匀，也不保证未被屏蔽位置一定获得显著权重。

```python
mask_probe = nn.MultiheadAttention(4, 1, dropout=0.0, bias=False, batch_first=True)
with torch.no_grad():
    mask_probe.in_proj_weight.zero_()
    mask_probe.in_proj_weight[8:].copy_(torch.eye(4))  # V投影
    mask_probe.out_proj.weight.copy_(torch.eye(4))
    probe_tokens = torch.arange(40, dtype=torch.float32).reshape(2, 5, 4)
    pair_mask = torch.zeros(5, 5, dtype=torch.bool)
    padding_mask = torch.zeros(2, 5, dtype=torch.bool)
    padding_mask[1, 4] = True
    padding_output, padding_weights = mask_probe(
        probe_tokens, probe_tokens, probe_tokens,
        attn_mask=pair_mask, key_padding_mask=padding_mask,
    )
    assert torch.allclose(padding_weights[0], torch.full((5, 5), 0.2))
    assert torch.equal(padding_weights[1, :, 4], torch.zeros(5))
    assert torch.allclose(padding_weights[1, :, :4], torch.full((5, 4), 0.25))
    assert bool((padding_output[1, 4] != 0).any())  # 被屏蔽为key不等于query输出为0
    causal_pair_mask = torch.ones(5, 5, dtype=torch.bool).triu(diagonal=1)
    _, combined_weights = mask_probe(
        probe_tokens, probe_tokens, probe_tokens,
        attn_mask=causal_pair_mask, key_padding_mask=padding_mask,
    )
    assert torch.equal(combined_weights[:, 2, 3:], torch.zeros(2, 2))
    assert torch.allclose(combined_weights[:, 2, :3], torch.full((2, 3), 1 / 3))
```

先运行“代码实现”中的导入，再运行此例。这里展示两个同类型布尔掩码，未把一整行键全部屏蔽，也不将注意力权重当成词表预测概率。

## 最小代码

E04 为独立编写的封装，使用已有 PyTorch 的架构模块。0 约定为 padding，位置表最多 32 个位置；本例输入均为非 padding 的随机 ID。词嵌入、位置和词表投影由本例提供，nn.Transformer 本身不负责将原始文本分词。

```python
import torch
from torch import nn


class TinySeq2Seq(nn.Module):
    def __init__(self, src_vocab=31, tgt_vocab=23, max_length=32):
        super().__init__()
        self.src_tokens = nn.Embedding(src_vocab, 32, padding_idx=0)
        self.tgt_tokens = nn.Embedding(tgt_vocab, 32, padding_idx=0)
        self.positions = nn.Embedding(max_length, 32)
        self.core = nn.Transformer(
            d_model=32, nhead=4, num_encoder_layers=2,
            num_decoder_layers=2, dim_feedforward=64,
            dropout=0.0, batch_first=True,
        )
        self.vocabulary_projection = nn.Linear(32, tgt_vocab)

    def add_position(self, ids, embedding):
        if ids.shape[1] > self.positions.num_embeddings:
            raise ValueError("序列长度超过位置表上限")
        position_ids = torch.arange(ids.shape[1], device=ids.device)
        return embedding(ids) + self.positions(position_ids)[None, :, :]

    def forward(self, source_ids, target_ids):
        source = self.add_position(source_ids, self.src_tokens)
        target = self.add_position(target_ids, self.tgt_tokens)
        causal_mask = torch.ones(
            target_ids.shape[1], target_ids.shape[1],
            dtype=torch.bool, device=target_ids.device,
        ).triu(diagonal=1)
        hidden = self.core(
            source, target, tgt_mask=causal_mask,
            src_key_padding_mask=source_ids.eq(0),
            tgt_key_padding_mask=target_ids.eq(0),
            memory_key_padding_mask=source_ids.eq(0),
        )
        return hidden, self.vocabulary_projection(hidden)


torch.manual_seed(7)
model = TinySeq2Seq().eval()
source_ids = torch.randint(1, 31, (2, 8))
target_ids = torch.randint(1, 23, (2, 6))
with torch.no_grad():
    hidden, logits = model(source_ids, target_ids)
assert hidden.shape == (2, 6, 32)
assert logits.shape == (2, 6, 23)
print("hidden:", tuple(hidden.shape))
print("logits:", tuple(logits.shape))
```

随机权重不会产生有意义的翻译。这里用框架模块演示数据流，没有展示其完整内部架构定义；内部注意力、前馈和残差实现可按 S24 的类名查阅。

## 推导与排错

### Teacher forcing怎样准备一条训练样本

只讨论本卡整数ID教学配置：PAD=0、BOS=1、EOS=2，目标词表大小23。完整目标为 `[1,7,9,2,0]`，传给解码器的输入为 `[1,7,9,2]`，监督标签为 `[7,9,2,0]`。第0位置看到BOS，预测7；第1位置看到历史前缀，预测9；以此类推。长度T=4，所以hidden是 `[1,4,32]`，词表logits是 `[1,4,23]`。BOS和EOS约定属于本例，不是nn.Transformer自动生成的内容。

训练允许将所有目标位置并行计算，但用因果mask禁止位置i读取其右侧的目标token；不等于训练时必须逐个token调用模型。nn.Transformer布尔tgt_mask中True表示禁止，4×4上三角mask为：

| 查询位置 / 键位置 | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| 0 | False | True | True | True |
| 1 | False | False | True | True |
| 2 | False | False | False | True |
| 3 | False | False | False | False |

### Padding、损失与概率分别在哪里处理

key_padding_mask屏蔽作为键的padding位置，不表示padding查询位置的输出一定为0。目标标签为PAD的位置还需在本例损失中忽略。交叉熵可直接接收未归一化的类别logits；先手动softmax再送入此接口，会改变所计算的目标，不是同一条损失公式。[PyTorch v2.6 cross_entropy定义](https://github.com/pytorch/pytorch/blob/v2.6.0/torch/nn/functional.py)

在 `[B,T,V]` 布局下，不能直接把这个三维张量当成CrossEntropyLoss默认的 `[B,C,...]` 输入；应把B与T合并为 `[B*T,V]`，把标签合并为 `[B*T]`，或明确转换为 `[B,V,T]`。本例忽略PAD后的平均损失只覆盖3个有效目标位置，不能仍除以4。

先运行“代码实现”中的TinySeq2Seq定义，再运行下段。它核对一次前向、损失与反向传播，没有执行优化器更新，也不代表已经训练好翻译模型。

```python
from torch.nn import functional as F

teacher_model = TinySeq2Seq().eval()
source_example = torch.tensor([[3, 4, 2, 0]])
full_target = torch.tensor([[1, 7, 9, 2, 0]])
decoder_input, labels = full_target[:, :-1], full_target[:, 1:]
teacher_hidden, teacher_logits = teacher_model(source_example, decoder_input)
assert teacher_hidden.shape == (1, 4, 32)
assert teacher_logits.shape == (1, 4, 23)
token_loss = F.cross_entropy(teacher_logits.reshape(-1, 23), labels.reshape(-1), ignore_index=0)
valid = labels.ne(0)
negative_log_likelihood = -teacher_logits.log_softmax(-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
manual_loss = negative_log_likelihood[valid].mean()
assert valid.sum().item() == 3
assert torch.allclose(token_loss, manual_loss)
token_loss.backward()  # eval不关闭autograd；这里只检查梯度存在。
assert teacher_model.vocabulary_projection.weight.grad is not None
```

### 如何排查未来信息泄露

固定随机种子和eval模式，只改解码输入末位置的一个非PAD token；因果mask生效时，较早位置的输出应在数值容差内保持一致。本卡dropout=0且两组输入的PAD位置相同，排除了这两项干扰。这个检查核对掩码行为，不能证明模型已学会语言，也不能证明数据预处理没有其他泄露。

```python
future_changed = decoder_input.clone()
future_changed[:, -1] = 11
with torch.no_grad():
    _, prefix_original = teacher_model(source_example, decoder_input)
    _, prefix_changed = teacher_model(source_example, future_changed)
assert torch.allclose(prefix_original[:, :-1], prefix_changed[:, :-1], atol=1e-5, rtol=1e-5)
```

若先前位置也变化，依次检查因果mask方向、API的布尔含义、dropout/模式、PAD位置是否改变，以及是否误把完整标签放进了未经约束的输入。纯手算softmax的一行若全部为负无穷，会出现未定义的归一化；不要为了屏蔽padding把一个有效查询的所有可用键都屏蔽。具体融合注意力内核的全屏蔽行为还应按该API核对，不能由手算例推断所有实现都会返回同一结果。

自回归推理则从BOS出发，将上一步选出的token追加为新输入，并设置EOS或最大长度等停止条件。它与训练时用真实前缀的teacher forcing不同；本卡未提供完整生成与训练循环。

## 与相关模型的联系

Transformer → TransformerEncoderLayer / TransformerDecoderLayer → 多头注意力、逐位置前馈网络、残差相加。编码器与解码器都使用多头注意力；解码器还对编码器 memory 做交叉注意力。[S21：编码器](https://docs.pytorch.org/docs/2.6/generated/torch.nn.TransformerEncoderLayer.html)、[S22：解码器](https://docs.pytorch.org/docs/2.6/generated/torch.nn.TransformerDecoderLayer.html)、[S24：结构实现](https://github.com/pytorch/pytorch/blob/v2.6.0/torch/nn/modules/transformer.py)

ResNet-18 的 BasicBlock 与这里的 Transformer 层都有残差相加，但分支函数、归一化和数据布局不同。共享连接机制不等于权重相同，也不证明某一模型由另一模型直接改造。

TinySeq2Seq 的词表投影是 Linear(32, 23)，作用于每个目标位置。它不是把整条序列变成一个类别的分类头，应与 CNN 图像分类头分开记录。[E04：输出投影](/models/transformer?section=code)

本库已整理 [ViT 模型卡](/models/vit) 和 [CLIP 模型卡](/models/clip)。ViT 将图像 patch 序列送入视觉 Transformer；CLIP 的视觉分支与文本分支需分别说明所属模型及固定实现。具体组成与论文采用关系以相应章节正文和已审核的关系为依据。

## 图示与资料状态

原论文的 Fig.1/2 可在 S13 PDF 中查看，本项目没有下载或插入图片。已核对与本卡相关的结构、注意力、前馈及位置编码段落；没有收录论文性能指标或近期 SOTA 排名。

## 来源与定位

- S13：[Attention Is All You Need](https://arxiv.org/pdf/1706.03762)，§3.1–3.5；在阅读页按章节定位原论文。
- S14：[PyTorch 2.6 Transformer](https://docs.pytorch.org/docs/2.6/generated/torch.nn.Transformer.html)，forward、Shape、mask 与 batch_first。
- S21：[TransformerEncoderLayer](https://docs.pytorch.org/docs/2.6/generated/torch.nn.TransformerEncoderLayer.html)，层组成与 norm_first。
- S22：[TransformerDecoderLayer](https://docs.pytorch.org/docs/2.6/generated/torch.nn.TransformerDecoderLayer.html)，解码器组成及 memory。
- S23：[PyTorch v2.6.0 activation.py](https://github.com/pytorch/pytorch/blob/v2.6.0/torch/nn/modules/activation.py)，MultiheadAttention 的头数与维度约定。
- S24：[PyTorch v2.6.0 transformer.py](https://github.com/pytorch/pytorch/blob/v2.6.0/torch/nn/modules/transformer.py)，编码器/解码器前向、残差与前馈实现。
- E04：[TinySeq2Seq 教学示例](/models/transformer?section=code)，原创封装与实际前向检查。
- S44：[PyTorch v2.6.0 functional.py](https://github.com/pytorch/pytorch/blob/v2.6.0/torch/nn/functional.py)，cross_entropy输入布局、原始logits和ignore_index；本节损失布局用自编数值例验证。

## 可继续追问

“32、64、23 分别是什么？”“True 是屏蔽还是允许？”“为什么输出长度是 6 而不是 8？”“为什么整数 ID 不能直接交给 nn.Transformer？”“多头注意力和跨模态是不是同一概念？”“与 BasicBlock 共享了什么连接机制？”
