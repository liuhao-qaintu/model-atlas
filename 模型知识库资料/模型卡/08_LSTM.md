# LSTM：序列输出、隐藏状态与记忆状态

模型 ID：lstm。整理日期：2026-10-13。

本卡采用 PyTorch v2.6.0 接口。E08 为双向、两层的序列分类教学封装，随机初始化；只核查接口，不训练或诊断真实设备。

## 概览与核心机制

LSTM 以输入门、遗忘门和输出门控制候选信息、记忆状态与隐藏状态。序列长度、输入特征数、隐藏维数和类别数承担不同作用。以下公式与接口均对应固定 PyTorch 实现，不宣称所有历史变体具有相同设置。[S11：LSTM 类文档](https://github.com/pytorch/pytorch/blob/v2.6.0/torch/nn/modules/rnn.py)

本例 TinyBiLSTM 包含一个双向 LSTM 主干和 Linear 分类头；主干用 input_size=6、hidden_size=8、num_layers=2、batch_first=True、bidirectional=True、proj_size=0。它用两个方向的最终隐藏状态组成整条序列的表示。[E08：代码](/models/lstm?section=code)

## 输入输出与数据流

### 条件与易错点

先说明布局条件：PyTorch nn.LSTM默认batch_first=False，批量输入output形状为 `[L,B,D×Hout]`；batch_first=True时为 `[B,L,D×Hout]`。h_n、c_n的层与方向轴不受batch_first影响。“最终有效状态”依赖是否正确处理变长输入；PackedSequence按给定长度处理，而普通含padding的张量不会自动跳过padding。

先区分三类返回值：`output` 是最后一层在每个位置的隐藏输出；`h_n` 是每层、每个方向完成各自整段递推后的隐藏状态；`c_n` 是相应的最终细胞状态。`c_n` 不直接出现在 `output` 中。这里的“最终”按递推方向定义，不能把两个方向都说成原序列末位。

| 设置与返回值 | 终点及使用条件 |
|---|---|
| 等长且无padding的序列；正向的h_n/c_n | 沿0→L−1递推，最终状态完成原序列L−1位置的更新 |
| 等长且无padding的序列；反向的h_n/c_n | 沿L−1→0递推，最终状态完成原序列0位置的更新 |
| PackedSequence按lengths处理的变长序列；正向 | 第i个样本的最终状态完成lengths[i]−1位置的更新 |
| PackedSequence按lengths处理的变长序列；反向 | 第i个样本的最终状态完成0位置的更新；假定本例采用右侧补齐 |
| 普通含padding的批量张量 | nn.LSTM仍按完整张量长度递推，不自动忽略padding，不能无条件称为最终有效状态 |

`h_0`、`c_0` 是用户传入的递推前初始状态；未提供时由接口使用零初值。它们与处理第一个输入后的输出不同。[S11：Inputs / Outputs](https://github.com/pytorch/pytorch/blob/v2.6.0/torch/nn/modules/rnn.py)

### 对照配置后再计算形状

下面固定B=2、L=5、F=6、两层、双向、batch_first=True、等长无padding，只改变proj_size。数值按PyTorch接口的Hout公式推导，并已用本机前向核查；不是同一模型同时返回两种形状。[S11：proj_size / Outputs](https://github.com/pytorch/pytorch/blob/v2.6.0/torch/nn/modules/rnn.py)

| 配置 | Hout | output | h_n | c_n | 最后一层的两个方向隐藏状态 |
|---|---|---|---|---|---|
| hidden_size=8，proj_size=0（原教学配置） | 8 | `[2,5,16]` | `[4,2,8]` | `[4,2,8]` | 正向 `output[:,-1,:8]`；反向 `output[:,0,8:]` |
| hidden_size=8，proj_size=4（投影核对配置） | 4 | `[2,5,8]` | `[4,2,4]` | `[4,2,8]` | 正向 `output[:,-1,:4]`；反向 `output[:,0,4:]` |

用户指定投影配置时，应使用Hout=proj_size计算输出和方向切片，不能先把无投影示例中的16写为该问题的output维数。普通张量与解包后的output可按这些轴阅读；直接传PackedSequence时，nn.LSTM直接返回的output也为PackedSequence，应先解包再使用本卡中的张量切片。

| 位置 | 本例形状 / 含义 |
|---|---|
| 浮点序列输入 | `[2,5,6]`；B=2，L=5，每步 F=6 |
| output | `[2,5,16]`；最后一层每步两个方向的隐藏特征 |
| h_n | `[4,2,8]`；层数×方向数、B、隐藏输出维数 |
| c_n | `[4,2,8]`；层数×方向数、B、记忆维数 |
| 最后一层双向最终状态拼接 | `[2,16]` |
| 序列分类 logits | `[2,3]`；3 个教学类别，无 softmax |

batch_first 只影响输入和 output 的布局，不影响 h_n/c_n。最后一层的前向和反向最终状态为 h_n[-2] 与 h_n[-1]；output[:, -1, :] 的反向部分并不是整条序列的反向最终状态，不能把二者直接视为等价。output 也不是类别分数，分类头属于 E08 封装。[S11：Outputs / Notes](https://github.com/pytorch/pytorch/blob/v2.6.0/torch/nn/modules/rnn.py)、[E08：状态读取](/models/lstm?section=code)

方向定位（本例batch_first=True、bidirectional=True、等长无padding、hidden_size=8、proj_size=0）：前向递推按位置0→L−1，反向递推按L−1→0。**在这些条件下**，最后一层的 `h_n[-2]` 等于 `output[:, -1, :8]`，`h_n[-1]` 等于 `output[:, 0, 8:]`。`output[:, -1, 8:]` 是反向分支处理L−1位置后得到的输出，已经完成第一步更新；它既不是递推前的初始状态 `h_0`，也不是完成整个反向递推后的最终状态（L=1时第一步也是最后一步）。有padding时，正向对应位置需按有效长度选择；有投影时，方向切片宽度需改用Hout，不能继续固定为8。[S11：bidirectional / Notes](https://github.com/pytorch/pytorch/blob/v2.6.0/torch/nn/modules/rnn.py)、[E08：状态读取](/models/lstm?section=code)

一般批量接口用 D 表示方向数，Hout 为 proj_size>0 时的投影维数，否则是 hidden_size：batch_first=True 时 output 为 `[B,L,D×Hout]`，默认 batch_first=False 时为 `[L,B,D×Hout]`；h_n 为 `[D×num_layers,B,Hout]`，c_n 为 `[D×num_layers,B,hidden_size]`。有投影时，隐藏与记忆的最后一维可以不同。本例没有投影，不能把 8 无条件推广到其他配置。[S11：proj_size / Inputs / Outputs](https://github.com/pytorch/pytorch/blob/v2.6.0/torch/nn/modules/rnn.py)

## 核心公式

下面是单方向、单层、无投影时的状态关系；各门的仿射输入省略参数下标，g_t 为候选记忆，⊙ 为逐元素乘法。

$$
i_t=\sigma(a_i),\quad f_t=\sigma(a_f),\quad
g_t=\tanh(a_g),\quad o_t=\sigma(a_o)
$$

$$
c_t=f_t\odot c_{t-1}+i_t\odot g_t,\qquad
h_t=o_t\odot\tanh(c_t)
$$

a_i 等来自当前输入与上一隐藏状态的仿射变换。双向模型分别沿两个方向递推，多层模型把上一层输出提供给下一层。上述 h_t 公式限定无投影；启用投影时还需应用投影矩阵。[S11：LSTM 公式与投影说明](https://github.com/pytorch/pytorch/blob/v2.6.0/torch/nn/modules/rnn.py)

## 变长序列与方向

### 条件与易错点

本例采用batch_first=True、bidirectional=True、proj_size=0、hidden_size=8，变长输入为右侧padding并按lengths打包。解包后，第i个样本的最后一层正向最终隐藏状态等于 `output[i, lengths[i]-1, :8]`，反向最终隐藏状态等于 `output[i, 0, 8:]`。短样本的 `output[i, -1, :8]` 对应解包后的补齐位置，不等于其正向最终状态；也不能把反向最终位置说成lengths[i]-1。按h_n的最后一层两个方向读取再拼接，可以避免混淆。上述索引是隐藏状态对应关系，不能用output切片读取c_n。

解包时本例使用默认padding_value=0：若样本有效长度小于解包后的长度，其末列两个方向的输出都是补齐值。此时不能把 `output[i,-1,8:]` 解释为该短样本的反向第一步输出；真正的反向第一步输出在 `output[i,lengths[i]-1,8:]`，反向最终输出在 `output[i,0,8:]`。只有等长无padding时，才可用统一的最后一列定位反向第一步。若有效长度为1，反向第一步也是最后一步；这仍不是递推前的初始状态。[S28：pad_packed_sequence](https://github.com/pytorch/pytorch/blob/v2.6.0/torch/nn/utils/rnn.py)

向输入末尾补零并不等于 LSTM 自动忽略补齐位置。E08 可传有效长度，用 pack_padded_sequence 处理，再用 pad_packed_sequence 恢复 output。lengths 为 CPU 整数向量，使用 enforce_sorted=False 保留未预先排序的批次；本例输出恢复到原输入长度 L。它与 Transformer 的布尔注意力 mask 是不同接口。[S28：pack / pad](https://github.com/pytorch/pytorch/blob/v2.6.0/torch/nn/utils/rnn.py)

双向主干会读取两个方向的信息，本例面向完整序列的分类。若任务只允许使用过去信息，不能直接把这一双向配置当成因果在线预测模型。本卡没有实现流式状态管理或预测未来步骤。[E08：bidirectional 配置](/models/lstm?section=code)

## 最小代码

E08 只使用已有 PyTorch。普通输入长度都为 5；可选 lengths 用于验证变长序列。分类标签和真实训练不在本例中。

```python
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class TinyBiLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.core = nn.LSTM(
            input_size=6, hidden_size=8, num_layers=2,
            batch_first=True, bidirectional=True,
        )
        self.classifier = nn.Linear(16, 3)

    def forward(self, values, lengths=None):
        if values.ndim != 3 or values.size(2) != 6:
            raise ValueError("本教学配置要求 [B,L,6] 浮点序列")
        if lengths is None:
            output, (h_n, c_n) = self.core(values)
        else:
            if lengths.ndim != 1 or lengths.numel() != values.size(0):
                raise ValueError("lengths 需与批次大小一致")
            if lengths.dtype != torch.long or (lengths <= 0).any() or (lengths > values.size(1)).any():
                raise ValueError("lengths 需为 1 到 L 的整数长度")
            packed = pack_padded_sequence(values, lengths.cpu(), batch_first=True, enforce_sorted=False)
            encoded, (h_n, c_n) = self.core(packed)
            output, _ = pad_packed_sequence(encoded, batch_first=True, total_length=values.size(1))
        features = torch.cat([h_n[-2], h_n[-1]], dim=-1)
        logits = self.classifier(features)
        return output, h_n, c_n, features, logits


torch.manual_seed(7)
model = TinyBiLSTM().eval()
values = torch.randn(2, 5, 6)
with torch.no_grad():
    output, h_n, c_n, features, logits = model(values)
assert output.shape == (2, 5, 16)
assert h_n.shape == c_n.shape == (4, 2, 8)
assert features.shape == (2, 16)
assert logits.shape == (2, 3)
print("output:", tuple(output.shape))
print("h_n / c_n:", tuple(h_n.shape), tuple(c_n.shape))
print("features / logits:", tuple(features.shape), tuple(logits.shape))
```

## 与相关模型的联系

LSTM → 双向 LSTM 主干 → 隐藏/记忆状态与门控更新。E08 → 双向最终隐藏状态 → Linear 分类头。这里的分类头输出整条序列的类别分数，不是 TinySeq2Seq 的逐位置词表投影。[S11：LSTM](https://github.com/pytorch/pytorch/blob/v2.6.0/torch/nn/modules/rnn.py)、[E08：教学封装](/models/lstm?section=code)

本例 Linear(16,3) 使用线性映射完成分类；Autoencoder 教学编码器也使用 Linear，但目标是压缩表示而非输出类别。可比较相同算子的输入维数与输出用途，不能把重构输出记录成分类分数。[S29：Linear](https://github.com/pytorch/pytorch/blob/v2.6.0/torch/nn/modules/linear.py)、[E08：分类头](/models/lstm?section=code)

本库 1D CNN 使用 `[B,C,L]` 布局，E08 使用 `[B,L,F]`。通道和每步特征含义一致时，可以比较这两种布局；交换张量轴并不会把卷积网络变为循环网络。[E01：1D CNN 示例](/models/cnn-1d?section=code)、[E08：LSTM 示例](/models/lstm?section=code)

## 图示与资料状态

本轮以数据流表与门控公式解释状态，不插入图片。门控教材可参考 D2L 的 LSTM 章节；固定接口和本例输出仍以 S11 / E08 为准。[S12：教材章节](https://d2l.ai/chapter_recurrent-modern/lstm.html)

## 来源与定位

- S11：[PyTorch v2.6.0 rnn.py](https://github.com/pytorch/pytorch/blob/v2.6.0/torch/nn/modules/rnn.py)，LSTM 文档中的公式、Inputs、Outputs、Notes 与 proj_size。
- S28：[PyTorch v2.6.0 utils/rnn.py](https://github.com/pytorch/pytorch/blob/v2.6.0/torch/nn/utils/rnn.py)，pack_padded_sequence、pad_packed_sequence 与长度恢复。
- S29：[PyTorch v2.6.0 linear.py](https://github.com/pytorch/pytorch/blob/v2.6.0/torch/nn/modules/linear.py)，Linear 的维数约定。
- E08：[TinyBiLSTM 教学示例](/models/lstm?section=code)，验证环境 核查序列/状态/分类形状、变长输入、最终状态读取与特征边界。
- E01：[已有 1D CNN 示例](/models/cnn-1d?section=code)，仅用于输入布局对比。

## 可继续追问

“为什么 h_n 第一维是 4？”“batch_first 会改变 c_n 吗？”“output 为什么不是三类分数？”“双向模型最后时间步与最终状态有什么不同？”“补零为什么还需要 lengths？”“投影 LSTM 的 h_n 和 c_n 能不同维吗？”
