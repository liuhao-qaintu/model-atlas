# Mamba-2：结构化状态空间对偶

模型ID：mamba2。固定依据2405.21060v1，2024-05-31。SSD不是任意softmax Transformer的完全等价替换。

## 概览与核心机制

Mamba-2利用结构化状态空间对偶（SSD）连接SSM与特定结构的注意力计算。同一个序列变换可以从递推状态或结构化矩阵角度理解，从而设计更适合矩阵乘法的算法。[S39：§5](https://arxiv.org/pdf/2405.21060v1#page=16)

## 输入输出与数据流

以B、L、D分别表示批次、长度与通道，模块输入输出通常都是`[B,L,D]`逐位置特征；这组说明符号不固定任意检查点参数。内部按head组织状态和通道，状态展开维数不是输出类别数。语言模型仍需token嵌入和词表头，其他任务需自己的读出。

## 核心公式

在论文讨论的标量乘单位阵的状态转移约束下，可用下式解释单通道递推：

$$
h_t=a_t h_{t-1}+B_t x_t,\qquad y_t=C_t^{\top}h_t.
$$

这里B吸收离散化因素，不能和另一论文中连续时间B的记号无条件混用。SSD关联的是受结构约束的SSM与masked attention形式，不意味着所有带softmax注意力的网络逐元素相同。[S39：§5.1–5.3](https://arxiv.org/pdf/2405.21060v1#page=17)

## 最小代码

原创结构伪代码；只是说明数据流，不是作者可运行API。

```text
x, gate, B, C, delta = parallel_input_projection(sequence)
x, B, C = short_causal_conv(x, B, C)
states_and_outputs = structured_state_space_duality(x, A, B, C, delta)
y = gated_normalization(states_and_outputs, gate)
output = output_projection(y)
```

## 与相关模型的联系

Mamba-2在Mamba的选择性SSM基础上改进模块与算法。输入相关投影能并行计算，head与状态转移设计服务于高效SSD计算；不是只改名称。Mamba-2与Transformer的联系需要区分数学对偶和具体网络实现。[S39：§7、Fig.6](https://arxiv.org/pdf/2405.21060v1#page=23)

## 应用流程与使用边界

序列输入→Mamba-2表示→任务头。长序列处理是使用方向，但实际显存和延迟依赖实现、精度、设备与批次；本项目没有运行预训练任务模型或复现论文性能。

## 架构图与论文插图

原论文Fig.6并列展示Mamba与Mamba-2块，可观察投影位置、卷积、SSM/SSD和归一化差异。图中的共同箭头不能说明两者权重兼容。[原图第23页](https://arxiv.org/pdf/2405.21060v1#page=23)

## 来源与定位

- S39：[固定论文v1](https://arxiv.org/pdf/2405.21060v1)，§5与§7。
- S43：[作者仓库](https://github.com/state-spaces/mamba)。
- A05：本卡原创结构伪代码。

## 可继续追问

“SSD为什么有递推和矩阵两种形式？”“它与Mamba有哪些结构差异？”“是否等价于任意Transformer？”
