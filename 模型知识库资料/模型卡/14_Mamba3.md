# Mamba-3：2026年的状态空间改进

模型ID：mamba3。固定依据2603.15569v1，2026-03-16。此处描述该版本，不推断所有后续实现。

## 概览与核心机制

Mamba-3结合更丰富的离散化递推、复值状态更新和多输入多输出（MIMO）设计。目标包括更丰富的状态跟踪和推理解码计算利用；理论复杂度与实际设备延迟是不同问题。[S40：§3](https://arxiv.org/pdf/2603.15569v1#page=4)

## 输入输出与数据流

骨干仍接收逐位置序列特征，并输出逐位置表示。用`[B,L,D]`说明批次、长度和模型维数；内部状态维数N不是类别数。复值描述内部状态动力学，不要求用户提供复数文本。

MIMO指状态空间层内部的多输入多输出通道，不表示必须输入图像和文本两种模态。语言任务需token嵌入和词表输出头；振动等信号任务则需相应编码和读出。[S40：§3.2–3.3](https://arxiv.org/pdf/2603.15569v1#page=8)

## 核心公式

论文将离散更新扩展为包含前一位置输入的三项形式：

$$
h_t=\alpha_t h_{t-1}+\beta_t B_{t-1}x_{t-1}+\gamma_t B_t x_t.
$$

alpha、beta、gamma取决于离散化；该式解释Table 1中的一般框架，不覆盖全部复值和MIMO张量配置。不能删去前一输入项后仍声称是完整Mamba-3递推。[S40：Table 1、§3.1](https://arxiv.org/pdf/2603.15569v1#page=5)

## 最小代码

原创结构伪代码，不执行作者内核，也不提供预训练推理。

```text
parameters = input_dependent_projection(sequence)
state = exponential_trapezoidal_update(previous_state, previous_input, current_input, parameters)
state = complex_rotation_or_equivalent_update(state, parameters)
representation = mimo_readout_and_gating(state, parameters)
output = output_projection(representation)
```

这些行按概念分解，不表示作者代码逐行采用这个调用次序。

## 与相关模型的联系

Mamba-3沿状态空间序列模型路线改进Mamba-2，但不是简单增大状态维数。原论文§3.4的总体网络交替使用Mamba-3和SwiGLU块；不能把原始Mamba“不含独立MLP块”的说明推广到此处。更新后的块用指数梯形递推替代原SSD层，并调整卷积、归一化与偏置。[S40：§3.4、Fig.2](https://arxiv.org/pdf/2603.15569v1#page=11)

## 应用流程与使用边界

用于解释序列表示、状态跟踪与语言模型骨干演进；具体输入预处理与任务头仍取决于任务。这里没有训练语言模型，也没有测评故障诊断准确率或复现解码速度。

## 架构图与论文插图

Fig.2直接对照Mamba-2与Mamba-3块：先看被替换的状态更新，再看归一化、卷积和投影。MIMO的含义是内部线性系统的通道组织。[原图第11页](https://arxiv.org/pdf/2603.15569v1#page=11)

## 来源与定位

- S40：[2026年固定论文v1](https://arxiv.org/pdf/2603.15569v1)。
- S43：[作者仓库](https://github.com/state-spaces/mamba)，仅实现入口。
- A06：原创结构伪代码。

## 可继续追问

“MIMO等于多模态吗？”“为何递推包含前一个输入？”“与Mamba-2相比改了什么？”“复值状态意味着输入必须是复数吗？”
