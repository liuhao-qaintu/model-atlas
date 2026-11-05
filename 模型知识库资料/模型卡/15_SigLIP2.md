# SigLIP 2：图文编码与多目标训练

模型ID：siglip2。固定依据2502.14786v1，2025-02-20。训练辅助模块与发布的编码器接口分别说明。

## 概览与核心机制

SigLIP 2在图文sigmoid配对目标之外，引入字幕相关训练、视觉自蒸馏、遮蔽预测及特定变体的数据筛选。不同训练阶段和变体并不完全相同。它首先是图文表示编码方法，不能直接等同于视觉对话大模型。[S41：§2、Fig.1](https://arxiv.org/pdf/2502.14786v1#page=2)

## 输入输出与数据流

以B表示图像数、K表示候选文本数、D表示共享表示维数：图像编码得到`[B,D]`，文本编码得到`[K,D]`；配对logits是`[B,K]`。这些是解释符号，不固定某个检查点D。

sigmoid配对分数独立计算，不是候选集合上总和为1的类别分布。想做零样本分类要提供候选文本，想生成问答则需额外生成模型。固定分辨率与NaFlex变体的输入预处理、patch长度及位置处理要分别看，不能随意替换图像尺寸。[S41：§2.1–2.4](https://arxiv.org/pdf/2502.14786v1#page=3)

## 核心公式

用简化符号表示配对logistic目标：

$$
s_{ij}=t f_i^{\top}g_j+b,\qquad \ell_{ij}=-\log\sigma(y_{ij}s_{ij}),\quad y_{ij}\in\{-1,+1\}.
$$

f、g是图文表示，t与b为缩放和偏置；这里只说明单对损失，不规定完整批次归约。sigmoid对每个图文配对分别判断正负，不要求对所有候选文本做softmax归一化。原论文将实现细节指向原始SigLIP，本卡公式是logistic目标的说明记号。[S41：§2.2](https://arxiv.org/pdf/2502.14786v1#page=3)

## 最小代码

原创接口伪代码，未加载作者权重。

```text
image_features = normalize(image_encoder(preprocess_images(images)))
text_features = normalize(text_encoder(tokenize(candidate_texts)))
pair_logits = learned_scale * image_features @ text_features.T + learned_bias
pair_scores = sigmoid(pair_logits)
```

不要把pair_scores当成每行归一化的多类概率，也不要把返回矩阵说成已生成的回答文本。

## 与相关模型的联系

SigLIP 2的视觉编码器采用ViT，文本编码器产生与图像对齐的表示。与CLIP都能形成图文表示和配对分数，但配对损失与训练目标不同；不能混用CLIP双向softmax损失公式。[S41：§2.1–2.2](https://arxiv.org/pdf/2502.14786v1#page=3)

## 应用流程与使用边界

图像与候选文本→双编码器→匹配分数，可用于检索和候选分类；视觉token也可接其他VLM的连接器。字幕解码器是训练时增加的学习目标，不意味着任意发布的双编码器接口都能直接生成回答。论文明确该辅助decoder不属于模型发布部分。[S41：§2.2](https://arxiv.org/pdf/2502.14786v1#page=3)

## 架构图与论文插图

Fig.1同时显示视觉塔、文本塔、辅助字幕decoder与EMA teacher。先区分训练时的损失分支与发布编码器，再看哪些箭头含stop-gradient。[原图第2页](https://arxiv.org/pdf/2502.14786v1#page=2)

## 来源与定位

- S41：[固定论文v1](https://arxiv.org/pdf/2502.14786v1)。
- S50：[Google Research big_vision](https://github.com/google-research/big_vision)，实现入口。
- A07：本卡结构伪代码。

## 可继续追问

“配对分数是否每行总和为1？”“与CLIP的损失有什么不同？”“训练有decoder是否意味着接口能生成文字？”
