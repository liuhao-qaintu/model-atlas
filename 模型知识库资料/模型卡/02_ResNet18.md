# ResNet-18

模型 ID：resnet-18。类型：ResNet 系列具体变体。整理日期：2026-10-12。

本卡以 Torchvision v0.21.0 的实现说明结构和输出，与本机验证版本一致；原始论文用于机制背景。框架实现、预训练设置与教学示例分别标注。

## 概览与核心机制

ResNet 的残差学习背景见 [S06：原论文](https://arxiv.org/abs/1512.03385)。本卡解释其基础块与框架前向过程，不整理实验性能数值。

Torchvision 的 ResNet-18 使用 BasicBlock，四个阶段各含两个块；块包含二维卷积和残差分支。对应源码定位为 `BasicBlock`、`resnet18`、`ResNet._forward_impl`。[S08：版本源码](https://github.com/pytorch/vision/blob/v0.21.0/torchvision/models/resnet.py)

块的机制可写为：

$$
y=\operatorname{ReLU}(F(x)+S(x))
$$

其中 F 是残差分支，S 在形状匹配时可为恒等映射；形状变化时需要匹配分支。相加要求形状兼容，不能直接把任意两层输出相加。

实现细节：BasicBlock 的 F 使用两个 3×3 卷积；S 需要对齐时采用 1×1 卷积与归一化。默认 layer1 不降采样，layer2–4 的首块 stride=2。相加后还有 ReLU，不能直接把上式改成 F=y−x。定位：`BasicBlock`、`_make_layer`、`ResNet.__init__`。[S08：版本源码](https://github.com/pytorch/vision/blob/v0.21.0/torchvision/models/resnet.py)

## 结构与原理

### 按BasicBlock.forward逐步阅读

固定Torchvision v0.21.0，BasicBlock主分支为 `conv1 → bn1 → relu → conv2 → bn2`；shortcut先保存输入x，需要时执行downsample。两路相加后再次ReLU。F指上述主分支在相加前的结果，S指shortcut；不是把整个块输出直接等同于F。[官方BasicBlock与_make_layer](https://github.com/pytorch/vision/blob/v0.21.0/torchvision/models/resnet.py)

例如layer2首块接收 `[B,64,56,56]`。主分支首个3×3卷积以stride=2、padding=1得到 `[B,128,28,28]`，第二个3×3卷积保持该形状；shortcut的downsample为 `1×1 Conv2d(stride=2) → BatchNorm2d`，输出同为 `[B,128,28,28]`。不能把整个downsample模块只称为一个卷积。主分支和shortcut各有一条数据流，downsample并不是把主分支再重复算一次。

| 检查点 | 主分支 | shortcut |
|---|---|---|
| 块输入 | `[B,64,56,56]` | 保存同一输入 |
| 主分支完成 | `[B,128,28,28]` | 原输入仍为 `[B,64,56,56]` |
| shortcut对齐后 | 不变 | `[B,128,28,28]` |
| 相加与激活 | 两路逐元素相加，再ReLU | 不额外产生类别输出 |

这些数字由该版本layer2首块配置与给定输入推导。不能只说“残差两路相加”而跳过通道和空间尺寸对齐。

### shortcut为什么有帮助，又不能保证什么

先看不含末端激活的加法 `u=F(x)+x`：对可微位置作局部分析，其Jacobian为 `J_u=J_F+I`，恒等支路给梯度提供一个直接项。实际本块输出还有 `y=ReLU(u)`，因此非零可微位置的局部梯度还要乘ReLU门；有投影的shortcut应使用J_S替代I。这是解释梯度路径的教学推导，不是“梯度永远不会消失”的证明。ReLU在0处不可微，其实现采用特定次梯度约定。

即使先不考虑ReLU，`J_F+I`也没有“至少保留一份非零梯度”的保证：两项可能抵消。例如教学加法分支取 `F(x)=-x+2`，则 `u=F(x)+x=2`、`J_F=-I`，所以 `J_u=0`；此时ReLU处于正区间，其导数为1，梯度仍为0。这个反例排除了“只是ReLU关门”的解释。它是加法机制的数学反例，不声称给定BasicBlock权重或训练过程必然出现该分支。

若相加前主分支 `F(x)=0`，一般结论是 `y=ReLU(S(x))`。只有恒等shortcut，即 `S(x)=x`、downsample为None时，才化为 `y=ReLU(x)`；这时只有x非负的部分才与x相同。layer2首块使用投影，不能套用该恒等条件。负输入不会因为存在shortcut就被原样保留。以下用不带downsample的官方BasicBlock展示恒等支路的反例，不是从零实现整个ResNet：

```python
import torch
from torchvision.models.resnet import BasicBlock

zero_branch = BasicBlock(4, 4).eval()
with torch.no_grad():
    zero_branch.conv1.weight.zero_()
    zero_branch.conv2.weight.zero_()
    signed_input = torch.linspace(-1, 1, 4 * 3 * 3).reshape(1, 4, 3, 3)
    residual_output = zero_branch(signed_input)
assert torch.allclose(residual_output, signed_input.relu())
assert not torch.allclose(residual_output, signed_input)

# 独立的教学加法反例：正激活区间也可能发生两路Jacobian抵消。
cancel_input = torch.tensor([0.5, 1.0], requires_grad=True)
cancel_branch = -cancel_input + 2
cancel_output = torch.relu(cancel_branch + cancel_input)
assert torch.equal(cancel_output, torch.tensor([2.0, 2.0]))
cancel_output.sum().backward()
assert torch.equal(cancel_input.grad, torch.zeros_like(cancel_input))
```

形状支持广播是张量加法的通用行为，不能据此判断残差设计正确；该BasicBlock预期两路表示对齐。共享“残差相加”也不代表Transformer具有同样的卷积分支、归一化顺序或权重。

## 输入输出与数据流

### 条件与易错点

本节的 `[1,512,7,7]` 是批量为1、输入为224×224时的演示形状，不是所有输入的固定规格。一般可写作 `[B,512,H′,W′]`，空间尺寸随输入及下采样变化；全局平均池化后为 `[B,512,1,1]`，展平为 `[B,512]`。默认1000类的输出与512维特征不同，修改num_classes时输出类别数随之改变。

以下形状对应本卡指定的 RGB 输入和原始分类头。尺寸不同或修改分类头后应重新说明。

| 步骤 | 示例形状 |
|---|---|
| 输入 RGB 图像张量 | `[1, 3, 224, 224]` |
| 初始卷积、BN、ReLU | `[1, 64, 112, 112]` |
| MaxPool | `[1, 64, 56, 56]` |
| layer1 | `[1, 64, 56, 56]` |
| layer2 | `[1, 128, 28, 28]` |
| layer3 | `[1, 256, 14, 14]` |
| layer4 | `[1, 512, 7, 7]` |
| 平均池化与展平 | `[1, 512]` |
| 默认分类头 | `[1, 1000]` |

默认分类输出是类别分数；去掉分类头后可返回 512 维特征。224×224 是本卡选用的示例尺寸，不能概括为所有用法都只接受这个尺寸。

S07 的 ImageNet 预训练权重提供相应预处理：缩放、裁剪及归一化。实际使用预训练权重时应采用其配套变换；本卡随机张量只用于形状检查。[S07：权重与预处理说明](https://docs.pytorch.org/vision/0.21/models/generated/torchvision.models.resnet18.html)

## 最小调用代码

原创调用示例。`weights=None` 不下载权重，没有训练；输出数值不用于真实分类。

```python
import torch
from torch import nn
from torchvision.models import resnet18


torch.manual_seed(7)
model = resnet18(weights=None).eval()
x = torch.randn(1, 3, 224, 224)
with torch.no_grad():
    logits = model(x)
assert logits.shape == (1, 1000)

model.fc = nn.Identity()
with torch.no_grad():
    features = model(x)
assert features.shape == (1, 512)
print("logits:", tuple(logits.shape))
print("features:", tuple(features.shape))
```

这段展示的是框架模型调用与输出切换，完整架构定义通过 S08 查阅，不把十几行调用代码标成从零实现。

## 推导与排错

### 改成4类，究竟改了什么

在该实现中，avgpool聚合空间轴，flatten得到 `[B,512]`，fc把它映射为类别分数。将fc从Linear(512,1000)换成Linear(512,4)，只把输出接口改为 `[B,4]`；新fc参数量为 `512×4+4=2052`，不改变512维骨干表示。将fc改为Identity则直接返回 `[B,512]` 特征，并不是一个512类分类器。[ResNet._forward_impl与__init__](https://github.com/pytorch/vision/blob/v0.21.0/torchvision/models/resnet.py)

互斥多类任务常在推理时对类别分数沿类别轴做softmax；训练交叉熵可直接接收logits。多标签、回归等任务有不同的头和损失，不能声称所有分类都必须使用softmax。特征即使归一化或偶然和为1，也不自动具有类别标签语义。

### 为什么冻结参数后BatchNorm还在变化

`requires_grad=False`控制可学习参数是否参与梯度计算；BatchNorm的running_mean、running_var是buffer，不是靠优化器梯度更新的参数。在train模式且开启running统计时，它们仍会随前向数据更新。eval模式改变这类模块的前向行为，但不会自动关闭autograd。[PyTorch 2.6梯度模式与eval](https://docs.pytorch.org/docs/2.6/notes/autograd.html#evaluation-mode-nn-module-eval)

`torch.no_grad()`关闭该上下文内的梯度记录，不会把BN切为eval，也不会阻止train模式下正常更新running统计；仅使用no_grad不会因此令BN报错。下面额外在no_grad中执行一次前向，核对这两个控制维度彼此独立。

**普通梯度模式也不保证输出有计算图。** 对本例中由Linear或BN实际运算得到的输出，须有参与运算的输入或参数需要梯度，autograd才记录相应路径。全部参数冻结且普通输入requires_grad=False时，即使不使用no_grad，输出也不需要梯度；直接对该无梯度输出的和backward会报RuntimeError，不能说“通常静默无效”。若输入requires_grad=True，普通模式可记录到输入的梯度路径，但冻结参数仍不累积梯度；no_grad内执行这些运算则不记录该路径。eval只切模块模式，不关闭上述梯度记录。

| 参数全部冻结 | 输入需要梯度 | 梯度模式 | 本例计算输出需要梯度 |
|---|---|---|---|
| 是 | 否 | 普通模式 | 否 |
| 是 | 是 | 普通模式 | 是，梯度可回到输入 |
| 是 | 是或否 | no_grad | 否 |

BN统计的更新机制与输出计算图是两回事。训练模式还需track_running_stats=True才跟踪running统计；一次更新后的数值也可能与原值相同，例如batch均值恰与running_mean一致。track_running_stats=False是不用这些running统计，不是“冻结并继续使用已有统计”的替代方案。前向能否运行还取决于输入是否合法，不能泛化为任何输入都不报错。

如果你的策略是“固定骨干及其BN统计，仅训练新头”，应分别设置参数梯度、优化器参数和模块模式；对全模型调用train()之后，还需重新将要固定统计的BN设为eval。是否固定BN应由数据规模和迁移策略决定，不能把一种策略称为全部微调的唯一做法。下面只用小BN模块验证状态变化，不进行任务训练。

```python
from torch import nn
from torchvision.models import resnet18

four_class_model = resnet18(weights=None).eval()
four_class_model.fc = nn.Linear(four_class_model.fc.in_features, 4)
assert sum(p.numel() for p in four_class_model.fc.parameters()) == 2052
with torch.no_grad():
    four_scores = four_class_model(torch.randn(2, 3, 64, 64))
assert four_scores.shape == (2, 4)
stage_probe = four_class_model.layer2[0]
with torch.no_grad():
    stage_result = stage_probe(torch.randn(2, 64, 56, 56))
assert stage_result.shape == (2, 128, 28, 28)

bn_probe = nn.BatchNorm2d(4)
for parameter in bn_probe.parameters():
    parameter.requires_grad_(False)
bn_probe.train()
before_train = bn_probe.running_mean.clone()
bn_probe(torch.ones(2, 4, 3, 3) * 5)
assert not torch.equal(before_train, bn_probe.running_mean)
before_no_grad = bn_probe.running_mean.clone()
with torch.no_grad():
    bn_probe(torch.ones(2, 4, 3, 3) * 7)
assert not torch.equal(before_no_grad, bn_probe.running_mean)
bn_probe.eval()
before_eval = bn_probe.running_mean.clone()
bn_probe(torch.ones(2, 4, 3, 3) * 9)
assert torch.equal(before_eval, bn_probe.running_mean)

# 参数冻结后，普通模式是否记录图还取决于输入。
frozen_linear = nn.Linear(2, 2)
for parameter in frozen_linear.parameters():
    parameter.requires_grad_(False)
with torch.no_grad():
    frozen_linear.weight.fill_(1)
    frozen_linear.bias.zero_()
plain_output = frozen_linear(torch.ones(1, 2))
assert not plain_output.requires_grad and plain_output.grad_fn is None
backward_rejected = False
try:
    plain_output.sum().backward()
except RuntimeError:
    backward_rejected = True
assert backward_rejected
input_leaf = torch.ones(1, 2, requires_grad=True)
frozen_linear.eval()  # eval不关闭autograd。
tracked_output = frozen_linear(input_leaf)
assert tracked_output.requires_grad and tracked_output.grad_fn is not None
tracked_output.sum().backward()
assert torch.equal(input_leaf.grad, torch.full_like(input_leaf, 2))
assert all(parameter.grad is None for parameter in frozen_linear.parameters())
with torch.no_grad():
    untracked_output = frozen_linear(input_leaf)
assert not untracked_output.requires_grad and untracked_output.grad_fn is None
```

### 使用时先核对这三件事

- 你要空间特征图还是全局特征？layer4的 `[B,512,H′,W′]` 仍有位置轴；Identity替换fc后已经经过avgpool与flatten，只剩 `[B,512]`，不能再假装含有7×7空间布局。
- 新任务的数据是否还是RGB图像？把传感器信号直接reshape成RGB图像，不会自动获得合理语义；修改输入通道也涉及第一层及预处理与权重兼容性，不只是改最后一层。
- 你是否真的加载并训练了权重？weights=None是随机初始化；接口能前向、形状正确和参数被冻结，都不代表具备新任务预测能力。本节不报告任务准确率。

## 模型联系与应用记录

原论文提出残差学习框架，本卡将其归为 ResNet 系列；ResNet-18 是具体变体。此处只使用论文摘要作为背景依据，不据此推断其他论文采用了 ResNet-18。[S06：摘要](https://arxiv.org/abs/1512.03385)

固定版本的分类实现使用 `nn.AdaptiveAvgPool2d((1, 1))` 聚合空间维度，随后展平并进入 `nn.Linear(512 * block.expansion, num_classes)`。BasicBlock 的 expansion=1，因此 ResNet-18 分类头输入为 512 维；默认类别数为 1000。定位：`ResNet.__init__`、`ResNet._forward_impl`。[S08：v0.21.0 源码](https://github.com/pytorch/vision/blob/v0.21.0/torchvision/models/resnet.py)

和 [E01：TinySignalCNN 教学实例](/models/cnn-1d?section=code)比较，两者都采用“全局平均聚合 → 展平 → Linear 分类头”的结构。TinySignalCNN 使用 AdaptiveAvgPool1d(1) 与 32 维分类头输入；这里的 ResNet-18 使用 AdaptiveAvgPool2d((1, 1)) 与 512 维分类头输入。相似性指这些模块的组织方式，不能由此推出整个架构、权重或任务相同，也不能推广到所有 1D CNN。

- 结构组成关系：ResNet-18 → BasicBlock → Conv2d / 残差相加等。
- 与普通 CNN 的共同点和差异，应分别说明共享算子与额外连接结构。
- 不推断 ResNet-18 必然优于其他模型，也不把更换输出头等同于完成新任务训练。
- 应用方向与具体变体请核对 [ResNet 原论文](https://arxiv.org/abs/1512.03385)。

## 图示与版本

当前尚未绑定图片。后续可核对原论文残差块或整体结构图，区分通用 ResNet 示意与具体 ResNet-18。源码固定为 v0.21.0，避免 main 分支更新改变定位。

## 来源与定位

- S06：[Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)，论文背景与图像识别任务；本轮重点核对论文元信息和摘要，完整方法图待进一步整理。
- S07：[Torchvision 0.21 resnet18](https://docs.pytorch.org/vision/0.21/models/generated/torchvision.models.resnet18.html)，参数与权重变换。
- S08：[Torchvision v0.21.0 resnet.py](https://github.com/pytorch/vision/blob/v0.21.0/torchvision/models/resnet.py)，`BasicBlock`、`resnet18`、`ResNet._forward_impl`。
- E02：本卡原创调用与形状表，已在 验证环境 的 PyTorch 2.6.0+cpu、Torchvision 0.21.0+cpu 中通过前向核查，代码示例可在相应PyTorch环境中核对。参考源码固定为 v0.21.0，运行版本单独记录。

## 可继续追问

“残差相加要求什么形状？”“分类分数与特征有什么区别？”“换成四个类别需要改哪里？”“代码修改后是否已经具备故障诊断能力？”
