# 第 6 章：GPU 和 GPU 相关的优化 — 模块 3：低精度计算与算子融合

> 📍 学习进度：第 6 章，第 3 / 5 模块
> 📅 生成时间：2026-05-05

---

## 学习目标

- 理解低精度计算为什么能提升速度（四大提速机制）
- 掌握 FP32/FP16/BF16/TF32/INT8/FP8 精度格式的差异及适用场景
- 理解混合精度训练的工作原理（autocast + GradScaler）
- 掌握算子融合的核心思想及在实际 Transformer 中的应用

---

## 核心内容

### 一、低精度计算：用精度换速度

模块 2 中我们了解了屋顶线模型——计算能力增长远快于内存带宽，GPU 严重受限于"内存墙"。低精度计算正是应对这一问题的核心手段之一：**用更少的比特表示每个数值，同时提升计算吞吐和内存带宽利用率**。

#### 1.1 常见精度格式对比

```
FP32（32位）:  1 位符号 + 8 位指数 + 23 位尾数    ← 传统精度
FP16（16位）:  1 位符号 + 5 位指数 + 10 位尾数    ← 范围小，容易溢出
BF16（16位）:  1 位符号 + 8 位指数 +  7 位尾数    ← 范围同 FP32，精度略低
TF32（19位）:  1 位符号 + 8 位指数 + 10 位尾数    ← A100 默认，内部格式
INT8（8位）:   8 位整数                            ← 纯推理用
FP8（8位）:    E4M3 或 E5M2 两种子格式            ← Hopper+ 新一代
```

各精度在 A100 上的性能对比：

| 精度类型 | 位数 | 表示范围 | 峰值算力（A100） | 相对 FP32 |
|----------|------|----------|-----------------|-----------|
| FP32 | 32 位 | 3.4×10³⁸ | 19.5 TFLOPS | 1×（基准） |
| TF32 | 19 位 | 3.4×10³⁸ | 156 TFLOPS | **8×** |
| FP16 | 16 位 | 6.5×10⁴ | 312 TFLOPS | **16×** |
| BF16 | 16 位 | 3.8×10³⁸ | 312 TFLOPS | **16×** |
| INT8 | 8 位 | 256 | 624 TOPS | **32×** |

> 🌐 **补充（Web Search）**：FP8 是当前最新一代精度格式（Hopper/Blackwell 架构原生支持），包含两种子格式：**E4M3**（4 位指数 + 3 位尾数，用于前向传播，精度更高）和 **E5M2**（5 位指数 + 2 位尾数，用于反向传播，范围更大）。H100 上 FP8 峰值约 1979 TFLOPS，是 FP16 的 2 倍。Meta Llama 3 和 DeepSeek-V3 均已采用 FP8 混合精度训练。来源：[NVIDIA H100 Tensor Core 白皮书](https://resources.nvidia.com/en-us-tensor-core)

#### 1.2 低精度提速的四大机制

为什么精度降低能让 GPU 变快？这不是单一因素，而是**四重优化叠加**：

**机制一：硬件电路简化**

```
浮点运算器复杂度 ∝ 位宽²

FP32 乘法器：面积 ≈ 0.1 mm²（假设）
FP16 乘法器：面积 ≈ 0.025 mm²（约为 FP32 的 1/4）

同样芯片面积：
  FP32 放 100 个乘法器
  FP16 放 400 个乘法器 → 吞吐量 4 倍
  INT8 放 1600 个乘法器 → 吞吐量 16 倍
```

**机制二：内存带宽节省**

同样大小的模型，精度减半意味着**搬运量减半**：

```
GPT-3 175B 参数：
  FP32: 175B × 4 字节 = 700 GB
  BF16: 175B × 2 字节 = 350 GB（减半！）

HBM2e 带宽 2 TB/s：
  FP32: 每秒加载 500 亿个参数
  BF16: 每秒加载 1000 亿个参数（翻倍！）

激活值缓存（每层）：
  FP32: 16 GB → BF16: 8 GB → 节省 50% 显存

梯度：
  FP32: 16 GB → BF16: 8 GB → 节省 50% 显存
```

这意味着：① 权重加载时间减半；② 同样显存能放更大模型；③ 缓存命中率提升（同样缓存容量可缓存更多数据）。

**机制三：Tensor Core 专用加速**

Tensor Core 不是缩小版 CUDA Core，而是**重构版矩阵引擎**：

```
普通 CUDA Core：
  每周期完成 1 次乘加（1×1）

Tensor Core（32×32 脉动阵列）：
  每周期完成 1024 次乘加（32×32）

关键：输入用低精度，累加用 FP32 保证精度

  FP16 输入 → Tensor Core → FP32 输出
  BF16 输入 → Tensor Core → FP32 输出
  FP8  输入 → Tensor Core → FP16/FP32 输出
```

> 💡 **精度-性能对照表（A100）**：
>
> | 精度 | 峰值 TFLOPS | 对比 FP32 CUDA Core |
> |------|------------|---------------------|
> | FP32（CUDA Core） | 19.5 | 1× |
> | TF32（Tensor Core） | 156 | 8× |
> | FP16/BF16（Tensor Core） | 312 | 16× |
> | INT8（Tensor Core） | 624 TOPS | 32× |

**机制四：并行度提升**

```
同样芯片面积：
  1 个 FP32 CUDA 核心 ≈ 0.1 mm²
  1 个 FP16 CUDA 核心 ≈ 0.05 mm²
  1 个 INT8 CUDA 核心 ≈ 0.025 mm²

  → 精度越低 → 电路越小 → 同面积放更多单元
  → 更多单元 → 更高并行度 → 更高吞吐量
```

**四大机制总结**：

```
低精度提速 = 电路简化（更多计算单元）
           + 带宽节省（数据搬运量减半）
           + Tensor Core（专用矩阵加速电路）
           + 并行度提升（同面积更多单元）

这四者不是"四选一"，而是同时生效、叠加放大的
```

#### 1.3 混合精度训练：低精度的正确使用方式

直接用 FP16 训练有两个严重问题：

```
问题 1: 溢出（Overflow）
  FP16 范围：6.1×10⁻⁵ ~ 65504
  某些梯度值超出这个范围 → 变成 inf 或 -inf → 训练崩溃

问题 2: 下溢出（Underflow）
  很小的梯度（如 1×10⁻⁸）→ FP16 无法表示 → 变成 0
  → 梯度消失 → 模型无法学习
```

**解决方案：混合精度训练（Mixed Precision Training）**

核心思想：**不同组件用不同精度**，兼顾速度和稳定性：

```
组件                 精度        原因
────────────────────────────────────────────
前向激活值            BF16/FP16   省显存、提速
参数（master copy）   FP32        数值稳定性
梯度                  FP32        精度敏感
Optimizer 状态        FP32        精度敏感
Softmax / LayerNorm   FP32        数值稳定
矩阵乘法              BF16/FP16   Tensor Core 加速
```

**BF16 vs FP16 的关键区别**：

```
FP16: 1 符号 + 5 指数 + 10 尾数
  范围小（6.1e-5 ~ 65504）→ 容易溢出/下溢 → 需要 GradScaler

BF16: 1 符号 + 8 指数 + 7 尾数
  范围和 FP32 一样大（1.2e-38 ~ 3.4e38）→ 几乎不会溢出
  精度略低（7 位 vs 23 位尾数）→ 但对训练影响很小

结论：BF16 是当前 LLM 训练的默认选择
  ✅ 范围大，不需要 GradScaler
  ✅ 精度足够，训练稳定
  ✅ 速度和 FP16 一样
```

**PyTorch 中的混合精度实现**：

```python
import torch

# BF16 混合精度（推荐，不需要 GradScaler）
model = MyModel().cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

for x, y in dataloader:
    # autocast 自动将 matmul/conv 降为 BF16，softmax/ln 保持 FP32
    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        out = model(x)
        loss = criterion(out, y)

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

```python
# FP16 混合精度（需要 GradScaler 防止下溢）
scaler = torch.amp.GradScaler()

for x, y in dataloader:
    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
        out = model(x)
        loss = criterion(out, y)

    # loss → loss×scale → backward → grad/scale → optimizer.step
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

> 💡 **补充（Context7 / PyTorch）**：`torch.amp.autocast` 的工作机制——模型参数保持 FP32 不变，autocast 只在算子执行时临时将输入转为低精度。具体规则：matmul / linear / conv → BF16/FP16；LayerNorm / Softmax / exp / div → FP32。梯度始终在 FP32 中累积。来源：[PyTorch AMP 文档](https://pytorch.org/docs/stable/amp.html)

---

### 二、算子融合（Operator Fusion）：消灭不必要的数据搬运

#### 2.1 算子融合的核心思想

回忆模块 2 的屋顶线模型：GPU 的计算能力远超内存带宽。当算子之间有数据依赖时，中间结果必须**写回 HBM 再读出来**——这就是性能杀手。

```
未融合的执行过程：

  x ──→ Kernel 1 (sin) ──→ HBM ──→ Kernel 2 (cos) ──→ HBM ──→ Kernel 3 (square)
         a = sin(x)    写回     b = cos(x)    写回      c = a², d = b²
                                                              ↓
                                                         HBM → Kernel 4 (add)
                                                              e = c + d → HBM

  数据在 HBM 和 SM 之间来回搬运 5 次！
```

```
融合后的执行过程：

  x ──→ Fused Kernel ──→ e = sin²(x) + cos²(x) → HBM
         a = sin(x)  → 寄存器
         b = cos(x)  → 寄存器
         c = a²      → 寄存器
         d = b²      → 寄存器
         e = c + d   → HBM（只写这一次）

  中间结果全部在寄存器/SRAM 中传递，不写回 HBM！
```

**核心原则**：如果多个连续算子操作同一份数据，与其每次都写回 HBM 再读出来，不如**在一个 Kernel 中完成所有操作**，只在最终结果时才写回。

#### 2.2 算子融合的类型

```
类型 1: 垂直融合（Vertical Fusion）
  连续的、有数据依赖的算子合并
  例: Linear → Bias → ReLU → Dropout
  融合为: FusedLinearBiasReLU

类型 2: 水平融合（Horizontal Fusion）
  无依赖但可并行的算子合并
  例: 同一输入的 Q、K、V 三个投影
  融合为: 一个大的 GEMM（三个矩阵拼接）

类型 3: 归约融合（Reduction Fusion）
  归约操作（sum/max）与其前后算子合并
  例: LayerNorm = (x - mean) / sqrt(var + eps)
  融合为: 一个 Kernel 完成 mean、var、normalize
```

#### 2.3 Transformer 中的常见算子融合

在实际 LLM 训练/推理中，以下融合模式是最关键的：

**① FlashAttention（最重量级的融合）**

```
标准 Attention（未融合）:
  S = Q @ K^T        → 写回 HBM（N×N 矩阵！）
  P = softmax(S)     → 读 HBM → 写回 HBM
  O = P @ V          → 读 HBM → 写回 HBM

  N = 4096 时，N×N 矩阵 = 16M 个元素 = 64 MB（FP32）
  → 光这个中间矩阵就吃掉大量 HBM 带宽

FlashAttention（融合）:
  Q, K, V 分块加载到 SRAM
  → 在 SRAM 中完成 Q@K^T、softmax、P@V
  → 只输出最终 O 到 HBM
  → 中间矩阵 N×N 永远不写回 HBM
```

**② Fused SiLU/GELU + Linear（SwiGLU）**

```python
# 未融合（2 次 HBM 往返）
gate = Linear1(x)        # x → HBM → Linear1 → HBM
act = SiLU(gate)         # HBM → SiLU → HBM
up = Linear2(x)          # HBM → Linear2 → HBM
out = act * up           # HBM → multiply → HBM

# 融合（减少 HBM 往返）
FusedSwiGLU(x)           # x → Linear1/SiLU/Linear2/multiply → HBM（1 次）
```

**③ Fused LayerNorm + Linear**

```python
# 未融合
x = LayerNorm(x)         # HBM 读 → mean/var/normalize → HBM 写
x = Linear(x)            # HBM 读 → matmul → HBM 写

# 融合
FusedLNLinear(x)         # HBM 读 → LN → 直接传给 matmul → HBM 写（1 次）
```

**④ Fused CrossEntropy**

```python
# 未融合
logits = model(x)        # shape: [batch, vocab_size]，vocab_size = 128K
                         # 这个张量要写回 HBM：128K × 4 bytes = 512KB/token
probs = softmax(logits)  # 又一次 HBM 往返
loss = -log(probs[label])

# 融合（关键：不物化完整的 logits 张量）
FusedCrossEntropy(x, labels)
  # 在 Kernel 内部计算 logits → 直接计算 loss
  # logits 永远不写回 HBM
  # 节省 vocab_size × 4 bytes/token 的显存
```

> 🌐 **补充（Web Search）**：torch.compile（PyTorch 2.x）可以自动实现算子融合——通过 TorchInductor 将计算图中的 pointwise 和 reduction 算子融合，生成 Triton 代码。实际应用中，`torch.compile(mode="reduce-overhead")` 可带来 10-30% 的自动加速。来源：[PyTorch 2.0 Compiler Blog](https://pytorch.org/get-started/pytorch-2.0/)

#### 2.4 算子融合 vs 内存墙

将算子融合放在屋顶线模型中理解：

```
未融合：
  每个 Kernel 独立运行
  Kernel 1: 计算 10μs，等待数据 50μs → 利用率 17%
  Kernel 2: 计算 10μs，等待数据 50μs → 利用率 17%

融合后：
  一个 Kernel 完成所有计算
  Fused Kernel: 计算 20μs，等待数据 50μs → 利用率 29%

  → 不是"计算更快"，而是"搬数据次数更少"
  → 操作强度提高（FLOPs 不变，但 Byte 减少了）
  → 从屋顶线模型左侧（内存受限）向右推移
```

**本质**：算子融合**不是减少计算量**，而是**减少数据搬运量**——分母（Byte）变小了，操作强度（FLOPs/Byte）变大了，程序更接近计算受限区。

---

## 🧠 本模块问题

请在下方回答以下问题后，输入 `提交作业` 提交。

**Q1**：低精度计算为什么能让 GPU 更快？请列出四大提速机制，并各用一句话说明原理。

**Q2**：什么是混合精度训练？为什么 FP16 训练需要 GradScaler 而 BF16 不需要？请用"溢出"和"下溢出"的概念解释。

**Q3**：算子融合的核心思想是什么？请用一个具体的 Transformer 中的例子（如 Fused SiLU + Linear 或 FlashAttention），说明融合前后的数据搬运过程有什么变化。

---

<!-- 学习者作答区（请在此处填写你的答案） -->

**A1**：

1. 低精度占用的 计算资源 更少，比如使用 FP32 乘法的量是 BF16 乘法量的 1/2
2. 低精度占用的 存储资源 更少，在数据搬运时的带宽占用更少
3. 低精度占用的 存储资源 更少，同样大小缓存可以存储的数据量更多，缓存命中率更高
4. 低精度可以用 Tensor Core 加速，每次运算能进行的 matrix multiply 操作更多


**A2**：

混合精度训练是指在 乘法/加法/卷积 等算子下，用低精度数据类型加速运算（输出保持数据类型不变），而其他 kernel 精度保持不变，加速训练流程。

FP16 需要 GradScaler 是因为 FP16 数据范围相较 FP32 指数位从 8 缩小到 5，范围变小，更容易下溢出（指数为 5，实际 FP16 的范围大概是 6e-5 ~ 65504，因此 小于 6e-5 和 大于 65504 都会发生下溢 和 溢出，所以需要 scale 来调整范围），而 BF16 指数位不变，尾数位置变小，因此整体范围不变，精度下降，所以不需要 GradScaler。




**A3**：
算子融合的思想是根据屋顶线将内存墙右移，即让多个 计算（pointwise、reduction）kernel 融合成一个，减少 HBM 搬运次数。
举例 Fused Silu + Linear

未融合：
  a = linear(x)   x -> HBM -> Linear -> HBM
  a_silu = silu(a)   HBM -> silu -> HBM
  b = linear(x)  HBM -> linear -> HBM
  out = a_silu * b  HBM -> multiply -> HBM

融合：
  FusedSilu(x) x-> HBM -> kernel -> HBM


---

<!-- 教师批改区（提交作业后由导师填写，请勿手动修改） -->

### 📝 批改结果

**Q1 批改**：列出了 4 个点，但有几处理解偏差：① "FP32 乘法量是 BF16 的 1/2"不准确，机制一是**硬件电路简化**（浮点运算器复杂度 ∝ 位宽²，FP16 乘法器面积仅为 FP32 的 1/4，同面积可放 4 倍乘法器）；② 第 2 和第 3 点本质都是内存带宽节省，拆成了两条但缺少机制一（电路简化）；③ Tensor Core 的描述不够精确——不是"每次运算能进行更多矩阵乘法"，而是脉动阵列每周期完成 1024 次 MAC（32×32）。四大机制的名称和逻辑关系不够清晰。 — 得分：**6/10**

<details>
<summary>📖 Q1 参考答案</summary>

低精度计算的四大提速机制：

**机制一：硬件电路简化** — 浮点运算器复杂度与位宽² 成正比，FP16 乘法器面积仅为 FP32 的 1/4，同芯片面积可集成 4 倍数量的乘法器，吞吐量相应提升。

**机制二：内存带宽节省** — 精度减半意味着每个数据的存储和搬运量减半（如 GPT-3 175B 从 FP32 的 700GB 降到 BF16 的 350GB），同一带宽下可传输的数据量翻倍，权重加载时间减半，显存可容纳更大模型。

**机制三：Tensor Core 专用加速** — Tensor Core 不是缩小版 CUDA Core，而是专为低精度矩阵乘设计的脉动阵列电路。A100 的 32×32 脉动阵列每周期完成 1024 次乘加（CUDA Core 只有 1 次），且输入用低精度、累加用 FP32 保证精度，实现 FP16/BF16 下 16 倍于 FP32 的算力。

**机制四：并行度提升** — 低精度电路更小（FP16 ≈ FP32 面积的 1/4，INT8 ≈ 1/16），同样芯片面积可容纳更多计算单元，并行度提升直接转化为吞吐量提升。

四者同时生效、叠加放大，而非"四选一"。

</details>

---

**Q2 批改**：混合精度训练的理解基本正确，FP16 vs BF16 的区别解释到位——FP16 指数位从 8 缩到 5，范围变小需要 GradScaler；BF16 指数位不变，范围同 FP32。但有两个小问题：① "输出保持数据类型不变"不够精确——autocast 对不同算子有不同的精度策略（matmul→低精度，softmax/LayerNorm→FP32），master 参数始终是 FP32；② "大于 65504 都会发生下溢"——大于 65504 是溢出（overflow），不是下溢出，两者是相反的概念。 — 得分：**8/10**

<details>
<summary>📖 Q2 参考答案</summary>

**混合精度训练**：在训练过程中，不同组件使用不同精度——前向激活值和矩阵乘法使用 BF16/FP16（提速、省显存），参数的 master copy、梯度、Optimizer 状态保持 FP32（数值稳定性），Softmax/LayerNorm 等数值敏感操作保持 FP32。PyTorch 中通过 `torch.amp.autocast` 自动实现精度切换。

**FP16 需要 GradScaler**：
- FP16 范围极窄（6.1×10⁻⁵ ~ 65504）
- **下溢出（Underflow）**：很小的梯度（如 1×10⁻⁸）无法表示 → 变成 0 → 梯度消失 → 模型无法学习
- **溢出（Overflow）**：较大的值超出 65504 → 变成 inf → 训练崩溃
- GradScaler 原理：loss×scale（如 65536）→ backward 时梯度也被放大 → 避免下溢 → 最后 grad/scale 恢复真实值

**BF16 不需要 GradScaler**：
- BF16 指数位和 FP32 一样是 8 位，范围完全相同（1.2×10⁻³⁸ ~ 3.4×10³⁸）
- 任何 FP32 能表示的梯度值，BF16 都能表示（只是尾数从 23 位降到 7 位，精度略低）
- 不存在下溢出和溢出问题 → 不需要 GradScaler
- 这就是 BF16 成为当前 LLM 训练默认选择的核心原因

</details>

---

**Q3 批改**：算子融合的核心思想（减少 HBM 搬运、操作强度右移）理解正确。SwiGLU 的例子大体正确，但有两处小问题：① 未融合版本中写的是"b = linear(x)"，实际上 SwiGLU 的第二路是 up projection（也是 Linear），但代码表述不够清晰，区分了 gate 和 up 两条线但没有明确是两个不同的 Linear 层；② 融合版本写得过于简略，没有说明中间结果在片上寄存器/SRAM 中传递。不过核心论点（减少 HBM 往返）是正确的。 — 得分：**7/10**

<details>
<summary>📖 Q3 参考答案</summary>

**算子融合的核心思想**：多个连续算子操作同一份数据时，与其每次都将中间结果写回 HBM 再读出来，不如在一个 Kernel 中完成所有操作，中间结果在寄存器/SRAM 中直接传递，只在最终结果时写回 HBM。本质是**减少数据搬运量**（分母 Byte 变小），提高操作强度（FLOPs/Byte 变大），将程序从内存受限区推向计算受限区。

**以 Fused SwiGLU 为例**：

未融合（4 次 HBM 往返）：
```
gate = Linear1(x)        # x 从 HBM 读出 → matmul → gate 写回 HBM
up   = Linear2(x)        # x 从 HBM 读出 → matmul → up 写回 HBM
act  = SiLU(gate)        # gate 从 HBM 读出 → SiLU → act 写回 HBM
out  = act * up          # act 和 up 从 HBM 读出 → 乘法 → out 写回 HBM

总共：5 次 HBM 读 + 4 次 HBM 写
```

融合（1 次 HBM 读 + 1 次 HBM 写）：
```
FusedSwiGLU(x):
  x 从 HBM 读出一次
  gate = Linear1(x) → 保持在寄存器
  up   = Linear2(x) → 保持在寄存器
  act  = SiLU(gate) → 保持在寄存器
  out  = act * up   → 写回 HBM（仅此一次）

总共：1 次 HBM 读 + 1 次 HBM 写
HBM 访问量减少约 80%
```

从屋顶线模型看：计算量（FLOPs）不变，数据搬运量（Byte）大幅减少 → 操作强度提高 → 更接近计算受限区。

</details>

---

**综合评价**：平均 7/10。核心概念（混合精度原理、算子融合思想）掌握较好。薄弱点：① 四大机制的区分和命名不够准确，容易把两个机制混为一条；② 溢出/下溢出的术语偶尔混用；③ 算子融合的例子中 HBM 往返次数的细节不够精确。建议重读"低精度提速的四大机制"一节，重点关注每种机制"加速的到底是什么"（电路面积、带宽、专用硬件、并行度）。

**批改时间**：2026-05-05
