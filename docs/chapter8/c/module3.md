# 第 8 章：分布式训练 — 模块 3：流水线并行

> 📍 学习进度：第 8 章，第 3 / 5 模块
> 📅 生成时间：2026-05-08

---

## 学习目标

- 理解朴素层并行的流水线气泡问题
- 掌握微批次（Micro-batch）如何缓解气泡
- 了解不同的流水线调度策略（GPipe、1F1B、Interleaved 1F1B）
- 理解零气泡流水线的原理

---

## 核心内容

### 一、朴素的层并行

#### 1.1 基本思路

```
朴素层并行（Layer Parallelism）：
  将模型按层切分，每个 GPU 负责一部分层

  GPU 0: [Layer 1, Layer 2]
  GPU 1: [Layer 3, Layer 4]
  GPU 2: [Layer 5, Layer 6]
  GPU 3: [Layer 7, Layer 8]

  前向传播：数据从 GPU 0 → 1 → 2 → 3 依次传递
  反向传播：梯度从 GPU 3 → 2 → 1 → 0 依次传递
```

#### 1.2 流水线气泡（Pipeline Bubble）

```
朴素实现的时间线（4 个 GPU，1 个批次）：

  前向：GPU 0 → 1 → 2 → 3 依次传递
  反向：GPU 3 → 2 → 1 → 0 依次传递（严格串行，不能重叠）

  时间   1      2      3      4      5      6      7      8
  GPU 0: [ F ]  ████   ████   ████   ████   ████   ████   [ B ]
  GPU 1: ████   [ F ]  ████   ████   ████   ████   [ B ]  ████
  GPU 2: ████   ████   [ F ]  ████   ████   [ B ]  ████   ████
  GPU 3: ████   ████   ████   [ F ]  [ B ]  ████   ████   ████

  F = Forward, B = Backward, █ = 空闲（气泡！）

  反向依赖链：GPU3[B](T5) → GPU2[B](T6) → GPU1[B](T7) → GPU0[B](T8)
  GPU2 必须等 GPU3 [B] 完成收到梯度后才能开始自己的 [B]

  问题：GPU 0 在 GPU 1~3 计算时完全空闲！
  利用率：每个 GPU 工作 2 步 / 总时间 8 步 = 2/8 = 25%
```

![朴素层并行](<../images/8-18-逐层并行.png>)

![层并行的流水线气泡问题](<../images/8-19-层状并行的问题.png>)

> 💡 **气泡的本质**：GPU 0 传完激活值后必须等待所有下游 GPU 计算完才能开始下一次前向。这种等待时间就是"气泡"。

---

### 二、微批次缓解气泡

#### 2.1 核心思想

```
微批次（Micro-batch）：
  不是把整个批次当 1 个任务，而是拆成 K 个小批次
  每个 GPU 处理完一个微批次后，立即将激活值传给下一个 GPU
  → GPU 0 不需要等所有层都算完，而是"流水线式"地处理

  大批次 = 1 个任务：  GPU 0 等待时间长
  拆成 4 个微批次：   GPU 0 处理完微批次 1 后立即开始微批次 2
                      → 空闲时间大幅减少

变量定义：
  N = 流水线阶段数 = GPU 数量
  K = 微批次数量

注意：层分配没有变化！GPU 0 仍然负责 Layer 1-2，GPU 1 仍然负责 Layer 3-4。
变化的是执行节奏：每个 GPU 把"一次算完整批"改为"多次算小块"。

朴素方式（1 个大批次）：
  GPU0 处理完整批 → 传给 GPU1 → GPU0 空闲等整条流水线跑完

微批次方式（K=4）：
  GPU0 处理 m1 → 立即传给 GPU1 → GPU0 开始处理 m2
  GPU1 收到 m1 → 开始算 m1 → 传给 GPU2 → ...
  多个 GPU 同时工作在不同微批次上！

对比 T=4 时刻：
  朴素：GPU0/1/2 空闲，只有 GPU3 在算
  微批次：GPU0 算 m4，GPU1 算 m3，GPU2 算 m2，GPU3 算 m1 → 4 GPU 全部工作

为什么不违反"GPU1 必须等 GPU0"的约束：
  约束仍成立——GPU1 确实必须等 GPU0 传完才能开始
  但朴素方式等的是"完整批"，微批次方式等的是"一小块"
  → 等待碎片化了，不再是一大段空闲
```

#### 2.2 GPipe 调度

```
GPipe 调度（Google, 2019）：
  将 mini-batch 切成 M 个 micro-batch。
  先让所有 micro-batch 完成前向流水线，再让所有 micro-batch 完成反向流水线，
  最后累积梯度并同步更新参数。

  变量：P = pipeline stages = GPU 数量，M = micro-batch 数量

  4 GPU，4 micro-batch 的时间线：

  时间:  01 02 03 04 05 06 07 08 09 10 11 12 13 14
  GPU0: F1 F2 F3 F4 -- -- -- -- -- -- B4 B3 B2 B1
  GPU1: -- F1 F2 F3 F4 -- -- -- -- B4 B3 B2 B1 --
  GPU2: -- -- F1 F2 F3 F4 -- -- B4 B3 B2 B1 -- --
  GPU3: -- -- -- F1 F2 F3 F4 B4 B3 B2 B1 -- -- --

  气泡率（按实际总流水时间近似）：
    bubble ≈ (P - 1) / (M + P - 1)
  注：有些资料把气泡时间除以理想计算时间，写作 (P - 1) / M。
      本文使用的是 气泡时间 / 实际总时间，因此分母多了 P - 1。
  4 GPU，4 micro-batch：3/7 ≈ 42.9%
  4 GPU，16 micro-batch：3/19 ≈ 15.8%
  4 GPU，32 micro-batch：3/35 ≈ 8.6%

  优点：实现简单；同步梯度更新，无权重陈旧问题
  缺点：all-forward-then-all-backward 需为所有 micro-batch 保留反向所需信息，
       内存占用较高（可用 rematerialization 降低激活内存，但仍高于 1F1B）
```

#### 2.3 1F1B 调度（One Forward One Backward）

```
1F1B = One Forward One Backward：
  常见于 PipeDream-Flush、Megatron-LM 等流水线并行实现。
  与原始异步 PipeDream 不同，同步 1F1B 会在一个 global batch 结束后统一更新参数。

  流水线填满后，每个 GPU 在稳定阶段交替执行：
    做一个较新的 micro-batch 的 Forward
    做一个较旧的 micro-batch 的 Backward
  这样可以更早释放 activation，显著降低显存占用。

  调度三阶段：
    1. Warmup：先做若干个 Forward，把流水线填起来
    2. Steady：进入 1F1B，Forward / Backward 交替
    3. Drain：所有 Forward 发完后，把剩余 Backward 做完

  4 GPU / 8 micro-batch 的时间线（Fi = 第 i 个 micro-batch 前向，Bi = 反向）：

  时间 -> 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22
  GPU0: F1 F2 F3 F4 F5 F6 F7 B1 F8 B2 -- B3 -- B4 -- B5 -- B6 -- B7 -- B8
  GPU1: -- F1 F2 F3 F4 F5 B1 F6 B2 F7 B3 F8 B4 -- B5 -- B6 -- B7 -- B8 --
  GPU2: -- -- F1 F2 F3 B1 F4 B2 F5 B3 F6 B4 F7 B5 F8 B6 -- B7 -- B8 -- --
  GPU3: -- -- -- F1 B1 F2 B2 F3 B3 F4 B4 F5 B5 F6 B6 F7 B7 F8 B8 -- -- --

  越靠后的 GPU 越早进入反向，越靠前的 GPU 需要先多做一些前向来填满流水线。

  反向依赖链（与 GPipe 相同）：
    B_k at GPU{i} 需要 B_k at GPU{i+1} 先完成
    GPU3 B_k → GPU2 B_k → GPU1 B_k → GPU0 B_k（级联）

  与 GPipe 对比：
    GPipe：  F F F F F F F F → B B B B B B B B（峰值激活内存随 M 增长）
    1F1B：   warmup 后 F/B 交替（峰值激活内存主要由流水线深度 P 决定）
    气泡率：  同阶，都是 (P-1)/(M+P-1)
    内存：    1F1B 明显更优

  重要区分：
    同步 1F1B / PipeDream-Flush（Megatron-LM 常用）：每个 global batch 结束后统一更新参数，无权重陈旧问题
    异步 PipeDream：可能有权重版本不一致问题，需要 weight stashing
```

![流水线并行架构](<../images/8-20-流水线架构.png>)

#### 2.4 微批次数量的影响

```
批次大小 vs 利用率（本文气泡率 = 气泡时间 / 实际总时间 = (P-1)/(M+P-1)，P=4 GPU）：

  微批次数量 M    气泡率（4 GPU）    内存开销
  ──────────────────────────────────────────
  M = 1          75%               最低
  M = 4          42.9%             中等
  M = 16         15.8%             较高
  M = 64         4.5%              高

  规律：M 越大气泡越少，但通信频率越高
  → 存在边际效益：M 超过某个值后，气泡减少收益递减
```

![批次尺寸与 GPU 利用率关系](<../images/8-21-批次尺寸和利用率关系.png>)

![批次大小的边际效益](<../images/8-17-批次大小存在边际效益.png>)

---

### 三、更多流水线调度策略

#### 3.1 Interleaved 1F1B

```
Interleaved 1F1B（又称 VPP，Virtual Pipeline Parallelism，NVIDIA Megatron-LM）：
  核心思想：让每个物理 GPU 负责多组非连续的模型阶段（虚拟阶段），
  打破"一个设备绑定一组连续层"的限制。

  设备与阶段分配对比：

  普通 1F1B（一对一连续绑定）：
    GPU 0: [Layer 1-2]
    GPU 1: [Layer 3-4]
    GPU 2: [Layer 5-6]
    GPU 3: [Layer 7-8]

  Interleaved 1F1B（一对多不连续绑定）：
    GPU 0: [Layer 1-2] 和 [Layer 9-10]    ← 两组不连续的层
    GPU 1: [Layer 3-4] 和 [Layer 11-12]
    GPU 2: [Layer 5-6] 和 [Layer 13-14]
    GPU 3: [Layer 7-8] 和 [Layer 15-16]

  为什么能压缩气泡：
    普通 1F1B：GPU 0 做完 Layer 1-2 的前向后，必须等下游 GPU 传回梯度才能继续
              → 这段等待时间是气泡
    交错 1F1B：GPU 0 等待期间，可以切换到 Layer 9-10 做前向
              → 大气泡被拆成小气泡，并被计算填充！

  变量：P = 物理 pipeline stages（GPU 数），M = micro-batch 数量，V = 每个 GPU 的虚拟阶段数
  气泡率近似从：
    (P - 1) / (M + P - 1)
  降为：
    (P - 1) / (M × V + P - 1)

  具体数字（P=4, M=8, V=2）：
    普通 1F1B：  (4-1)/(8+4-1) = 3/11 ≈ 27.3%
    交错 1F1B：  (4-1)/(8×2+4-1) = 3/19 ≈ 15.8%
    → 气泡率几乎减半

  核心权衡：
    优点：通过虚拟阶段交错执行，显著压缩 warmup/drain 气泡
    缺点：每个 micro-batch 需要在更多 GPU 间传递（通信次数增加）；
         同一物理 GPU 上多个虚拟阶段增加调度复杂度；
         通信链路从"仅相邻 GPU"变为"需要与更多前后阶段通信"→ 带宽压力增大
```

![其他流水线调度策略](<../images/8-22-其他的流水线策略.png>)

**普通 1F1B vs Interleaved 1F1B 对比**：

| 维度 | 普通 1F1B | Interleaved 1F1B（VPP） |
|------|----------|------------------------|
| 设备-阶段关系 | 1 个 GPU → 1 组连续阶段 | 1 个 GPU → 多组不连续阶段 |
| 气泡率 | (P-1)/(M+P-1) | (P-1)/(M×V+P-1) |
| 通信复杂度 | 低（仅相邻 GPU 通信） | 高（通信次数、链路数增加） |
| 调度复杂度 | 简单 | 复杂（需协调多阶段切换） |
| 适用场景 | 设备数较少、模型中等 | 大模型训练、设备数受限时最大化利用率 |

#### 3.2 零气泡流水线

```
零气泡流水线（Zero Bubble Pipeline, 2024）：
  核心思想：将反向传播进一步拆分为 B 和 W
    B = 计算输入梯度 ∂L/∂x，并把梯度传给上一个 stage
    W = 计算权重梯度 ∂L/∂W，可以延迟到气泡时间执行

  为什么可以拆分？
    反向传播有两个任务：
    ① 计算 ∂L/∂x（输入梯度）→ 传给上一层，必须立即执行
    ② 计算 ∂L/∂W（权重梯度）→ 用于更新参数，可以稍后执行

    → 将 W 延迟到气泡时间执行 → 气泡被填满！

  在依赖关系满足、W 能填入气泡且优化器仍在 global batch 末尾统一更新时，
  可以在保持同步语义的前提下把 1F1B 的 warmup/drain 气泡压到接近 0。
  代价是调度更复杂，并且延迟 W 可能需要更久地保留计算 W 所需的激活/梯度。
```

![零气泡流水线技术](<../images/8-23-零气泡流水线技术.png>)

---

### 四、流水线并行的代码实现

来自原课程的 `pipeline_parallelism_main` 函数：

```python
def pipeline_parallelism_main(rank, world_size, data, num_layers, num_micro_batches):
    setup(rank, world_size)
    data = data.to(get_device(rank))

    # 每个 rank 分配 local_num_layers 层
    local_num_layers = int_divide(num_layers, world_size)
    local_params = [get_init_params(num_dim, num_dim, rank) for i in range(local_num_layers)]

    # 拆分为微批次
    micro_batch_size = int_divide(batch_size, num_micro_batches)

    if rank == 0:
        micro_batches = data.chunk(chunks=num_micro_batches, dim=0)
    else:
        micro_batches = [torch.empty(micro_batch_size, num_dim, device=get_device(rank))
                         for _ in range(num_micro_batches)]

    for x in micro_batches:
        # 接收上一级的激活值（点对点通信）
        if rank - 1 >= 0:
            dist.recv(tensor=x, src=rank - 1)

        # 计算本 rank 负责的层
        for param in local_params:
            x = x @ param
            x = F.gelu(x)

        # 发送给下一级（点对点通信）
        if rank + 1 < world_size:
            dist.send(tensor=x, dst=rank + 1)
```

**关键点**：
- 使用 `dist.send()` / `dist.recv()` 做点对点通信（而非集合通信）
- `data.chunk()` 将大批次拆分为微批次
- 这是**最朴素的实现**：没有计算-通信重叠，没有反向传播
- 代码片段依赖外部定义的 `batch_size`、`num_dim`、`setup()`、`get_device()` 等变量/函数，
  用于说明通信流程，不是可直接独立运行的完整训练脚本

![四层模型的流水线并行示意](<../images/8-54-假设模型有四层.png>)

---

### 五、流水线并行 vs 数据并行

| 维度 | 数据并行（DDP） | 流水线并行（PP） |
|------|:----------:|:----------:|
| 切分方式 | 数据批次 | 模型层 |
| 每个 GPU 存储 | 完整模型 | 部分层 |
| 通信操作 | All-Reduce（梯度） | Send/Recv（激活值） |
| 通信特征 | 反向传播中按梯度 bucket 做 All-Reduce | 每个 micro-batch 在相邻 stage 间传激活/梯度 |
| GPU 利用率 | 无流水线气泡，但可能等待梯度同步 | 有 warmup/drain 气泡 |
| 适用场景 | 模型能放下单卡 | 模型太大单卡放不下 |
| 通信要求 | 可跨节点，对 All-Reduce 带宽敏感 | 相邻 stage 通信频繁，最好有高速互联 |

> 💡 **选择建议**：如果模型、优化器状态和激活值能放入单卡，优先从数据并行开始，系统更简单。
> 当单卡放不下，或需要把更大模型扩展到多机多卡时，再结合流水线并行、张量并行和数据并行。

---

## 参考资料

- GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism, NeurIPS 2019：<https://research.google/pubs/gpipe-efficient-training-of-giant-neural-networks-using-pipeline-parallelism/>
- NVIDIA Megatron-LM 技术博客（1F1B、Interleaved 1F1B、气泡与内存开销）：<https://developer.nvidia.com/blog/scaling-language-model-training-to-a-trillion-parameters-using-megatron/>
- PipeDream: Generalized Pipeline Parallelism for DNN Training, SOSP 2019：<https://www.microsoft.com/en-us/research/blog/pipedream-a-more-effective-way-to-train-deep-neural-networks-using-pipeline-parallelism/>
- Zero Bubble (Almost) Pipeline Parallelism, ICLR 2024：<https://openreview.net/forum?id=tuzTN0eIO5>

---

## 🧠 本模块问题

请在下方回答以下问题后，输入 `提交作业` 提交。

**Q1**：4 个 GPU 使用朴素层并行训练一个 8 层模型（每 GPU 2 层），单个批次的前向和反向各耗时 1 秒。请画出时间线并计算 GPU 利用率。如果将批次拆分为 4 个微批次，利用率如何变化？

**Q2**：1F1B 调度相比 GPipe 调度有什么优势？为什么 1F1B 的峰值激活内存主要受流水线深度影响，而不像 GPipe 那样随微批次数量线性增长？

**Q3**：零气泡流水线将反向传播拆分为 B（计算输入梯度）和 W（计算权重梯度）。为什么 W 可以延迟执行？延迟执行 W 对训练结果有影响吗？

---

<!-- 学习者作答区（请在此处填写你的答案） -->

**A1**：

朴素层并行，每个 GPU 负责 2 层
step  0  1    2   3   4   5   6   7
GPU0 [F]                         [B]
GPU1     [F]                 [B]
GPU2         [F]         [B]
GPU3             [F] [B]

GPU 利用率为 2/8 = 25%，即朴素层并行单个 GPU 利用率为 25%

将批次分拆为 4 个微批次，则
step    1     2     3     4     5     6     7     8     9     10    11    12    13    14
GPU0  [F1]  [F2]  [F3]  [F4]                                      [B1]  [B2]  [B3]  [B4]
GPU1        [F1]  [F2]  [F3]  [F4]                          [B1]  [B2]  [B3]  [B4]
GPU2              [F1]  [F2]  [F3]  [F4]              [B1]  [B2]  [B3]  [B4]    
GPU3                    [F1]  [F2]  [F3]  [F4]  [B1]  [B2]  [B3]  [B4]

以上是微批次（微批次为 4）并行的调度图，可以看到每个 GPU 利用率为
(N - 1) / (M + N - 1) = 3/7 ≈ 42.85%


**A2**：

GPipe 其实就是 微批次调度，它有个弊端就是 微批次 在进行完 所有 前向传播 后，才能进行反向传播，气泡率高
但是因为 1F1B 调度，每个设备（GPU）可以处理多个不连续的层，因此每个设备所需要的存储空间更大

Gpipe 每个设备处理的 模型层数 是连续的，并且是一开始固定的，所以 微批次 越多，内存占用越高（单个微批次占用一定）
而 1F1B 每个设备可以处理不连续的层，比如 4 个 GPU 设备，模型层数为 16，那么 GPU0 可以处理模型层数 [1, 2] 以及 [9, 10] 层，如果流水线越深（步骤越多），那么 设备需要处理的 层数越多，所以峰值激活内存越多。



**A3**：

模型的前向传播可以理解为
Y = W*X^T + b
则反向传播
有梯度 ∇Y ，则 ∇X = ∇Y/∇W_T，而 ∇W = ∇Y/∇X_T
而 ∇X 是上一层计算梯度需要的输入，即输入梯度 B，是需要立即计算的，因为上一层需要
而权重梯度 ∇W 即 W，是不需要马上计算的，而是在梯度更新前完成就可以了，可以延后计算。
延迟计算对结果无影响，而且可以优化整个 GPU 并行利用率。


---

<!-- 教师批改区（提交作业后由导师填写，请勿手动修改） -->

### 📝 批改结果

**Q1 批改**：时间线和朴素利用率完全正确，微批次调度图画的是 GPipe 风格，依赖链和气泡率公式正确 — 得分：**9/10**

- ✅ 朴素层并行时间线正确：GPU0 在 T1 算 F、T8 算 B，GPU3 在 T4 算 F、T5 算 B
- ✅ 利用率 2/8 = 25% 正确
- ✅ 微批次时间线正确（GPipe 风格：全前向后全反向）
- ✅ 气泡率公式 `(N-1)/(M+N-1) = 3/7 ≈ 42.85%` 正确
- ⚠️ 没明确写出利用率：气泡率 42.85% → 利用率 = 1 - 42.85% = **57.1%**（从 25% 提升到 57.1%）

<details>
<summary>📖 Q1 参考答案</summary>

**朴素层并行（4 GPU，8 层，每 GPU 2 层，F 和 B 各 1 秒）**：

```
时间   1      2      3      4      5      6      7      8
GPU 0: [ F ]  ████   ████   ████   ████   ████   ████   [ B ]
GPU 1: ████   [ F ]  ████   ████   ████   ████   [ B ]  ████
GPU 2: ████   ████   [ F ]  ████   ████   [ B ]  ████   ████
GPU 3: ████   ████   ████   [ F ]  [ B ]  ████   ████   ████

反向依赖链：GPU3[B](T5) → GPU2[B](T6) → GPU1[B](T7) → GPU0[B](T8)
```

利用率 = 2 步工作 / 8 步总时间 = **25%**

**微批次调度（4 micro-batch，GPipe 风格）**：

```
时间:  01 02 03 04 05 06 07 08 09 10 11 12 13 14
GPU0: F1 F2 F3 F4 -- -- -- -- -- -- B4 B3 B2 B1
GPU1: -- F1 F2 F3 F4 -- -- -- -- B4 B3 B2 B1 --
GPU2: -- -- F1 F2 F3 F4 -- -- B4 B3 B2 B1 -- --
GPU3: -- -- -- F1 F2 F3 F4 B4 B3 B2 B1 -- -- --
```

- 总时间：14 步
- 有用计算：4 GPU × 2 步（4F + 4B）= 8 步
- 气泡：14 - 8 = 6 步
- 气泡率 = 6/14 = 3/7 ≈ **42.9%**
- 利用率 = 1 - 42.9% = **57.1%**（从 25% 提升到 57.1%）

</details>

---

**Q2 批改**：核心概念混淆，把 1F1B 和 Interleaved 1F1B（VPP）搞混了 — 得分：**4/10**

- ❌ "1F1B 每个设备可以处理多个不连续的层"——这是 **Interleaved 1F1B / VPP** 的特征，不是普通 1F1B
- ❌ 普通 1F1B 的 GPU 仍然只负责一组连续层，优势在于 **F/B 交替调度降低峰值激活内存**
- ⚠️ GPipe 的内存问题描述部分正确（微批次越多内存越高），但原因是"所有前向做完才能反向，需保留全部激活"
- ⚠️ 缺少关键对比：1F1B 的峰值激活内存由流水线深度 P 决定（稳定阶段只有 P 个激活在手），不像 GPipe 随 M 线性增长

<details>
<summary>📖 Q2 参考答案</summary>

**1F1B vs GPipe 的核心差异**：

| 维度 | GPipe | 1F1B |
|------|-------|------|
| 调度方式 | 全前向 → 全反向 | F/B 交替 |
| 峰值激活内存 | 随微批次数 M 线性增长 | 主要由流水线深度 P 决定 |
| 气泡率 | (P-1)/(M+P-1) | 同阶，也是 (P-1)/(M+P-1) |

**为什么 GPipe 激活内存随 M 增长**：
- GPipe 必须等所有 M 个 micro-batch 做完前向，才能开始反向
- 反向需要前向的激活值，所以在整个前向阶段，所有 M 个 micro-batch 的激活都必须保留
- 峰值激活内存 = M × 单个 micro-batch 的激活内存

**为什么 1F1B 激活内存由 P 决定**：
- 1F1B 在稳定阶段交替执行 Forward/Backward
- 做 Backward 时立即释放该 micro-batch 的激活值
- 同一时刻最多只有约 P 个 micro-batch 的激活在手（对应流水线深度）
- 峰值激活内存 ≈ P × 单个 micro-batch 的激活内存

**举例（P=4, M=16）**：
- GPipe 峰值：16 份激活（全部保留等反向）
- 1F1B 峰值：约 4 份激活（交替释放）
- 节省：75% 激活内存

**常见误解**：
- 1F1B 的优势不是"处理不连续层"（那是 VPP/Interleaved 1F1B）
- 1F1B 的气泡率与 GPipe 同阶，不是通过减少气泡来优化，而是通过降低内存来优化

</details>

---

**Q3 批改**：核心推理正确，公式表述有小错 — 得分：**8/10**

- ✅ ∇X 必须立即计算（上一层需要）— 正确
- ✅ ∇W 可以延后（参数更新前完成即可）— 正确
- ✅ "延迟计算对结果无影响"— 正确，因为参数在 global batch 结束时才更新
- ⚠️ 公式 `∇X = ∇Y/∇W_T` 不准确，应为 `∇X = ∇Y @ W`（链式法则的矩阵乘法，不是除法）

<details>
<summary>📖 Q3 参考答案</summary>

**反向传播的两个任务**：

对于一层的线性变换 `Y = X @ W + b`，反向传播需要计算：

1. **输入梯度 B**：`∇X = ∇Y @ W^T` — 传给上一层，必须立即执行
2. **权重梯度 W**：`∇W = X^T @ ∇Y` — 用于更新参数，可以延迟执行

**为什么 W 可以延迟**：
- 参数更新发生在 global batch 结束时（优化器 step）
- 只要在 optimizer.step() 之前计算完 ∇W，就不会影响结果
- ∇W 的计算不依赖其他层，是完全独立的

**延迟执行 W 对训练结果有影响吗**：
- **没有影响**。因为：
  - 同步训练中，参数在 global batch 结束时统一更新
  - 无论 ∇W 是立即算还是延迟算，只要在更新前算完，最终更新的参数完全一致
  - W 的延迟只是"调度上的延迟"，不是"计算上的省略"

**零气泡流水线的价值**：
- 将 W 延迟到气泡时间执行 → 原本空闲的 GPU 被填满 → 气泡率接近 0
- 代价：需要更久保留计算 W 所需的激活值和中间梯度

</details>

---

**综合评价**：Q1 掌握扎实；Q3 核心理解正确但公式需修正；Q2 存在概念混淆（1F1B vs VPP），建议复习 2.3 节中 1F1B 的 F/B 交替机制和激活内存释放逻辑。

**批改时间**：2026-05-13
