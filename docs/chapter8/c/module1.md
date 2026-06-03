# 第 8 章：分布式训练 — 模块 1：通信原语与硬件架构

> 📍 学习进度：第 8 章，第 1 / 5 模块
> 📅 生成时间：2026-05-08

---

## 学习目标

- 理解为什么需要分布式训练（单卡内存墙与计算墙）
- 掌握 7 种集合通信原语的含义与区别
- 理解 All-Reduce 的 Ring 算法原理
- 了解 GPU 互联硬件：NVLink、NVSwitch、InfiniBand 的带宽层级
- 理解计算与通信重叠的基本思路

---

## 核心内容

### 一、为什么需要分布式训练

#### 1.1 单卡的两道墙

```
内存墙（Memory Wall）：
  70B 参数模型 → 70B × 4 字节 = 280 GB 仅存参数
  加上梯度（280 GB）+ Adam 优化器状态（280 × 2 = 560 GB）
  总计：≈ 1,120 GB = 1.1 TB

  A100（80GB）：单卡根本放不下
  即使 8×A100 节点：8 × 80 = 640 GB，仍然不够

计算墙（Compute Wall）：
  训练 1T token 的 GPT-3 级别模型
  单卡 A100：≈ 数十年
  分布式：数天到数周
```

![GPU 算力增长 vs 模型规模增长](<../images/8-1-GPU的算力增强曲线.png>)

![模型尺寸随年份的增长趋势](<../images/8-2-模型的尺寸变化.png>)

> 核心矛盾：**GPU 算力在增长，但模型规模增长更快**。单卡算力和内存永远不够用，分布式训练是必然选择。

#### 1.2 分布式训练的基本思路

```
核心思想：分而治之

  数据并行（Data Parallelism）：每个 GPU 持有完整模型，处理不同数据
  模型并行（Model Parallelism）：模型切分到多个 GPU
    ├── 流水线并行（Pipeline Parallelism）：按层切
    └── 张量并行（Tensor Parallelism）：按维度切

  实际训练中往往混合使用 → 3D 并行（后续模块详细讲解）
```

![多机并行示意图](<../images/8-3-多机并行.png>)

---

### 二、通信原语

分布式训练的基础是 **GPU 之间的通信**。通信操作分为两类：集合通信（Collective Communication）和点对点通信（Point-to-Point）。

#### 2.1 集合通信原语

集合通信涉及一组 GPU（通常称为一个通信组/Process Group）同时参与的操作。

**七种核心原语**：

```
① Broadcast（广播）：1 → N
  一个 GPU 将数据发送给组内所有 GPU
  例：GPU 0 把参数发给 GPU 1, 2, 3

② Scatter（散射）：1 → N（分块）
  一个 GPU 将数据的不同部分分发给不同 GPU
  例：GPU 0 把 [A|B|C|D] → GPU 0 得 A, GPU 1 得 B, ...

③ Gather（收集）：N → 1
  一个 GPU 收集所有 GPU 的数据
  例：GPU 0 收集 GPU 0 的 A, GPU 1 的 B, GPU 2 的 C, GPU 3 的 D

④ Reduce（归约）：N → 1
  所有 GPU 的数据做运算（如求和）后汇总到一个 GPU
  例：GPU 0 得到 sum(所有 GPU 的梯度)

⑤ All-Reduce（全归约）：N → N
  所有 GPU 都得到归约结果
  例：每个 GPU 都得到 sum(所有 GPU 的梯度)
  → DDP 的核心操作！

⑥ All-Gather（全收集）：N → N
  所有 GPU 都收集到所有 GPU 的数据
  例：每个 GPU 都持有 [A, B, C, D]
  → 张量并行的核心操作！

⑦ Reduce-Scatter（归约散射）：N → N（分块归约）
  先归约再散射：每个 GPU 只得到归约结果的一个分片
  例：GPU 0 得 sum(各 GPU 数据)[0:N/4]
  → ZeRO 的核心操作！
```

![集体通讯操作概览](<../images/8-4-集体通讯操作.png>)

**逐个详解**：

![Broadcast 广播机制](<../images/8-40-广播机制.png>)

![Scatter 散射](<../images/8-41-散射.png>)

![Gather 收集](<../images/8-42-Gather.png>)

![Reduce 归约](<../images/8-43-Reduce.png>)

![All-Gather 全收集](<../images/8-44-AllGather.png>)

![Reduce-Scatter 归约散射](<../images/8-45-reduce_scatter.png>)

![All-Reduce 全归约](<../images/8-46-all_reduce.png>)

#### 2.2 对比与映射

| 原语 | 方向 | 典型用途 | 对应的并行策略 |
|------|------|---------|--------------|
| Broadcast | 1→N | 参数初始化同步 | — |
| Scatter | 1→N（分块） | 数据分发 | 流水线并行 |
| Gather | N→1 | 激活值汇总 | — |
| Reduce | N→1 | 梯度汇总到主节点 | — |
| **All-Reduce** | N→N | 梯度同步 | **数据并行（DDP）** |
| **All-Gather** | N→N | 激活值/参数收集 | **张量并行** |
| **Reduce-Scatter** | N→N | 分片归约 | **ZeRO** |

> 💡 **记忆技巧**：All-Reduce = Reduce-Scatter + All-Gather。先将归约结果分片（每个 GPU 拿一部分），再让所有 GPU 收集完整结果。

#### 2.3 点对点通信

```
Send / Recv：两个 GPU 之间直接传输数据
  → 用于流水线并行中层与层之间的激活值传递
  → 不需要所有 GPU 同步参与

  GPU 0: dist.send(tensor=x, dst=1)
  GPU 1: dist.recv(tensor=x, src=0)
```

---

### 三、All-Reduce 的 Ring 算法

All-Reduce 是分布式训练中最核心的操作。理解 Ring 算法有助于理解通信开销。

#### 3.1 直觉类比

```
场景：4 个学生各持有一份作业，要让每人都有全部 4 份作业

朴素方法：每个人把自己的作业发给其他 3 人 → 4 × 3 = 12 次传输
Ring 方法：
  Step 1：每人传给右边邻居 → 每人有 2 份
  Step 2：每人传给右边邻居 → 每人有 3 份
  Step 3：每人传给右边邻居 → 每人有 4 份 ✅
  只需 3 步（N-1 步），每步所有链接同时工作
```

#### 3.2 算法分析

```
N 个 GPU 的 Ring All-Reduce：

  通信量：每个 GPU 发送 2 × (N-1)/N × 数据量
  延迟：N-1 步（每步有固定延迟）
  带宽利用率：100%（所有链路在每一步都同时工作）

  对比 Tree All-Reduce：
    带宽利用率：~95%
    延迟：log(N) 步（延迟更低）

  选择：大数据量用 Ring，小数据量用 Tree
```

> 🌐 **补充（Web Search / NCCL 官方）**：NVIDIA NCCL 库根据消息大小自动选择 Ring 或 Tree 算法。据 NCCL 官方文档，Ring 算法自 NCCL 2.0 起支持（100% 带宽，线性延迟），Tree 算法自 NCCL 2.4 起支持（95% 带宽，对数延迟）。对于大消息（MB 级别），Ring 算法带宽效率更优。
>
> 来源：[NCCL Performance Documentation](https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md)、[NVIDIA NCCL Presentation (SC15)](https://images.nvidia.com/events/sc15/pdfs/NCCL-Woolley.pdf)

---

### 四、硬件架构：GPU 互联

#### 4.1 互联层级

```
通信带宽从高到低：

  ① GPU 片内（SM 之间）：~10 TB/s（SRAM/寄存器）
  ② NVLink（GPU ↔ GPU，节点内）：600 GB/s（A100）/ 900 GB/s（H100）
  ③ NVSwitch（8 GPU 全互联）：600 GB/s per GPU（A100 DGX）
  ④ PCIe Gen4（GPU ↔ CPU/其他设备）：~32 GB/s（x16）
  ⑤ InfiniBand NDR（节点间）：400 Gb/s ≈ 50 GB/s（单端口）
  ⑥ 以太网（普通网络）：10-25 Gb/s ≈ 1.25-3.1 GB/s

关键启示：
  节点内通信（NVLink）>> 节点间通信（InfiniBand）
  → 张量并行（频繁通信）应该放在同一节点内
  → 数据并行（通信量较少）可以跨节点
```

![典型 GPU 硬件架构](<../images/8-47-典型的GPU硬件架构.png>)

![现代数据中心 GPU 集群](<../images/8-48-现代的数据中心.png>)

#### 4.2 NCCL：GPU 通信的底层库

```
NCCL（NVIDIA Collective Communications Library）：
  → NVIDIA 专门为 GPU 间通信优化的库
  → 提供 All-Reduce、Broadcast、Reduce-Scatter 等集合通信原语
  → 自动检测硬件拓扑（NVLink、PCIe、InfiniBand）
  → 自动选择最优通信算法（Ring、Tree）

PyTorch 的 dist.all_reduce() 底层调用的就是 NCCL
```

> 🌐 **补充（Web Search / NCCL 2025）**：据 2025 年 arXiv 论文 "Demystifying NCCL" 分析，NCCL 支持三种通信协议：Simple（通用协议）、LL（Low Latency，小消息优化）、LL128（中等消息优化）。此外，NVSwitch 支持 SHARP（Scalable Hierarchical Aggregation and Reduction Protocol）硬件内归约，可以将 All-Reduce 的部分计算卸载到交换机上完成，进一步降低延迟。
>
> 来源：[Demystifying NCCL (arXiv 2507.04786)](https://arxiv.org/html/2507.04786v1)、[NCCL NVLS/SHARP](https://wentao.site/nccl_summary/)

> 💡 **补充（Context7 / PyTorch）**：
>
> **NCCL 后端可用性检查**：在编写分布式代码时，可以使用 `torch.distributed.is_nccl_available()` 检查当前环境是否支持 NCCL 后端。这在编写可移植的分布式训练脚本时非常有用——如果 NCCL 不可用，可以回退到 Gloo 后端：
> ```python
> import torch.distributed as dist
> if dist.is_nccl_available():
>     backend = "nccl"
> else:
>     backend = "gloo"  # CPU fallback
> dist.init_process_group(backend=backend)
> ```
>
> **使用 torch.profiler 分析通信性能**：PyTorch 提供 `torch.profiler` 工具可以精确测量集合通信操作的耗时。在 All-Reduce、All-Gather 等操作前后添加 profiler 上下文，可以直观看到通信在总训练时间中的占比：
> ```python
> with torch.profiler.profile(
>     activities=[torch.profiler.ProfilerActivity.CUDA],
>     record_shapes=True,
> ) as prof:
>     dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
>     torch.cuda.synchronize()
> print(prof.key_averages().table(sort_by="cuda_time_total"))
> ```
> 这对于诊断通信瓶颈（如判断是否需要升级互联硬件）至关重要。
>
> 来源：[PyTorch Distributed Communication](https://docs.pytorch.org/docs/stable/distributed.html)

---

### 五、计算与通信重叠

```
核心思想：在 GPU 计算的同时进行通信，隐藏通信延迟

  传统方式（串行）：
    计算梯度 → 通信（All-Reduce）→ 更新参数
    总时间 = 计算时间 + 通信时间

  重叠方式（并行）：
    计算第 N 层梯度的同时 → 通信第 N-1 层梯度
    总时间 ≈ max(计算时间, 通信时间)

PyTorch DDP 的实现：
  反向传播过程中，每计算完一层的梯度就立即发起 All-Reduce
  而不是等所有梯度计算完再统一通信
  → 流水线化：计算和通信并行执行
```

> 这就是为什么 DDP 通常比手动实现 all_reduce 快得多——它在反向传播过程中自动实现了计算-通信重叠。

---

## 🧠 本模块问题

请在下方回答以下问题后，输入 `提交作业` 提交。

**Q1**：请解释 All-Reduce 和 Reduce-Scatter 的区别。在数据并行（DDP）中使用的是哪一个？在 ZeRO 优化中呢？

**Q2**：为什么张量并行（Tensor Parallelism）需要放在同一节点内的 GPU 之间，而数据并行可以跨节点？请用具体的带宽数字解释。

**Q3**：在 4 个 GPU 上用 Ring 算法执行 All-Reduce，传输一个 100MB 的梯度张量。请描述每一步发生了什么，并解释为什么 Ring 算法的带宽利用率是 100%。

---

<!-- 学习者作答区（请在此处填写你的答案） -->

**A1**：

All-Reduce = Reduce Scatter + All-Gather
每个节点获取所有数据的分片 运算结果，再将这部分结果统一发送到每个节点中，得到 gather 的结果。

数据并行（DDP）中使用的是 All-Reduce，在 ZeRO 优化中使用的是 Reduce-Scatter。


**A2**：

同一个节点内的 GPU 之间通信带宽是 600GB/s，而不同节点直接使用 infinite HDR 通信是 50~60GB/s。

对于张量（Tensor）来，同一个节点内通信带宽可以满足要求，并且通过 NVSwitch 可以满载通信带宽，而数据并行，带宽要求没那么严苛，所以可以跨节点。


**A3**：

4个 GPU 上 用 Ring 算法执行 All-Reduce，传输一个 100MB 的梯度张量。
那么每个 GPU 上应该得到 25MB 的梯度张量。
假设 A、B、C、D 4个GPU
A GPU 存放 100MB 梯度张量 [a, b, c, d]
那么根据 Ring 算法
step 1. A(a, b, c, d) -> B 25MB 数据 d,  因为 B、C、D 无数据不操作
step 2. A(a, b, c) -> B(d,) 25MB 数据 c, B(d,)->C 25MB d
step 3. A(a, b) -> B(c,) 25MB 数据 b, B(c,) -> C(d,), C(d,) -> D
最后 A、B、C、D 各分到数据 a, b, c ,d


---

<!-- 教师批改区（提交作业后由导师填写，请勿手动修改） -->

### 📝 批改结果

**Q1 批改**：核心关系正确，但对 Reduce-Scatter 的描述不够准确 — 得分：**7/10**

- ✅ All-Reduce = Reduce-Scatter + All-Gather，公式正确
- ✅ DDP 用 All-Reduce，ZeRO 用 Reduce-Scatter，映射正确
- ⚠️ "每个节点获取所有数据的分片运算结果"表述有歧义：Reduce-Scatter 的结果是**每个节点只拿到归约结果的一个分片**（不是"所有分片"），例如 4 个 GPU 各持有 100MB 数据，Reduce-Scatter 后每个 GPU 只有 25MB 的求和结果

<details>
<summary>📖 Q1 参考答案</summary>

**All-Reduce**：所有 GPU 都得到完整的归约结果（如梯度求和）。4 个 GPU 各有 100MB 梯度 → All-Reduce 后每个 GPU 都有 100MB 的 sum 结果。

**Reduce-Scatter**：先归约再散射，每个 GPU 只得到归约结果的**一个分片**。同样 4 个 GPU 各 100MB → Reduce-Scatter 后每个 GPU 只有 25MB（sum 结果的 1/4）。

**关系**：All-Reduce = Reduce-Scatter + All-Gather
- Reduce-Scatter：sum 结果分片，每 GPU 拿 1/N
- All-Gather：收集所有分片，每 GPU 拿到完整结果

**应用场景**：
- DDP（数据并行）→ All-Reduce：每个 GPU 需要完整的同步梯度来更新完整模型
- ZeRO → Reduce-Scatter：每个 GPU 只更新自己负责的参数分片，只需梯度的 1/N，节省内存

**常见误解**：Reduce-Scatter 的输出不是"每个 GPU 拿到不同的原始数据"，而是"每个 GPU 拿到归约（如求和）结果的不同部分"。

</details>

---

**Q2 批改**：核心正确，但缺少对"为什么 TP 通信更频繁"的解释 — 得分：**8/10**

- ✅ 带宽数字正确（NVLink 600 GB/s vs InfiniBand 50-60 GB/s）
- ✅ 结论正确：TP 放节点内，DP 可跨节点
- ⚠️ 缺少关键因果链：**为什么 TP 需要高带宽？** 因为 TP 每层都要 All-Gather（高频通信），而 DP 每个 step 只做一次 All-Reduce（低频通信）。正是因为通信频率不同，才对带宽有不同要求

<details>
<summary>📖 Q2 参考答案</summary>

**带宽数据**：
- 节点内 NVLink：600 GB/s（A100）/ 900 GB/s（H100）
- 节点间 InfiniBand HDR：50 GB/s
- 差距：12-18 倍

**根本原因：通信频率不同**

张量并行（TP）：
- 每层结束时都需要 All-Gather 收集激活值
- 一个 Transformer 块 = Attention + MLP → 每块至少 2 次 All-Gather
- 100 层模型 → 每个 step 200+ 次通信
- 高频通信 → 必须用 NVLink（否则通信成为瓶颈）

数据并行（DP）：
- 每个 step 只在反向传播结束后做 1 次 All-Reduce
- 而且可以和计算重叠（DDP 梯度桶机制）
- 低频通信 → InfiniBand 50 GB/s 够用

**结论**：不是 TP "不能"跨节点，而是跨节点后通信延迟会严重拖慢训练。

</details>

---

**Q3 批改**：理解了基本思路，但有两个重要错误 — 得分：**5/10**

- ❌ **关键错误**：All-Reduce 的结果是每个 GPU 都得到**完整的 100MB**，不是 25MB。你说的"每个 GPU 上应该得到 25MB"描述的是 Reduce-Scatter，不是 All-Reduce
- ❌ Ring 算法中**所有 GPU 同时操作**，不是"只有 A 发给 B"。每步每个 GPU 都向右邻居发送一块数据
- ✅ 数据分块 [a, b, c, d] 的思路是对的
- ⚠️ 缺少对"100% 带宽利用率"的解释

<details>
<summary>📖 Q3 参考答案</summary>

**前提纠正**：All-Reduce 的结果是**每个 GPU 都得到完整的 100MB 归约结果**，不是 25MB。

**Ring 算法步骤**（4 GPU，100MB 数据，分 4 块各 25MB）：

所有 GPU 同时操作！每步每条链路传输 25MB：

```
Reduce-Scatter 阶段（3 步）：

Step 1（所有 GPU 同时传给右邻居）：
  A → B: 发送 d(25MB)    B → C: 发送 a(25MB)
  C → D: 发送 b(25MB)    D → A: 发送 c(25MB)
  每个 GPU 接收后与本地对应块求和

Step 2：
  A → B: 发送 c(25MB)    B → C: 发送 d+a(25MB)
  C → D: 发送 a+b(25MB)  D → A: 发送 b+c(25MB)

Step 3：
  各 GPU 完成一个分片的完整归约
  A 有 sum(a), B 有 sum(b), C 有 sum(c), D 有 sum(d)

All-Gather 阶段（3 步）：
  将各自归约好的分片传递给所有 GPU
  Step 4-6：每个 GPU 逐步收集到 sum(a), sum(b), sum(c), sum(d)

结果：A、B、C、D 都有完整的 [sum(a), sum(b), sum(c), sum(d)] = 100MB
```

**为什么带宽利用率是 100%**：
- 每一步，所有 4 条链路（A→B, B→C, C→D, D→A）**同时**在传输数据
- 没有任何链路空闲
- 对比朴素方法：A 同时发给 B、C、D → A 的链路是瓶颈，B、C、D 的链路部分空闲

**常见错误 vs 正确理解**：
- 错误："A 传给 B，B 传给 C，C 传给 D"（串行传递）
- 正确：**所有 GPU 同时传给各自的右邻居**（并行传递）

</details>

---

**综合评价**：Q1 和 Q2 掌握较好，Q3 对 Ring 算法的核心机制（并行传递 + All-Reduce 完整结果）理解有偏差。建议重点复习 Ring All-Reduce 的两个阶段（Reduce-Scatter + All-Gather）以及"所有 GPU 同时操作"的并行特性。

**批改时间**：2026-05-09
