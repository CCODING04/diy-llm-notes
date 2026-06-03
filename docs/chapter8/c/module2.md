# 第 8 章：分布式训练 — 模块 2：数据并行与 ZeRO/FSDP

> 📍 学习进度：第 8 章，第 2 / 5 模块
> 📅 生成时间：2026-05-08

---

## 学习目标

- 掌握数据并行（DDP）的工作原理和内存分布
- 理解 ZeRO 优化的三个阶段及其内存节省计算
- 了解 FSDP（Fully Sharded Data Parallel）与 ZeRO-3 的关系
- 能够计算数据并行下的内存占用

---

## 核心内容

### 一、数据并行（Data Parallelism / DDP）

#### 1.1 直觉理解

```
核心思想：每个 GPU 都持有完整的模型副本，各自处理不同的数据批次
同步方式：每个 step 结束后，通过 All-Reduce 同步梯度

  GPU 0: [数据批次 0-31] → 前向 → 反向 → 梯度 ↘
  GPU 1: [数据批次 32-63] → 前向 → 反向 → 梯度 → All-Reduce → 同步梯度 → 各自更新参数
  GPU 2: [数据批次 64-95] → 前向 → 反向 → 梯度 ↗
  GPU 3: [数据批次 96-127] → 前向 → 反向 → 梯度 ↗
```

#### 1.2 标准训练 vs DDP

```
标准训练（单卡）：
  for batch in data:
      loss = forward(model, batch)
      loss.backward()              # 计算梯度
      optimizer.step()             # 更新参数

DDP（多卡）：
  for batch in local_data:         # 每个 GPU 只看到自己的数据
      loss = forward(model, batch)
      loss.backward()              # 计算梯度
      all_reduce(param.grad)       # ← 唯一的额外操作：同步梯度
      optimizer.step()             # 更新参数
```

> 💡 **DDP 的优雅之处**：只需插入一行 `all_reduce`，其余代码与标准训练完全相同。每个 GPU 计算的梯度不同，但 all_reduce 后梯度相同，所以更新后的参数也相同。

#### 1.3 代码示例

来自原课程的 `data_parallelism_main` 函数：

```python
def data_parallelism_main(rank, world_size, data, num_layers, num_steps):
    setup(rank, world_size)

    # 按 rank 分配数据切片
    local_batch_size = batch_size // world_size
    data = data[rank * local_batch_size : (rank + 1) * local_batch_size].to(device)

    # 每个 rank 创建完整的模型参数
    params = [get_init_params(num_dim, num_dim, rank) for i in range(num_layers)]
    optimizer = torch.optim.AdamW(params, lr=1e-3)

    for step in range(num_steps):
        # 前向传播（与标准训练相同）
        x = data
        for param in params:
            x = x @ param
            x = F.gelu(x)
        loss = x.square().mean()

        # 反向传播（与标准训练相同）
        loss.backward()

        # 梯度同步（DDP 的唯一额外操作）
        for param in params:
            dist.all_reduce(tensor=param.grad, op=dist.ReduceOp.AVG)

        optimizer.step()
```

**关键点**：
- `dist.all_reduce(tensor=param.grad, op=dist.ReduceOp.AVG)` — 对每个参数的梯度做全归约（取平均）
- `async_op=False` — 同步操作，所有进程在此等待直到完成
- All-Reduce 本身是**同步点**，确保所有进程步调一致

![数据并行打印输出](<../images/8-55-打印输出信息.png>)

> 各节点的**损失值不同**（因为数据不同），但**参数相同**（因为梯度被同步了）。这正是数据并行的核心特征。

---

### 二、数据并行的内存问题

#### 2.1 内存组成

```
每个 GPU 上存储的内容：

  参数（Parameters, P）    :  Ψ × 4 字节（FP32）
  梯度（Gradients, G）     :  Ψ × 4 字节
  优化器状态（Optimizer, O）:  对于 Adam：m（4Ψ 字节）+ v（4Ψ 字节）= 8Ψ 字节

  Adam 的完整内存：
    P + G + O = 4Ψ + 4Ψ + 8Ψ = 16Ψ 字节

  例：7B 参数模型 → 7 × 10⁹ × 16 = 112 GB
```

![朴素数据并行的内存使用](<../images/8-7-朴素数据并行中的内存使用情况.png>)

#### 2.2 冗余问题

```
朴素数据并行的内存冗余：

  每个 GPU 都存储了完整的 P、G、O
  4 个 GPU → 4 × 16Ψ = 64Ψ 字节（总内存）
  实际需要：16Ψ 字节
  冗余率：4×（完全冗余！）

  问题：大部分内存被优化器状态（O）占用了
  Adam 的 O = 8Ψ，占总内存的 50%
  而每个 GPU 的优化器状态是完全相同的 → 纯浪费
```

---

### 三、ZeRO 优化

ZeRO（Zero Redundancy Optimizer）由微软 DeepSpeed 团队提出，核心思想是**消除数据并行中的内存冗余**。

#### 3.1 ZeRO 的三个阶段

```
ZeRO Stage 1（ZeRO-1）：分片优化器状态 O
  每个 GPU 只存储 O 的 1/N
  内存从 16Ψ → 4Ψ + 4Ψ + 8Ψ/N = (8 + 8/N)Ψ

ZeRO Stage 2（ZeRO-2）：分片 O + 梯度 G
  每个 GPU 只存储 O 和 G 的 1/N
  内存从 16Ψ → 4Ψ + (4Ψ + 8Ψ)/N = (4 + 12/N)Ψ

ZeRO Stage 3（ZeRO-3）：分片 O + G + 参数 P
  每个 GPU 只存储所有内容的 1/N
  内存从 16Ψ → 16Ψ/N

具体数字（N=4, 7B 模型）：
  无 ZeRO：  112 GB / GPU
  ZeRO-1：  72 GB / GPU  （节省 36%）
  ZeRO-2：  52 GB / GPU  （节省 54%）
  ZeRO-3：  28 GB / GPU  （节省 75%）
```

![ZeRO 分片示意图](<../images/8-8-ZeRO示意图.png>)

![ZeRO-1 优化器状态分片](<../images/8-9-优化器状态分片.png>)

![ZeRO Stage 1 工作流程](<../images/8-10-ZeRO工作阶段1.png>)

#### 3.2 ZeRO 各阶段详解

**ZeRO-1：只分片优化器状态**

```
工作流程：
  ① 前向传播：每个 GPU 用完整参数计算
  ② 反向传播：每个 GPU 计算完整梯度
  ③ All-Reduce 同步梯度（同 DDP）
  ④ 每个 GPU 只更新自己负责的那部分优化器状态
  ⑤ 每个 GPU 只有更新后的部分参数

  通信量：与 DDP 相同（All-Reduce 梯度）
  内存节省：优化器状态减少为 1/N
```

**ZeRO-2：分片优化器状态 + 梯度**

```
关键改进：使用 Reduce-Scatter 替代 All-Reduce
  All-Reduce：每个 GPU 得到完整梯度 → 内存开销大
  Reduce-Scatter：每个 GPU 只得到梯度的一个分片 → 内存节省

  通信量：与 All-Reduce 相同（因为 All-Reduce = Reduce-Scatter + All-Gather）
  内存节省：优化器状态 + 梯度都减少为 1/N
```

![ZeRO Stage 2 工作流程](<../images/8-11-ZeRO工作阶段2.png>)

![ZeRO-2 详细工作流程](<../images/8-12-ZeRO工作阶段2的工作流程.png>)

**ZeRO-3：分片一切**

```
最激进的优化：参数也分片
  每个 GPU 只持有参数的 1/N

  前向传播：
    ① 用 All-Gather 临时收集完整参数
    ② 计算该层的激活值
    ③ 释放不需要的参数（可以立即释放！）

  反向传播：
    ④ 用 All-Gather 收集参数
    ⑤ 计算该层的梯度
    ⑥ 用 Reduce-Scatter 分发梯度
    ⑦ 每个 GPU 只更新自己负责的参数分片

  通信量：比 ZeRO-2 更多（前向也需要通信）
  内存节省：最大的（16Ψ/N）
```

![ZeRO Stage 3 工作流程](<../images/8-13-ZeRO工作阶段3.png>)

#### 3.3 内存计算公式

```
令：
  Ψ = 模型参数量
  N = 数据并行的 GPU 数量
  K = 优化器状态倍数（Adam = 2，SGD = 1）

每 GPU 内存 = P/N_分片 + G/N_分片 + O/N_分片

  无 ZeRO：     Ψ × (4 + 4 + 4K) = Ψ × (4 + 4 + 8) = 16Ψ
  ZeRO-1：     Ψ × (4 + 4 + 4K/N) = Ψ × (8 + 8/N)
  ZeRO-2：     Ψ × (4 + 4/N + 4K/N) = Ψ × (4 + 12/N)
  ZeRO-3：     Ψ × (4/N + 4/N + 4K/N) = Ψ × 16/N
```

---

### 四、FSDP（Fully Sharded Data Parallel）

#### 4.1 FSDP 与 ZeRO-3 的关系

```
FSDP = PyTorch 官方实现的 ZeRO-3

  DeepSpeed ZeRO-3：第三方库实现
  PyTorch FSDP：PyTorch 内置实现，API 更统一

  核心思想完全相同：将参数、梯度、优化器状态全部分片
```

![FSDP 原理](<../images/8-14-FSDP的原理.png>)

#### 4.2 FSDP 的工作流程

```
前向传播（Forward）：
  ① All-Gather：收集完整参数
  ② 计算该层的激活值
  ③ 释放非本分片的参数（节省内存！）

反向传播（Backward）：
  ④ All-Gather：再次收集完整参数
  ⑤ 计算梯度
  ⑥ Reduce-Scatter：每个 GPU 只保留梯度的 1/N

更新：
  ⑦ 每个 GPU 用本地的参数分片 + 梯度分片 + 优化器状态分片进行更新
```

![FSDP 实际工作情况](<../images/8-15-FSDP的实际工作情况.png>)

#### 4.3 FSDP vs DDP 选择指南

```
什么时候用 DDP：
  ✅ 模型能放入单卡内存
  ✅ 通信开销不是瓶颈
  → 简单、高效、代码改动最小

什么时候用 FSDP：
  ✅ 模型太大，单卡内存不够
  ✅ 需要训练 7B+ 参数的模型
  → 内存节省显著，但通信量更大
```

---

### 五、ZeRO 实际效果

![ZeRO 实际工作情况对比](<../images/8-16-ZeRO的实际工作情况.png>)

```
关键发现：

  ZeRO-1 和 ZeRO-2 通信量与 DDP 相同 → "免费"的内存节省
  ZeRO-3 通信量增加约 1.5× → 需要权衡内存 vs 通信时间
  ZeRO-3 + 激活重计算 → 可以训练 3-4× 大的模型

实际选择（业界经验）：
  LoRA/QLoRA 微调：ZeRO-2 通常比 ZeRO-3 更快（可训练参数少）
  全参数训练：ZeRO-3 是必需的
  超大模型：ZeRO-3 + 激活重计算 + 混合精度
```

> 🌐 **补充（Web Search / 业界实践）**：据 PyTorch 官方 FSDP2 教程，FSDP2 引入了更细粒度的分片策略和更好的通信-计算重叠。在 LoRA 微调场景中，由于可训练参数很少，ZeRO-2 往往比 ZeRO-3 更快——因为 ZeRO-3 的 All-Gather 开销不值得为冻结参数付出。
>
> 来源：[PyTorch FSDP2 Tutorial](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html)、[DeepSpeed ZeRO Tutorial](https://www.deepspeed.ai/tutorials/zero/)

> 💡 **补充（Context7 / DeepSpeed）**：
>
> **DeepSpeed ZeRO 各阶段配置示例**：DeepSpeed 通过 JSON 配置文件控制 ZeRO 行为。以下是三个阶段的典型配置参数：
> ```json
> {
>   "zero_optimization": {
>     "stage": 2,
>     "reduce_bucket_size": 5e8,
>     "allgather_bucket_size": 5e8,
>     "overlap_comm": true,
>     "contiguous_gradients": true
>   }
> }
> ```
> 关键参数说明：
> - `reduce_bucket_size`：梯度分桶大小（默认 500MB），控制 Reduce-Scatter 的粒度
> - `allgather_bucket_size`：参数收集的分桶大小，影响 All-Gather 效率
> - `overlap_comm`：是否将通信与计算重叠（类似 DDP 的梯度桶机制）
> - `contiguous_gradients`：是否使用连续内存存储梯度，减少内存碎片
>
> **ZeRO-Infinity（Stage 3 + 卸载）**：ZeRO-3 还支持将优化器状态和参数卸载到 CPU 内存甚至 NVMe SSD，突破 GPU 内存限制：
> ```json
> {
>   "zero_optimization": {
>     "stage": 3,
>     "offload_optimizer": { "device": "cpu", "pin_memory": true },
>     "offload_param": { "device": "cpu", "pin_memory": true }
>   }
> }
> ```
> 这使得在有限 GPU 内存下也能训练超大模型，但代价是 CPU-GPU 数据传输开销。

> 💡 **补充（Context7 / PyTorch DDP）**：
>
> **DDP Communication Hooks**：PyTorch DDP 支持通过 `register_comm_hook` 自定义梯度通信策略。默认的 All-Reduce 可以替换为压缩通信（如 PowerSGD），在通信带宽受限的场景下显著加速训练：
> ```python
> from torch.distributed.algorithms.ddp_comm_hooks import default_hooks as default
> model.register_comm_hook(state=None, hook=default.fp16_compress_hook)
> ```
> `fp16_compress_hook` 将梯度压缩为 FP16 后再通信，通信量减半但精度损失极小。还有 `powerSGD_hook` 等更激进的压缩方案，适合带宽极度受限的跨节点场景。
>
> 来源：[PyTorch DDP Communication Hooks](https://docs.pytorch.org/docs/stable/ddp_comm_hooks.html)

---

## 🧠 本模块问题

请在下方回答以下问题后，输入 `提交作业` 提交。

**Q1**：一个 13B 参数的模型使用 Adam 优化器在 8 张 A100（80GB）上训练。请计算：(a) 无 ZeRO 时每卡内存占用；(b) ZeRO-2 时每卡内存占用。80GB 的卡能装下哪种配置？

**Q2**：ZeRO-2 用 Reduce-Scatter 替代 All-Reduce 来同步梯度。请解释为什么 Reduce-Scatter 可以节省内存，同时通信量与 All-Reduce 相同。

**Q3**：在前向传播中，FSDP 需要通过 All-Gather 收集完整参数，计算完后立即释放。为什么不能一直保留完整参数？如果保留会怎样？

---

<!-- 学习者作答区（请在此处填写你的答案） -->

**A1**：

Adam 优化器需要 2 倍 梯度参数量，所以 一个 13B 参数的模型需要占用
13(参数) + 13(梯度) + 13 * 2(优化器) = 13 * 4 * 10^9 * 4 Bytes = 210GB 

因此 无 ZeRO 时，DDP 时每卡显存占用一致，都是 210GB，但是 A100 单卡 只有 80GB，显然做不到。

使用 ZeRO-2 时，即 梯度 和 优化器参数 分片，则 8 卡每卡占用的显存是
( 13 + (13 + 13*2)/8 B )* 4 Bytes ≈ 72GB
即每张卡显存占用约为 72GB

=================

80GB 无 ZeRO 可以装下 4 * #params  * 4Bytes = 80GB
则 #params = 5 B，即使用 Adam 优化器的 A100 最多能装下 5B 大小的模型
使用 ZeRO-2 可以装下 (1 + 3/8)*#params * 4 Bytes = 80GB
则 #params ≈ 14.5 B 大小的模型



**A2**：

All-Reduce = Reduce-Scatter + All-Gather

Reduce-Scatter 每个节点（GPU）只保留部分 参数梯度 切片，并且只在计算每层梯度时使用 Reduce-Scatter 同步每一层梯度，并且在所属 参数梯度切片的 GPU 保留该梯度 avg 后的备份，其他 节点删除这部分梯度信息，所以能节省显存占用。

并且因为 All-Reduce = Reduce-Scatter + All-Gather， 同步所有节点梯度信息更新参数 和 分片更新所有梯度信息，再对其每个节点的参数信息，通信一样。


**A3**：

一直保存参数信息，会一直占用显存，如果多层都保留，那么多层都会占用，并且每个节点都会保留，那就和 ZeRO-2 一样了，参数的占用不会下降，会占用更多的显存，但是后续更新参数、计算梯度时，通信开销会因为有保留而下降。


---

<!-- 教师批改区（提交作业后由导师填写，请勿手动修改） -->

### 📝 批改结果

**Q1 批改**：计算过程基本正确，ZeRO-2 公式推导清晰 — 得得：**8/10**

- ✅ 公式正确：P + G + O = 4Ψ + 4Ψ + 8Ψ = 16Ψ
- ✅ 无 ZeRO：208GB（你说 210GB，数值接近，取决于 Ψ=13B 还是 13.1B）
- ✅ ZeRO-2 公式正确：(4 + 12/8) × 13 × 4 ≈ 71.5GB（你说 72GB ✅）
- ✅ 额外补充了最大模型计算（无 ZeRO 5B，ZeRO-2 约 14.5B），很好
- ⚠️ 小问题："Adam 需要 2 倍梯度参数量"应为"2 倍**参数量**的优化器状态"（m 和 v 各存一份参数量的值，与梯度无关）

<details>
<summary>📖 Q1 参考答案</summary>

**前提公式**：P + G + O = 4Ψ + 4Ψ + 8Ψ = 16Ψ 字节（Adam）

**(a) 无 ZeRO，每卡内存**：
- 16 × 13 × 10⁹ = 208 × 10⁹ 字节 ≈ **208 GB**
- 208 GB >> 80 GB → **A100 装不下**

**(b) ZeRO-2，每卡内存**：
- 公式：4Ψ + (4Ψ + 8Ψ)/N = (4 + 12/N)Ψ
- (4 + 12/8) × 13 × 4 = 5.5 × 52 = **71.5 GB**
- 71.5 GB < 80 GB → **A100 刚好装得下**

**结论**：无 ZeRO 完全无法训练 13B 模型；ZeRO-2 可以，但余量很小（仅剩 ~8.5 GB 给激活值）。

</details>

---

**Q2 批改**：核心思路正确但表述不够精确 — 得分：**6/10**

- ✅ All-Reduce = Reduce-Scatter + All-Gather，公式正确
- ✅ 知道 Reduce-Scatter 后每个 GPU 只保留部分梯度
- ⚠️ "只在计算每层梯度时使用 Reduce-Scatter 同步每一层梯度"表述有误：Reduce-Scatter 不是"每层"执行一次，而是**整个反向传播结束后**对所有参数的梯度统一执行一次
- ⚠️ 核心原因没说清楚：**每个 GPU 只负责更新 1/N 的参数，所以只需要 1/N 的梯度**。省掉了 All-Gather（每人拿完整结果）→ 通信量不变但内存节省

<details>
<summary>📖 Q2 参考答案</summary>

**为什么通信量相同**：
- All-Reduce = Reduce-Scatter + All-Gather
- All-Reduce 的总通信量 = 2(N-1)/N × 数据量
- Reduce-Scatter 的通信量 = (N-1)/N × 数据量
- 但 ZeRO-2 用 Reduce-Scatter 替代了 All-Reduce，省掉了 All-Gather 那半部分
- 实际上：ZeRO-2 的通信量与 All-Reduce **相同**，因为 Reduce-Scatter 和 All-Gather 的通信量各占一半

**为什么能节省内存**：
- 核心前提：ZeRO-2 中每个 GPU 只负责更新 1/N 的参数（优化器状态已分片）
- 既然 GPU 0 只更新 W1，它只需要 ∇W1，不需要 ∇W2, ∇W3, ∇W4
- All-Reduce：每个人得到完整梯度 → 每人存 4 份 → 内存 = 4 份
- Reduce-Scatter：每人只拿自己需要的 1 份 → 内存 = 1 份
- 节省：梯度内存从 Ψ×4 字节 → Ψ×4/N 字节（节省 75%，N=4 时）

</details>

---

**Q3 批改**：核心推理正确（保留参数 = ZeRO-2），但表述不够完整 — 得分：**7/10**

- ✅ "那就和 ZeRO-2 一样了"——**这个判断是对的**，我之前的批改有误，抱歉
- ✅ 保留完整参数 + 梯度/优化器分片 = ZeRO-2 的配置，推理正确
- ⚠️ "通信开销会因为有保留而下降"——这句是对的（少了一次 All-Gather），但应该展开说明
- ⚠️ 缺少定量对比：从 ZeRO-3 退化为 ZeRO-2 具体多占多少内存

<details>
<summary>📖 Q3 参考答案</summary>

**你的推理完全正确**：
- FSDP/ZeRO-3：参数分片 + 梯度分片 + 优化器分片
- 保留完整参数：参数完整 + 梯度分片 + 优化器分片 = **ZeRO-2**
- 不是退化为 DDP（DDP 是三者都完整），而是退化为 ZeRO-2

**保留参数的代价（定量）**：
- FSDP/ZeRO-3：每卡参数内存 = 4Ψ/N（只存 1/N）
- 保留参数后（ZeRO-2）：每卡参数内存 = 4Ψ（完整）
- 以 13B 模型、4 GPU 为例：
  - FSDP：52 / 4 = 13 GB/卡（参数）
  - ZeRO-2：52 GB/卡（参数）
  - 差距：39 GB/卡
- 39GB 的差距直接影响能否训练更大模型

**保留参数的好处**：
- 反向传播不需要再做 All-Gather（参数已经在手）
- 通信量减少（从每层 2 次通信 → 每层 1 次）
- 实际上就是 ZeRO-2 的优势：通信与 DDP 相同，但节省了优化器和梯度内存

**权衡**：
- 如果显存够 → ZeRO-2（保留参数，少通信）
- 如果显存不够 → FSDP/ZeRO-3（释放参数，多通信但省内存）

</details>

---

**综合评价**：Q1 计算掌握良好；Q2 表述需更精确；Q3 推理正确（保留参数 = ZeRO-2），是我之前的批改有误，已更正。整体理解扎实，继续加油。

**批改时间**：2026-05-11
