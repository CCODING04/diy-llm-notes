# 第 8 章 学习笔记

> 记录学习过程中的临时提问与解答，供复习参考。

---

## 交叉验证记录

### Context7 查询（2026-05-08）

本章涉及的主要库：PyTorch (`/websites/pytorch_stable`)、DeepSpeed (`/deepspeedai/deepspeed`)

**查询 1 — PyTorch 分布式通信**：
- DDP Communication Hooks：`register_comm_hook` 自定义梯度通信策略（如 `fp16_compress_hook` 压缩梯度通信量减半）
- `torch.profiler` 分析集合通信性能：在通信操作前后添加 profiler 上下文，按 CUDA 时间排序查看通信占比
- NCCL 后端检查：`torch.distributed.is_nccl_available()` 用于编写可移植脚本
- 补充位置：module1.md（profiler + NCCL 检查）、module2.md（Communication Hooks）、module5.md（profiler 示例）

**查询 2 — DeepSpeed ZeRO**：
- ZeRO 各阶段 JSON 配置：`reduce_bucket_size`、`allgather_bucket_size`、`overlap_comm`、`contiguous_gradients`
- ZeRO-Infinity 卸载：`offload_optimizer` / `offload_param` 支持卸载到 CPU 或 NVMe SSD
- `deepspeed.zero.Init()` 上下文管理器：在 ZeRO-3 下初始化模型参数时自动分片
- 补充位置：module2.md（ZeRO 配置 + 卸载示例）

---

## 模块 1：通信原语与硬件架构 — QA 记录
> 📅 2026-05-09

**Q**：Ring All-Reduce 延迟 N-1 步以及 Tree All-Reduce 延迟 log(N) 步如何计算得到的？

**A**：
- Ring：N 个 GPU 排成环形，每步每个 GPU 只能从 1 个邻居收到数据。从"只有自己"到"看到所有人"，最少 N-1 步。
- Tree：二叉树结构，每步参与节点数翻倍（指数增长）。Reduce 阶段 log₂(N) 步，Broadcast 阶段 log₂(N) 步，共 2×log₂(N) 步。
- 对比：N=4 时 Ring 3 步 vs Tree 4 步；N=64 时 Ring 63 步 vs Tree 12 步。

---

**Q**：大数据为什么 Ring 更优？Ring 步数更多延迟不是更大吗？

**A**：
关键在于每步传输的数据量：Ring 每步每条链路只传 D/N，所有 N 条链路同时工作（100% 利用率）。Tree 每步也是传 D/N，但受树形依赖限制，同一时刻只有部分链路工作（~95% 利用率）。
- Ring 总时间 ≈ (N-1)×D/(N×B)，所有链路满载
- Tree 总时间 ≈ 2×log₂(N)×D/(N×B)，但利用率低 ~5%
- 大数据：带宽是瓶颈 → Ring 100% 利用率更优
- 小数据：延迟是瓶颈 → Tree 步数少更优
NCCL 根据消息大小自动选择：大消息走 Ring，小消息走 Tree。

---

**Q**：PCIe Gen4 `~32 GB/s (x16)` 是什么意思？

**A**：
`(x16)` 表示 16 条 PCIe 通道并行。PCIe Gen4 单通道速率 16 GT/s，有效带宽约 1.97 GB/s，x16 = 1.97 × 16 ≈ 32 GB/s。GPU 通常用 x16 插槽。对比 NVLink 600 GB/s，差距约 20 倍，所以 GPU 间通信必须走 NVLink 而非 PCIe。

---

**Q**：NVLink 和 NVSwitch 有什么区别？

**A**：
- NVLink：GPU 之间的物理直连链路（点对点），每条 25 GB/s，A100 有 12 条 = 600 GB/s
- NVSwitch：交换芯片，汇聚所有 NVLink，实现 8 GPU 全互联（任意 GPU 对之间 600 GB/s）
- 类比：NVLink = 一条高速公路，NVSwitch = 立交桥枢纽
- 没有 NVSwitch：GPU 只能直连少数邻居，远距离通信需多跳转发
- 有 NVSwitch：任意 GPU 一跳直达，独享满带宽

---

**Q**：`如果 NCCL 不可用，可以回退到 Gloo 后端`，什么是 Gloo 后端？

**A**：
Gloo 是 Meta 开发的集合通信库，主要为 CPU 环境设计。PyTorch 支持三种分布式后端：
- NCCL：NVIDIA 专用，GPU 通信最优（NVLink 600 GB/s）
- Gloo：CPU/GPU 通用，GPU 通信慢 10-30 倍（数据经 CPU 中转）
- MPI：经典科学计算标准
Gloo 的价值：在没有 GPU 的环境下（本地调试、CPU 训练）能保底运行。生产环境必须用 NCCL。

---

## 模块 2：数据并行与 ZeRO/FSDP — QA 记录
> 📅 2026-05-11

**Q**：Zero-2 中梯度分片没有理解清楚，请给我用简单易懂的方式配合 ASCII 图示来说明。

**A**：
用 4 GPU、4 参数举例：
1. 反向传播后，每个 GPU 独立算出完整梯度（∇W1, ∇W2, ∇W3, ∇W4）
2. Reduce-Scatter：GPU 0 只保留 avg(∇W1)，GPU 1 只保留 avg(∇W2)，...
3. 因为 GPU 0 只负责更新 W1，所以不需要其他梯度
4. 对比 All-Reduce = Reduce-Scatter + All-Gather：ZeRO-2 只做前半部分，省掉 All-Gather
5. 结果：每 GPU 梯度内存从 4 份 → 1 份，节省 75%

---

**Q**：Zero-2 中反向传播"临时占用 4 份内存"，这 4 份梯度各不相同，是由 All-Gather 后的 avg(loss) 计算得到的不同梯度吗？

**A**：
不是。4 份梯度的来源更简单：就是 4 张卡各自用不同数据独立反向传播算出来的。
- 每个 GPU 处理不同的数据批次 → loss 不同 → 梯度不同
- 整个前向/反向过程没有通信，完全独立
- 最后 Reduce-Scatter 对 4 份梯度取平均 → 等效于用全局 batch size 计算的梯度
- 不存在"All-Gather loss"的操作，loss 不会跨 GPU 平均

---

<!-- 学习过程中追加 QA 记录 -->

## 模块 3：流水线并行 — 正式批改记录
> 📅 2026-05-13

**Q1 批改**：时间线和朴素利用率完全正确，微批次调度图画的是 GPipe 风格，依赖链和气泡率公式正确 — 得分：**9/10**

- ✅ 朴素层并行时间线正确：GPU0 在 T1 算 F、T8 算 B，GPU3 在 T4 算 F、T5 算 B
- ✅ 利用率 2/8 = 25% 正确
- ✅ 微批次时间线正确（GPipe 风格：全前向后全反向）
- ✅ 气泡率公式 `(N-1)/(M+N-1) = 3/7 ≈ 42.85%` 正确
- ⚠️ 没明确写出利用率：气泡率 42.85% → 利用率 = 1 - 42.85% = **57.1%**（从 25% 提升到 57.1%）

**Q2 批改**：核心概念混淆，把 1F1B 和 Interleaved 1F1B（VPP）搞混了 — 得分：**4/10**

- ❌ "1F1B 每个设备可以处理多个不连续的层"——这是 **Interleaved 1F1B / VPP** 的特征，不是普通 1F1B
- ❌ 普通 1F1B 的 GPU 仍然只负责一组连续层，优势在于 **F/B 交替调度降低峰值激活内存**
- ⚠️ GPipe 的内存问题描述部分正确（微批次越多内存越高），但原因是"所有前向做完才能反向，需保留全部激活"
- ⚠️ 缺少关键对比：1F1B 的峰值激活内存由流水线深度 P 决定（稳定阶段只有 P 个激活在手），不像 GPipe 随 M 线性增长

**Q3 批改**：核心推理正确，公式表述有小错 — 得分：**8/10**

- ✅ ∇X 必须立即计算（上一层需要）— 正确
- ✅ ∇W 可以延后（参数更新前完成即可）— 正确
- ✅ "延迟计算对结果无影响"— 正确，因为参数在 global batch 结束时才更新
- ⚠️ 公式 `∇X = ∇Y/∇W_T` 不准确，应为 `∇X = ∇Y @ W`（链式法则的矩阵乘法，不是除法）

**综合评价**：Q1 掌握扎实；Q3 核心理解正确但公式需修正；Q2 存在概念混淆（1F1B vs VPP），建议复习 2.3 节中 1F1B 的 F/B 交替机制和激活内存释放逻辑。

---

## 模块 4：张量并行、序列/上下文并行与 3D 并行 — 正式批改记录
> 📅 2026-05-14

**Q1 批改**：逻辑链完整，从 W1 列切 → GELU 逐元素 → W2 行切 → 求和 → All-Reduce，覆盖了核心推理。矩阵形状推导正确。有一点需要修正：你提到"GELU 可以使用 pointwise 的 tanh 来近似"，这个说法不准确——GELU 本身就是逐元素函数（element-wise），不需要"近似"才能分开计算，直接就可以在各 GPU 上独立完成。tanh 只是 GELU 的一种数值近似实现方式，与"能否分开计算"无关。 — 得分：**9/10**

- ✅ W1 列切分逻辑正确
- ✅ GELU 逐元素可分开计算 — 正确
- ✅ W2 行切分 + 求和 → All-Reduce — 正确
- ⚠️ "GELU 可以使用 pointwise 的 tanh 来近似"——GELU 本身就是逐元素函数，不需要近似才能分开计算

**Q2 批改**：CP 和 Ring Attention 的描述基本正确。但 SP 的描述有明显偏差：你说 SP "适用于上下文特别长的场景"，这是把 SP 和 CP 混淆了。SP 的核心定位是**配合 TP 使用**，切分的是 LayerNorm、Dropout 等 TP 无法拆分的激活，而不是为了解决长上下文问题。Ring Attention 的"向右发送数据"说法也不够准确——K/V block 沿环形拓扑传递，方向不限，核心是 blockwise attention + 通信计算重叠。 — 得分：**7/10**

- ❌ SP "适用于上下文特别长的场景"——SP 是配合 TP 使用的，切分 LayerNorm/Dropout 激活，不是解决长上下文
- ✅ CP 沿 Sequence 维度切分，比 SP 更激进 — 正确
- ✅ Ring Attention 是 CP 的一种通信方案 — 基本正确
- ⚠️ "向右发送数据"不够准确，K/V block 沿环形拓扑传递，方向不限

**Q3 批改**：推导过程完整，Step 1-5 的计算逻辑正确，8K 和 32K 两个场景都给出了合理配置。两个小问题：① 第 798 行和 808 行 `s = 8192` 写成了 `8292`（笔误）；② 32K 场景的推理逻辑有跳跃——先说"只能使用完全重计算"，然后突然跳到"CP=2, DP=4"，没有解释为什么选择 CP=2 而不是继续用完全重计算 + CP=1。 — 得分：**8/10**

- ✅ Step 1 模型内存估算 840GB — 正确
- ✅ Step 2 TP=4 推导 — 正确
- ✅ Step 3 PP=4 推导 + bubble 公式 — 正确
- ✅ Step 4 激活内存计算逻辑 — 正确
- ⚠️ 笔误：`8292` 应为 `8192`（出现两次）
- ⚠️ 32K 推理链跳跃：未解释为什么选 CP=2 而非完全重计算+CP=1

**综合评价**：三道题的核心知识点都掌握了，Q1 的 MLP TP 推导尤其扎实。主要薄弱点是 Q2 中 SP 的定位（容易和 CP 混淆）和 Q3 中推理链的完整性（跳跃了中间判断步骤）。建议复习第五节 SP 部分，重点理解 SP 与 TP 的配合关系。

---

## 模块 4：张量并行、序列/上下文并行与 3D 并行 — 课下 QA 记录
> 📅 2026-05-14

**Q**：文档中 `[B, S, H]` 的 activation 和 `[B, n_heads, S, S]` 这两部分，一个是 MLP 层的 activation 一个是 attention 内的 activation 吗？

**A**：对，本质上是同一层（Transformer block）中两个不同子模块的激活：
1. `[B, S, H]` — 是每个激活张量的通用尺寸，MLP 和 Attention 子模块的激活都包含多个 `[b,s,h]` 张量。公式里的 34 就是把这些张量加起来的总系数。
2. `[B, n_heads, S, S]` — 是注意力计算中显式存储 attention score 矩阵的额外开销。这个矩阵尺寸是 `[batch, num_heads, seq_len, seq_len]`，与 `[b,s,h]` 完全不同维度，是 attention 独有的（MLP 没有这个）。

---

**Q**：`[B, n_heads, S, S]` 是 transformer 中哪个模块的内存开销？

**A**：这是 Multi-Head Self-Attention (MHSA) 模块内部的开销。存在于 QK^T 点乘之后、softmax 之前。计算流程：
```
Q, K reshape → [B, n_heads, S, d_head]
attention score = Q @ K^T → [B, n_heads, S, S]  ← 这个矩阵
softmax(score) → [B, n_heads, S, S]
output = softmax(score) @ V → [B, n_heads, S, d_head]
```
这个矩阵两个维度都是 S（序列长度），随序列长度平方增长。选择性重计算的核心收益就是不存这个矩阵。

---

**Q**：Step 5 中 128 个 GPU 总数是如何得到的？

**A**：128 是已知的硬件约束（问题设定：16 节点 × 8 GPU = 128），不是推导出来的。Step 5 的公式是用已知的总 GPU 数反推 DP。反过来，如果先确定了所需的 DP，也可以推算所需总 GPU 数：`总 GPU = TP × PP × CP × DP`。

---

**Q**：文档中 `8/3·h ≈ 21843` 和 `28672` 两个值不一致，哪里有问题？

**A**：`8/3·h` = 2.67h ≈ 21845，而 `28672` = 3.5h。两者不同。Meta 源码中 FFN 中间维度的计算过程：
```
4 × dim = 32768
× 2/3  → int(2 × 32768 / 3) = 21845     ← 8/3·h 的来源（7B/13B 默认值）
× 1.3  → int(1.3 × 21845) = 28398       ← 70B 特有的 ffn_dim_multiplier = 1.3
取整到 256 的倍数 → 28672                ← 最终值 = 3.5h
```
来源：Meta 官方 `meta-llama/llama` 仓库 `model.py` 第 307-348 行。

---

**Q**：选择性重计算和完全重计算的内存公式有什么区别？

**A**：
- 选择性重计算（Selective）：只丢弃注意力 score（去掉 `5·a·s/h` 项），保留其余 34 个 `[b,s,h]` 激活。公式：`sbh(34/t)`
- 完全重计算（Full）：只保留每层的输入激活（1 个 `[b,s,h]`），其余全部反向时重算。公式：`sbh(2/t)`（2 = 输入 + 残差连接）

选择性重计算是"省一部分"，完全重计算是"几乎全省"。代价是反向传播计算量增加更多（约 30-40%），但大模型训练中内存瓶颈通常比算力瓶颈更致命。

---

## 模块 5：实践代码与总结 — 正式批改记录
> 📅 2026-05-15

**Q1 批改**：推导过程正确（Reduce-Scatter + All-Gather 各 (N-1)/N），但**数值计算错误**——把 `/N` 写成了 `×N`。 — 得分：**6/10**

- ✅ All-Reduce = Reduce-Scatter + All-Gather — 正确
- ✅ 每步通信量 (N-1)/N × D 的推导逻辑正确
- ❌ 数值计算：`2×(N-1)/N` 是**归一化因子**（无量纲），不是总通信量
  - 4 GPU：`2×(4-1)/4 = 2×3/4 = 1.5`，不是 24
  - 8 GPU：`2×(8-1)/8 = 2×7/8 = 1.75`，不是 112
  - 你算成了 `2×(N-1)×N`（乘以 N），应为 `2×(N-1)/N`（除以 N）
- ⚠️ 这个因子的物理含义：表示 All-Reduce 的总数据移动量是原始数据量的多少倍。N 越大越接近 2（极限为 2D），N=4 时为 1.5D，N=8 时为 1.75D

<details>
<summary>📖 Q1 参考答案</summary>

`2×(N-1)/N` 的含义：

Ring All-Reduce 由两个阶段组成：
1. **Reduce-Scatter**：每个 GPU 沿环发送 (N-1) 轮，每轮发 D/N 数据 → 每 GPU 发送量 = (N-1)/N × D
2. **All-Gather**：每个 GPU 沿环发送 (N-1) 轮，每轮发 D/N 数据 → 每 GPU 发送量 = (N-1)/N × D

总通信量（每 GPU）= 2 × (N-1)/N × D

这是一个**无量纲的归一化因子**，表示 All-Reduce 的总数据移动量是原始数据量 D 的多少倍：
- N=4：2×(4-1)/4 = **1.5**（总通信量是数据量的 1.5 倍）
- N=8：2×(8-1)/8 = **1.75**（总通信量是数据量的 1.75 倍）
- N→∞：极限为 **2**（无论多少 GPU，通信量最多翻倍）

这个因子的关键意义：Ring All-Reduce 的通信量与 GPU 数量几乎无关（只影响 1/N 的尾项），这就是它被称为"带宽最优"算法的原因。

来源：Patarasuk & Yuan (2009)
</details>

---

**Q2 批改**：对流水线并行的分析正确。对张量并行的修正（MLP 用 All-Reduce 而非 All-Gather）也是对的，说明你理解了 TP 的通信模式。但**数据并行的分析有误**——你说 DP 用 All-Gather 而非 All-Reduce，这是错的。 — 得分：**7/10**

- ✅ 流水线并行用 Send/Recv（点对点传递激活值）— 正确
- ✅ 张量并行 MLP 层用 All-Reduce（列切+行切后求和）— 正确，且正确指出了题目的简化
- ❌ 数据并行用 All-**Reduce**，不是 All-Gather
  - DP 的核心操作：反向传播后，所有 GPU 的梯度需要取平均 → All-Reduce
  - 你提到的"参数同步"是梯度更新**之后**的事（用 Broadcast 即可），不是 DP 的主要通信
  - DDP 的关键就是在反向传播过程中自动插入 All-Reduce

<details>
<summary>📖 Q2 参考答案</summary>

三种并行使用不同通信原语，根本原因是**数据切分方式不同**：

**数据并行 → All-Reduce（梯度归约）**
- 每卡有完整模型，数据按 batch 切分
- 反向传播后每卡有独立梯度，需要全局平均
- All-Reduce = 所有卡的梯度求平均，每卡得到相同结果

**张量并行 → All-Reduce（MLP）/ All-Gather（部分场景）**
- MLP：W1 列切 → GELU → W2 行切 → 两卡结果求和 → All-Reduce
- 注意力：Q/K/V 按头切分，输出可能需要 All-Gather 收集
- 通信频率高（每层至少一次）

**流水线并行 → Send/Recv（点对点）**
- 模型按层切分，相邻 stage 传递激活值
- 只有相邻 GPU 之间通信，不需要全员参与
- 通信频率低（每个微批次一次），但有气泡开销

通信原语的选择由参与通信的 GPU 数量决定：
- 全员参与 → 集合通信（All-Reduce, All-Gather）
- 只有邻居参与 → 点对点（Send/Recv）
</details>

---

**Q3 批改**：方向正确但**深度不够**。PyTorch 显式 vs JAX 声明式的对比基本正确，但 DeepSeek 优化 NCCL 的原因可以更具体。 — 得分：**6/10**

- ✅ PyTorch 显式编程，便于定位通信问题 — 正确
- ✅ JAX 声明式，XLA 自动编排，调试困难 — 正确
- ⚠️ DeepSeek 优化 NCCL 的原因过于笼统（"挖掘有限资源"），缺少具体技术动机

<details>
<summary>📖 Q3 参考答案</summary>

**PyTorch 显式编程**
- 优点：通信逻辑透明，可以用 profiler 逐操作分析，调试定位方便
- 缺点：代码冗长，需要手动管理通信，容易出错

**JAX 声式编程**
- 优点：只声明分片策略（PartitionSpec），编译器自动编排通信，代码简洁
- 缺点：通信逻辑对用户不透明，出问题时难以定位是计算还是通信的 bug

**DeepSeek 需要深入 NCCL 层级的原因**：

这不仅仅是"优化有限资源"，而是因为高层 API 存在**无法突破的性能天花板**：
1. **通信/计算重叠**：DDP 默认的 All-Reduce 在反向传播结束后才开始，但手动插入 NCCL 通信钩子（如梯度分桶 + 异步 All-Reduce）可以实现"边算边传"
2. **混合精度通信**：默认 All-Reduce 传输 FP32 梯度，但可以用 FP16/BF16 传输再在接收端恢复，通信量减半
3. **拓扑感知路由**：NCCL 默认的 Ring/Tree 算法不一定是最优的，针对特定集群拓扑（如特定 NVLink 连接方式）定制通信策略可以提升 10-20% 吞吐
4. **自定义归约操作**：标准 All-Reduce 做 SUM/AVG，但某些场景需要自定义归约逻辑（如梯度裁剪 + 归约的融合操作）

这些优化在 PyTorch/JAX 的高层 API 中无法实现，必须深入到 NCCL 甚至更底层。
</details>

---

**综合评价**：Q1 的推导逻辑扎实但数值计算有粗心错误（/N 写成 ×N）；Q2 对 TP 的理解比题目本身更准确（正确指出 All-Gather 是简化说法），但 DP 的核心通信搞混了；Q3 方向对但缺少具体技术细节。本模块的核心知识点（硬件层级、NCCL、通信基准测试）在前面的学习中已经掌握较好，建议重点复习 DP 的梯度同步机制。
