# 第 10 章：推理 — 模块 1：推理基础（训练 vs 推理、Transformer 推理流程、算术强度）

> 📍 学习进度：第 10 章，第 1 / 3 模块
> 📅 生成时间：2026-05-21

---

## 学习目标

- 理解训练与推理在目标、计算模式、资源瓶颈上的本质差异及其联系
- 掌握 Transformer 推理的两阶段流程（Prefill + Decode）及 KV Cache 的工作原理
- 理解线性注意力机制的核心思想、前缀累加的因果性保证，以及 MiniMax 混合架构
- 掌握算术强度的计算方法，理解推理为何受内存带宽限制

---

## 核心内容

### 一、推理与训练的差异

训练与推理都涉及模型的前向计算，但目标、计算模式和资源瓶颈完全不同。

假设生成第 $i$ 个 token，训练和推理的统一数学形式为：

$$P(\text{token}_i \mid \text{token}_1, \text{token}_2, \dots, \text{token}_{i-1})$$

**关键差异在于 $token_i$ 前文的来源**：

| 阶段 | 前文来源 | 并行性 | 瓶颈 |
|------|---------|--------|------|
| **训练** | 真实标签（ground-truth，即训练数据中的正确答案），使用 teacher-forcing（每步用正确答案的前序 token 作为输入，而非模型自己预测的 token，相当于老师每步都"强行"给出正确答案） | 整个序列可并行（causal mask 因果掩码保证每个位置只看前面 token，不"偷看"未来） | **算力**（FLOPs 即浮点运算次数、通信量） |
| **推理** | 模型先前预测的 token（auto-regressive 自回归方式，每步依赖自己上一步的输出） | 时间维度严格顺序，每步依赖上一步 | **显存与带宽**（KV Cache 访问） |

**训练阶段**：输入完整目标序列，使用因果掩码（causal mask）——一种注意力掩码，将未来位置的注意力分数设为 $-\infty$，softmax 后变为 0，从而保证每个位置只能访问前面 token。由于输入已知，整个序列可通过一次或少量大矩阵乘法**并行计算所有位置**。中间激活保留用于反向传播。

**推理阶段**：使用固定参数生成输出，采用自回归（auto-regressive）方式——每步依赖自己上一步的输出作为输入，逐步生成。每生成一个 token 需要参考之前所有 token 的 KV Cache（Key-Value 缓存，存储历史 token 的键值向量，避免重复计算）。随上下文增长，显存和带宽成为瓶颈。

> 💡 **补充（Context7）**：训练时使用 teacher-forcing 的两个核心原因：① 提供稳定监督信号便于快速收敛；② 允许整段序列并行计算。如果用模型预测的 token 作为输入，采样操作的非微分性会破坏标准反向传播。

**训练 vs 推理并行对比**：

```
训练阶段（全序列并行，使用 ground-truth 前文）
输入序列：   x1          x2           x3            x4
模型前向：  f(x1)     f(x1,x2)   f(x1,x2,x3)  f(x1..x4)
输出序列：   y1          y2           y3            y4
→ 所有 y_i 可并行计算，y_i 仅用于计算损失，不作为下一步输入

推理阶段（自回归生成，逐步依赖模型生成的前文）
Step 1:  y1 = f(x1)
Step 2:  y2 = f(x1, y1)
Step 3:  y3 = f(x1, y1, y2)
→ 每步必须等上一步完成，y_i 用作下一步输入
```

**推理阶段的特点**：时间维度是**顺序的**（自回归约束），计算维度是**并行的**（矩阵运算加速）。

### 二、训练与推理的联系

训练与推理并非独立阶段，而是紧密关联的系统循环：

1. **训练的根本目的是优化推理行为**：数据选择、损失函数、正则化等设计最终都影响推理表现
2. **推理本身贯穿训练周期**：模型验证、RLHF（Reinforcement Learning from Human Feedback，基于人类反馈的强化学习，用于对齐模型输出与人类偏好）的奖励计算等都依赖推理
3. **结构设计决定推理的可优化性**：
   - Transformer 结构、注意力复杂度、MoE（Mixture of Experts，混合专家模型——多个 FFN 专家并行存在，Router 每次只激活少量专家以降低计算量）专家布局等决定推理显存与延迟
   - 是否能用 FlashAttention、长上下文机制等取决于训练阶段的结构设计

> 🌐 **补充（Web Search / Exa）**：2025 年出现的新趋势——**分离式推理服务（Disaggregated Serving）**，将 Prefill 和 Decode 分配到不同 GPU 池，Prefill 节点用大 batch 充分利用算力，Decode 节点专注于低延迟的逐 token 生成（来源：Nexus 论文、llm-d 项目）。

### 三、Transformer 推理的两阶段流程

![图10.1 Transformer推理过程](<../images/10.1.png>)

Transformer 自回归推理分为**两个阶段**：

**阶段 1：Prefill（预填充）**

一次性处理用户输入的所有 prompt token：
- 每一层、每个注意力头计算这些 token 的 Key 和 Value
- 将 $(K, V)$ 存入 **KV Cache**（Key-Value 缓存，存储已计算 token 的键值向量，后续生成时直接复用，避免重复计算）
- 这是一个**compute-bound**（计算密集型，瓶颈在 GPU 算力而非数据搬运）操作——大矩阵乘法充分利用 GPU 算力

> 🌐 **补充（Web Search / Tavily）**：Llama 70B 在 H100 上 Prefill 阶段可达 92% 的计算利用率，是典型的 compute-bound 操作（来源：Towards Data Science, arXiv 2512.22066）。

**阶段 2：Decode（逐 token 生成）**

每步只处理 1 个新 token：
- 对新 token 计算 Query（Q）
- Q 与 KV Cache 中所有已缓存的 K 做点积 → 注意力分数
- 用注意力分数对 V 做加权求和 → 输出
- 新 token 的 $(K, V)$ 追加到 KV Cache
- 输出 logits（模型最后一层输出的未归一化分数）经 softmax → 词汇表概率分布 → 采样下一个 token

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^{\top}}{\sqrt{d_k}}\right)V$$

![图10.2 自回归Transformer推理](<../images/10.2.png>)

**图 10.2 解读**：左侧展示的是无 KV Cache 的朴素采样（Naive Sampling），每生成一个新 token 都要对所有前文重新计算 Q、K、V——大量重复计算；右侧是使用 KV Cache 的高效推理：Prefill 阶段一次性计算并缓存所有 prompt 的 K、V，Decode 阶段只计算新 token 的 Q，与缓存的 K、V 做注意力计算。

**KV Cache 的显存占用**大致与 `token数 × 层数 × 注意力头数 × 隐藏层维度` 成正比。

> 🌐 **补充（Web Search / Exa）**：一个 70B 模型在 8K 上下文下，单个请求的 KV Cache 约需 20GB；batch 为 32 时达到约 640GB。KV Cache 在实际推理中常**超过模型权重本身**的显存占用（来源：Introl KV Cache Optimization Guide, Dec 2025）。

**KV Cache 的适用范围**：

| 场景 | 是否适合 KV Cache | 原因 |
|------|------------------|------|
| 自回归生成（GPT/LLaMA） | ✅ 适合 | 过去 token 的 KV 不会被未来 token 改变 |
| 双向注意力（BERT） | ❌ 不适合 | 任一 token 变动影响其他 token 的 QK 关系 |
| 扩散模型 | ⚠️ 部分适用 | 条件编码器的 KV 可在扩散步骤间复用，但不会随序列长度累积增长 |

#### KV Cache 推理简易代码

来自 [推理.md](../推理.md) 的代码示例，展示自回归生成 + KV Cache 的核心逻辑：

```python
def generate_kv_readable(prefix, max_len=10, min_len=3):
    ids = [token2id[w] for w in prefix]
    K_cache = np.zeros((0, d_model))  # 空 KV Cache
    V_cache = np.zeros((0, d_model))

    for step in range(max_len):
        last_id = ids[-1]
        x = E[last_id]
        q, k, v = x, x, x  # 简化: Q=K=V=embedding

        # 追加新 token 的 KV 到缓存
        K_cache = np.vstack([K_cache, k])
        V_cache = np.vstack([V_cache, v])

        # 注意力计算：Q 与所有历史 K 做点积
        att_scores = K_cache @ q          # 维度: (已生成token数,)
        att_scores /= np.sqrt(d_model)
        att_weights = softmax(att_scores)
        context = att_weights @ V_cache   # 加权求和

        # 用手动 logits 预测下一个 token
        logits = manual_logits[last_id]
        probs = softmax(logits)
        next_id = int(np.argmax(probs))
        # ...
```

> 代码片段依赖外部定义的 `vocab`、`E`（embedding矩阵）、`manual_logits` 等变量，用于说明 KV Cache 的逐步累积逻辑，不是完整可运行脚本。

### 四、线性注意力与 MiniMax 混合架构

标准 Softmax 注意力的推理复杂度为 $O(n^2 d)$，随序列长度平方增长。线性注意力通过改变计算顺序将复杂度降为 $O(n d^2)$。

![图10.5 全局注意力机制 vs 线性注意力机制](<../images/10.5.png>)

**图 10.5 解读**——两种注意力的计算路径对比：

| | Softmax Attention（全局） | Linear Attention（线性） |
|---|---|---|
| 计算顺序 | 先 $Q \times K^\top$ → softmax 归一化 → 再 $\times V$ | 先 $K^\top \times V$ → 再 $Q \times$ |
| 因果性保证 | 训练时用 **causal mask**：将 $QK^\top$ 矩阵的上三角（未来位置）设为 $-\infty$，softmax 后变为 0，确保每个 token 只看到前面 | 推理时用**前缀累加**：$S$ 只累加到当前位置，天然不包含未来信息 |
| 中间矩阵 | $N \times N$（依赖序列长度） | $d \times d$（仅依赖特征维度） |
| 时间复杂度 | $O(N^2 d)$ | $O(N d^2)$ |
| 空间复杂度 | $O(N^2)$ | $O(d^2)$ |

**核心思想**：当 $N \gg d$（长序列）时，线性注意力的 $O(Nd^2)$ 远优于 $O(N^2 d)$。

> 💡 **关于 mask 的补充**：图 10.5 中 Softmax Attention 的流程简化了 mask 步骤。实际训练时的完整流程是：$QK^\top$ → **加上 causal mask（上三角设为 $-\infty$）** → softmax → $\times V$。这个 mask 是标准 Softmax 注意力在训练时保证因果性的关键——没有它，每个 token 会"看到"所有其他 token（包括未来的）。而线性注意力则通过前缀累加天然实现了因果性，不需要显式 mask。

**但有一个关键难点——因果性**。直接用 $\phi(Q)(K^\top V)$ 会把整句话所有 token 的信息混在一起，导致"偷看未来"。解决方案是**前缀累加**：

```python
S = np.zeros((d, d))  # K^T V 累加器
Z = np.zeros((d, 1))  # K 归一化因子累加器
for i in range(L):
    ki, vi, qi = phi(K[i:i+1]).T, V[i:i+1].T, phi(Q[i:i+1]).T
    S += ki @ vi.T     # 只累加到当前位置
    Z += ki
    y_i = (qi.T @ S) / (qi.T @ Z)  # 仅依赖前缀信息
```

代码来自 [推理.md](../推理.md) 的线性注意力示例。`S` 和 `Z` 是增量更新的累加器——每步只加入当前 token 的贡献，确保每个 token 的输出**只依赖已生成的前文**。

**数值验证**（来自原始代码输出，对比"全局计算"和"前缀累加"两种方式的输出差异）：

```
错误方式：一次性计算 K^T V（包含所有 token，含未来）
  → Y_wrong[i] = q_i^T × (k₁v₁ᵀ + k₂v₂ᵀ + ... + k₅v₅ᵀ) / q_i^T × (k₁ + ... + k₅)
  → 第 i 个 token 能看到全部 5 个 token 的信息

正确方式：前缀累加（只累加到当前位置）
  → Y_correct[i] = q_i^T × (k₁v₁ᵀ + ... + kᵢvᵢᵀ) / q_i^T × (k₁ + ... + kᵢ)
  → 第 i 个 token 只看到位置 1~i 的信息

偏差 = Y_wrong[i] - Y_correct[i]（偷看到未来信息后输出变化了多少）：
  token "I"    (位置0) → 偏差 [0.05, -0.196]   ← 偷看到位置1~4共4个未来token，偏差最大
  token "like" (位置1) → 偏差 [-0.097, -0.037]  ← 偷看到位置2~4共3个未来token
  token "deep" (位置2) → 偏差 [-0.058, 0.028]   ← 偷看到位置3~4共2个未来token
  token "😄"  (位置4) → 偏差 [0.0, 0.0]         ← 没有未来token可偷看，两种方式完全一致
```

结论：前缀累加保证因果性，且每步计算量线性增长（而非平方增长）。

> 💡 **常见误解澄清**：线性注意力虽然复杂度低，但由于设计复杂（核函数选择、因果性保证），目前主流开源 LLM（LLaMA3、Qwen2.5、DeepSeek V3）仍未将其作为默认方案。

#### MiniMax 混合注意力架构

MiniMax-Text-01（456B 总参数，45.9B 激活参数）采用了**线性注意力 + Softmax 注意力的混合架构**：每 8 层中，7 层使用 Lightning Attention（线性注意力），1 层使用标准 Softmax 注意力，总共 80 层。

![图10.3 MINMAX混合注意力机制](<../images/10.3.png>)

**图 10.3 解读**：

```
Input Hidden
  ↓
  ├─ M× 重复：Lightning Attention（SiLU核函数 + 门控） → RMSNorm → MoE FFN → RMSNorm
  └─ 1× ：Softmax Attention → RMSNorm → MoE FFN → RMSNorm
  ↓
Output Hidden
```

**Lightning Attention 模块**：
- 输入投影为 Q、K、V、G（门控信号）四个分支
- Q、K 经 SiLU 激活（替代 softmax 作为核函数）
- 逐元素乘法近似 $QK^\top$ 的线性计算
- 门控 G 经 Sigmoid 激活后调制输出

**MoE FFN 模块**：Router（路由器）从 N 个 FFN 专家中选 top-2，稀疏激活降低计算量。MoE 即 Mixture of Experts（混合专家模型），多个 FFN 专家并行存在，每次只激活少量专家。

**推理效率**：

![图10.4 推理时间对比](<../images/10.4.png>)

**图 10.4 解读**：横轴为上下文长度（8K → 1M tokens），纵轴为推理延迟。MiniMax-Text-01（红色实线）在 1M token 时延迟仅约 10,000ms，而 Llama-3-70B（浅紫色）接近 100,000ms——**10 倍差距**。这是因为线性注意力的推理可通过递归更新累积项 $\sum K^\top V$ 实现，无需随序列增长重新计算所有注意力。

> 🌐 **补充（Web Search / Exa）**：MiniMax-Text-01 支持 1M token 训练上下文和 4M token 推理外推，MMLU 得分 88.5（GPT-4o 为 85.7，Claude-3.5-Sonnet 为 88.3）。后续推出的 MiniMax-M1 推理模型在生成时消耗不到 DeepSeek R1 50% 的 FLOPs。

### 五、算术强度分析

**算术强度**（Arithmetic Intensity）是判断一个操作是 compute-bound（瓶颈在 GPU 算力）还是 memory-bound（瓶颈在显存带宽，即数据搬运速度跟不上算力）的核心指标：

$$I = \frac{\text{FLOPs}}{\text{Bytes Transferred}}$$

以矩阵乘法 $X_{B \times D} \times W_{D \times F}$ 为例：

| 组成 | 计算 |
|------|------|
| 读 X | $2 \times B \times D$ 字节（FP16） |
| 读 W | $2 \times D \times F$ 字节 |
| 写 Y | $2 \times B \times F$ 字节 |
| **总传输** | $2(BD + DF + BF)$ |
| **FLOPs** | $2BDF$ |
| **算术强度** | $\frac{2BDF}{2(BD + DF + BF)} = \frac{BDF}{BD + DF + BF}$ |

**当 B=1（推理 Decode 阶段）时的数值推演**：

设 $B=1, D=4096, F=11008$（Llama2-7B 的配置）：

```
FLOPs = 2 × 1 × 4096 × 11008 = 90,177,536
Bytes = 2 × (1×4096 + 4096×11008 + 1×11008) = 2 × (4096 + 45,088,768 + 11008)
      = 2 × 45,103,872 = 90,207,744

I = 90,177,536 / 90,207,744 ≈ 1.0 FLOPs/byte
```

对比 NVIDIA H100 的临界值：

```
H100 峰值算力: 989 TFLOPs/s (FP16)
H100 带宽: 3.35 TB/s
临界值 = 989e12 / 3.35e12 ≈ 295 FLOPs/byte
```

**I ≈ 1.0 ≪ 295** → Decode 阶段是**严重的 memory-bound**——GPU 算力大量空闲，等待显存搬运数据。

**增大 batch 的效果**（"拼车效应"）：

| B | 算术强度 | vs H100 临界值 295 |
|---|---------|-------------------|
| 1 | ≈ 1.0 | memory-bound |
| 16 | ≈ 14.8 | memory-bound |
| 64 | ≈ 56.4 | memory-bound |
| 256 | ≈ 180.3 | memory-bound（接近临界） |
| 512 | ≈ 236.7 | memory-bound（接近临界） |

> 增大 B 后，权重矩阵 W 只读一次，分摊到更多输入——类似"大客车一次拉更多乘客"。但即使 B=512，单操作仍未突破 H100 的临界值。

**推理的两难困境**：
- **"拼车效应"**：增大 batch 提升整体吞吐，但单请求延迟可能增加
- **"串行诅咒"**：自回归生成无法并行，每步需访问全部 KV Cache
- **"内存墙"**：Decode 阶段 GPU 计算单元空闲等待数据搬运，算力再强也无法加速

> 💡 **补充（Context7）**：vLLM 的 **PagedAttention**（类似操作系统虚拟内存的分页机制，将 KV Cache 分成固定大小的页来管理，避免连续内存分配造成的碎片浪费）技术通过分页管理 KV Cache，将内存浪费从传统的 60-80% 降至 <4%，实现 2-4 倍吞吐提升（来源：Introl KV Cache Guide）。这是工程层面缓解内存墙的重要手段。

---

## 🧠 本模块问题

请在下方回答以下问题后，输入 `提交作业` 提交。

**Q1**：推理的 Decode 阶段为何是 memory-bound？请用算术强度的公式和具体数值（B=1, D=4096, F=11008）解释，并与 H100 的临界值比较。

**Q2**：MiniMax 的混合注意力架构中，为什么采用"7 层线性注意力 + 1 层 Softmax 注意力"的交替模式，而不是全部使用线性注意力？请从计算效率和信息建模能力两个角度分析。

**Q3**：线性注意力使用前缀累加（$S += k_i v_i^\top$）来保证因果性。请解释：如果不使用前缀累加，而直接计算全局的 $K^\top V$，会对模型推理产生什么具体影响？结合数值推演示例说明。

---

<!-- 学习者作答区（请在此处填写你的答案） -->

**A1**：

对于 BxD 矩阵 X 和 DxF 矩阵 Y 的矩阵乘法运算
FLOPS 是 2BDF
数据搬运总传输是 2BD + 2DF + 2BD （推理阶段矩阵元素按照 FP16 存储）

则 计算强度 = 2BDF / (2BD + 2DF + 2BD) = BDF/(BD + DF + BD)

因此对于 B = 1, D = 4096, F = 11008 来说
计算强度 = 4096 * 11008 / (4096 + 4096 * 11008 + 11008) ≈ 1.0 FLOPs/Bytes

对于 H100 来说，峰值算力是 989 TFLOPs/s (FP16)
带宽是 3.5TB/s
因此 H100 的计算强度时 989e12 FLOPs/ 3.5e12TB ≈ 282.57 FLOPs/Bytes

因此 计算强度远远没达到H100的计算强度上限，是 memory bound

**A2**：

注意力机制的本质是 $ O = Q K^\top V  $
传统注意力是需要 $ O = Softmax(QK^\top) V $ （简化掉GELU以及casual mask等操作），即 attn 是一个 n x n 的矩阵（Q、K、V 均为 n x d 矩阵）
而线性注意力因为是 $ O = Q(K^\top V) $ ，因此 先生成 S = K^\top V 即 d x d 矩阵，再做 query 操作

两者有关联，但是因为 attn = Softmax(QK^\top) 是 nxn 矩阵，因此其实相当于注意力是在序列空间中做注意力搜索，维度更多，信息更充足
而 线性注意力 虽然逻辑一样，但是先计算得到 S 是 dxd 矩阵，是在维度上做计算，相当于先计算维度相关的信息，而不是序列相关的信息，最后再跟序列 query 乘积得到 结果，信息通过加法级联起来，信息更恒定但是序列之间的关系有可能会损失

所以需要全局注意力一层来保证整个网络信息能稳定计算，再用 线性注意力 加速计算过程。



**A3**：


如果不通过前缀累加，线性注意力计算会让每个token都能看到未来的信息
而前缀累加避免了这个问题，通过逐个累加的方式，让 i 位置的 key 只能与 i 位置的 value 计算

---

<!-- 教师批改区（提交作业后由导师填写，请勿手动修改） -->

### 📝 批改结果

**Q1 批改**：算术强度的计算思路正确，数值结果 I≈1.0 也对。有两个小问题：① 公式中 `2BD + 2DF + 2BD` 第三项应为 `2BF`（写 Y 矩阵的字节数），不过由于 B=1 时 BD 和 BF 的数值恰好分别等于 D 和 F，代入后数字是对的；② H100 带宽应为 **3.35 TB/s** 而非 3.5 TB/s，临界值应为 ≈295 而非 282.57。结论完全正确。 — 得分：**8/10**

<details>
<summary>📖 Q1 参考答案</summary>

矩阵乘法 $X_{B \times D} \times W_{D \times F}$ 的算术强度推导：

**数据搬运量**（FP16，每元素 2 字节）：
- 读 X：$2BD$ 字节
- 读 W：$2DF$ 字节
- 写 Y：$2BF$ 字节（注意是 **BF** 不是 BD）
- 总传输：$2(BD + DF + BF)$

**计算量**：$\text{FLOPs} = 2BDF$

**算术强度**：
$$I = \frac{2BDF}{2(BD + DF + BF)} = \frac{BDF}{BD + DF + BF}$$

**代入 B=1, D=4096, F=11008**：
$$I = \frac{4096 \times 11008}{4096 + 4096 \times 11008 + 11008} = \frac{45,088,768}{45,103,872} \approx 1.0 \text{ FLOPs/byte}$$

**H100 临界值**：
$$I_{\text{crit}} = \frac{989 \text{ TFLOPs/s}}{3.35 \text{ TB/s}} \approx 295 \text{ FLOPs/byte}$$

$I \approx 1.0 \ll 295$ → Decode 阶段是严重的 **memory-bound**，GPU 算力大量空闲等待数据搬运。

**常见错误**：将写入 Y 的字节数写成 $2BD$ 而非 $2BF$。当 B=1 时数值碰巧一样（$BD = D$, $BF = F$），但符号公式不同。B>1 时这个错误会导致计算结果偏差。

</details>

---

**Q2 批改**：核心洞察很好——线性注意力在"维度空间"聚合信息（$d \times d$），标准注意力在"序列空间"做 token 对 token 的精细建模（$N \times N$），这是两者表达能力差异的本质。但缺少一个关键论据：MiniMax 论文实验发现纯线性注意力**无法完成检索任务（retrieval）**，因为 $d \times d$ 的 S 矩阵是所有 token 的"模糊混合"，无法定位特定 token 的精确位置。7:1 的比例不是随意选的，而是在保证检索能力的前提下最大化线性注意力的比例。 — 得分：**7/10**

<details>
<summary>📖 Q2 参考答案</summary>

从**计算效率**角度：
- 线性注意力复杂度 $O(Nd^2)$，在长序列（$N \gg d$）时远优于 Softmax 的 $O(N^2d)$
- 7 层线性注意力处理绝大部分长序列依赖，只需极低计算成本
- 1 层 Softmax 覆盖全局精确建模，成本可接受（只有 1/8 的层）

从**信息建模能力**角度：
- **线性注意力的 S 矩阵（$d \times d$）是所有 token 的 K-V 对的加权累加**，本质是"模糊混合"——无法区分不同 token 的精确位置，也无法做精确的 token-to-token 匹配
- **Softmax 注意力的注意力矩阵（$N \times N$）保留了每对 token 的独立权重**，可以做精确检索
- MiniMax 论文实验证实：**纯线性注意力模型在检索任务（needle-in-a-haystack）上显著弱于混合模型**。这是因为检索需要"找到序列中特定位置的特定 token"，而线性注意力的累加器天然不区分位置
- 7:1 比例是实验验证的平衡点：更多 Softmax 层 → 检索更好但长序列推理更慢；更少 Softmax 层 → 速度更快但检索能力下降

**类比**：线性注意力像一个"压缩摘要"——知道大意但找不到原文具体在哪一行；Softmax 注意力像"全文检索"——可以精确定位到每一行。7 层压缩 + 1 层全文检索 = 兼顾效率和精度。

</details>

---

**Q3 批改**：第一点正确——不使用前缀累加会导致"偷看未来"。但第二点的表述不够精确："让 i 位置的 key 只能与 i 位置的 value 计算"——实际上前缀累加是让位置 i 的输出**只依赖位置 1 到 i 的所有** K-V 对（$S_i = k_1v_1^\top + k_2v_2^\top + \dots + k_iv_i^\top$），而不是"只与自己的 K-V 计算"。此外题目要求"结合数值推演示例说明"，但回答中没有引用具体的偏差数据。 — 得分：**5/10**

<details>
<summary>📖 Q3 参考答案</summary>

**不使用前缀累加时的具体影响**：

直接计算全局的 $K^\top V$，则第 $i$ 个 token 的输出为：
$$y_i = \frac{q_i^\top \cdot \sum_{j=1}^{L} k_j v_j^\top}{q_i^\top \cdot \sum_{j=1}^{L} k_j}$$

这里 $L$ 是序列总长度，包含了位置 $i+1, \dots, L$ 的所有未来 token 的信息。而正确的前缀累加只累加到位置 $i$：
$$y_i = \frac{q_i^\top \cdot \sum_{j=1}^{i} k_j v_j^\top}{q_i^\top \cdot \sum_{j=1}^{i} k_j}$$

**数值验证**（来自代码输出的偏差数据）：

```
偏差 = Y_wrong[i] - Y_correct[i]

token "I"    (位置0) → 偏差 [0.05, -0.196]
  → 全局计算包含了位置1~4共4个未来token的K-V
  → 偏差最大，因为这些未来信息完全不应该被看到

token "like" (位置1) → 偏差 [-0.097, -0.037]
  → 包含位置2~4共3个未来token，偏差减小

token "😄"  (位置4) → 偏差 [0.0, 0.0]
  → 最后一个token，没有未来信息可泄露，两种方式完全一致
```

**偏差递减的规律**：位置越靠前，偷看到的未来 token 越多，偏差越大。最后一个 token 没有未来信息，偏差为零。

**关键澄清**：前缀累加不是"让 i 位置的 key 只与 i 位置的 value 计算"。$S_i$ 包含的是位置 1 到 i 的**所有** K-V 对的累加：$S_i = k_1v_1^\top + k_2v_2^\top + \dots + k_iv_i^\top$。每个 token 的输出依赖它前面所有 token 的信息，但绝不包含它后面的 token。

</details>

---

**综合评价**：Q1 计算能力扎实，公式符号有小瑕疵但数值推导正确；Q2 对线性/标准注意力的信息建模差异理解深入，补充检索任务的论据会更完整；Q3 核心概念正确但表述精度和数值引证需要加强。建议复习前缀累加的精确数学表达。可以继续下一模块。

**批改时间**：2026-05-22
