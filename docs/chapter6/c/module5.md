# 第 6 章：GPU 和 GPU 相关的优化 — 模块 5：FlashAttention 与 PagedAttention

> 📍 学习进度：第 6 章，第 5 / 5 模块
> 📅 生成时间：2026-05-05

---

## 学习目标

- 理解标准 Attention 的 IO 瓶颈（N×N 中间矩阵的 HBM 读写问题）
- 掌握 FlashAttention V1 的核心思想：分块 + Online Softmax
- 理解 V2 和 V3 的演进（并行优化 → 异步流水线）
- 掌握 PagedAttention 的动机和原理（分页管理 KV Cache）
- 理解 FlashAttention 与 PagedAttention 的定位差异

---

## 核心内容

### 一、标准 Attention 的 IO 瓶颈

#### 1.1 回顾注意力计算

```
标准注意力计算：

  S = Q @ K^T        → (N, N) 矩阵
  P = softmax(S)     → (N, N) 矩阵
  O = P @ V          → (N, d) 矩阵

其中 N = 序列长度，d = 头维度
```

#### 1.2 N×N 矩阵的 HBM 灾难

```
N = 4096 时：

  S 矩阵：4096 × 4096 = 16,777,216 个元素
  FP32 下：16M × 4 字节 = 64 MB
  FP16/BF16 下：16M × 2 字节 = 32 MB

标准实现需要：
  ① 计算 S = Q@K^T → 写回 HBM（32~64 MB）
  ② 计算 P = softmax(S) → 从 HBM 读 S → 写回 P（32~64 MB 读 + 写）
  ③ 计算 O = P@V → 从 HBM 读 P → 写回 O

  仅中间矩阵 S 的读写就占用了 64~128 MB 的 HBM 带宽！
  而 A100 的 HBM 带宽是 2 TB/s，看似很快，
  但 128 MB 的搬运在序列长度增长时迅速成为瓶颈

N 的影响：复杂度 O(N²)
  N=4096  → 中间矩阵 64 MB
  N=16384 → 中间矩阵 1 GB
  N=65536 → 中间矩阵 16 GB  ← 超过单卡显存！
```

> 💡 **核心问题**：瓶颈不在计算量（FLOPs），而在**中间矩阵 N×N 的 HBM 读写**。这不是"算得慢"，而是"搬数据搬得慢"——典型的内存受限问题。

---

### 二、FlashAttention V1：Tiling + Online Softmax

#### 2.1 核心思想

FlashAttention 的核心思路可以用一句话概括：

```
把 Q、K、V 分成小块（Tile），在 SRAM（片上高速内存）中完成
Q@K^T、softmax、P@V 的全部计算，只把最终结果 O 写回 HBM。

N×N 的中间矩阵永远不写回 HBM！
```

#### 2.2 为什么需要 Online Softmax

直接 Tiling 遇到一个技术难题：**softmax 需要整行的全局信息**。

```
标准 softmax：
  P[i,j] = exp(S[i,j]) / Σ_k exp(S[i,k])
                        ↑
          这个分母需要 S 的一整行！
          如果 S 被分块了，不同块的 exp 值无法直接累加
```

FlashAttention 用 **Online Softmax** 解决这个问题——在流式处理过程中动态维护当前最大值和归一化因子：

```
维护两个状态变量：
  m_i = 当前行已处理部分的最大值
  l_i = 当前行已处理部分的 exp 累加和

处理新的块 j 时：
  m_new = max(m_i, rowmax(S_ij))
  l_i = exp(m_i - m_new) × l_i + Σ exp(S_ij - m_new)
  O_i = exp(m_i - m_new) × O_i + exp(S_ij - m_new) @ V_j
  m_i = m_new

→ 每处理一个块就更新一次，不需要等所有块处理完
→ 数学上与标准 softmax 完全等价！
```

#### 2.3 V1 算法伪代码

```
# 初始化
m_i = -inf
l_i = 0
O_i = 0

for each Q block i:                    # 外循环：遍历 Q 的块
    load Q_i into SRAM
    m_i = -inf
    l_i = 0
    O_i = 0

    for each K,V block j:             # 内循环：遍历 K,V 的块
        if causal and j > i:
            continue

        load K_j, V_j into SRAM

        S_ij = Q_i @ K_j^T            # 在 SRAM 中计算（N/T × N/T）

        if causal and i == j:
            apply mask to S_ij

        # Online Softmax 更新
        m_new = max(m_i, rowmax(S_ij))
        l_i = exp(m_i - m_new) * l_i + sum(exp(S_ij - m_new), axis=1)
        O_i = exp(m_i - m_new) * O_i + exp(S_ij - m_new) @ V_j
        m_i = m_new

    O_i = O_i / l_i                   # 最终归一化
    write O_i to HBM                  # 只写最终结果！
```

#### 2.4 V1 的效果

```
IO 复杂度对比：

  标准 Attention：O(N²) HBM 读写（N×N 矩阵）
  FlashAttention V1：O(N) HBM 读写（只读 Q/K/V，只写 O）

显存占用：
  标准：O(N²)（需要存储 N×N 中间矩阵）
  V1：O(N)（只需存储最终输出 O）

实际性能（A100）：
  标准实现：~60 TFLOPS
  V1：~225 TFLOPS（3.75× 加速）
  模型 FLOPs 利用率（MFU）：~72%
```

---

### 三、FlashAttention V2：并行调度优化

#### 3.1 V1 的问题

V1 虽然解决了 IO 瓶颈，但有一个性能缺陷：

```
V1 的执行模式：
  外循环遍历 Q 块，内循环遍历 K,V 块
  每处理一个 K,V 块 → 必须做一次 online softmax 更新（标量运算）
  → 标量运算和矩阵乘法交替执行
  → Tensor Core 不能持续满载（被标量运算打断）
```

#### 3.2 V2 的核心改进

```
V1：一个 Q_i 对应一个线程块，串行扫描所有 K,V
V2：同一个 Q_i 在 K,V 维度进行切分，多个线程块并行处理（split-KV）
```

```
V2 算法：

for each Q block i in parallel:          # Q 维度并行
    load Q_i into SRAM

    for each K,V block j:
        load K_j, V_j into SRAM

        S_ij = Q_i @ K_j^T              # Tensor Core
        update m_i, l_i, O_i            # online softmax 更新

    O_i = O_i / l_i
    write O_i to HBM
```

V2 的关键变化：

```
① 去掉 Q 维度的外循环串行依赖 → 更多 Q 块可并行处理
② 减少非 matmul 操作对 Tensor Core 的打断
③ 增加并行线程块数 → 提高 Tensor Core 整体占用率
```

#### 3.3 V2 的效果

```
性能提升（A100）：
  相比 V1：加速 1.7~2.0 倍
  相比 PyTorch 标准实现：加速 8~10 倍
  MFU：进一步提升

关键：V2 不是减少 FLOPs，而是把"online softmax 的时间串行依赖"改写成"空间并行"
```

> 💡 **一句话对比**：V1 解决 IO 瓶颈（减少 HBM 读写），V2 解决并行调度（提高 Tensor Core 利用率）。

---

### 四、FlashAttention V3：异步流水线（H100）

#### 4.1 H100 的新特性

H100（Hopper 架构）引入了两个革命性特性：

```
① 异步执行模型
  TMA（Tensor Memory Accelerator）可以异步搬运数据
  数据加载与 Tensor Core 计算完全重叠

② WGMMA 指令
  Warpgroup Matrix Multiply-Accumulate
  4 个 Warp（128 线程）组成一个 Warpgroup
  硬件层面支持异步执行
```

#### 4.2 V3 的核心改进

```
生产者-消费者流水线 + 双缓冲（Double Buffering）：

buffer_K[2], buffer_V[2]               # 双缓冲

for each Q block i:
    load Q_i
    async_load(buffer_K[0], buffer_V[0])

    for j in range(num_blocks):
        curr = j % 2
        next = (j + 1) % 2

        # 生产者：提前加载下一块（异步）
        if j + 1 < num_blocks:
            async_load(buffer_K[next], buffer_V[next])

        # 消费者：用当前 buffer 计算
        S_ij = wgmma(Q_i, buffer_K[curr])
        update m_i, l_i
        O_i += wgmma(P_ij, buffer_V[curr])

        # 只在必要时同步
        wait_for(buffer_K[next])

    write O_i
```

```
V2 的问题：计算和数据搬运是串行的
  [计算] → [等数据] → [计算] → [等数据] → ...
  SM 利用率：~60%

V3 的解决：计算和搬运重叠（异步流水线）
  [计算 buffer_0] [搬运 buffer_1]
                   [计算 buffer_1] [搬运 buffer_0]
                                    [计算 buffer_0] [搬运 buffer_1]
  SM 利用率：80%+
```

#### 4.3 V3 的 FP8 支持

```
H100 FP8 算力：~1979 TFLOPS（是 FP16 的 2 倍）

V3 的混合精度策略：
  Q@K^T 矩阵乘法 → FP8（充分利用 Tensor Core）
  累加器 → FP16/BF16（保证精度）
  Softmax → FP32（数值稳定性，指数运算易溢出）

动态缩放因子：按 tile 动态计算，防止 FP8 溢出
```

#### 4.4 V3 的效果

```
性能（H100 SXM）：
  FP8 TFLOPS 利用率：75~80%（接近理论峰值）
  相比 V2 on H100（FP16）：加速 1.5~2.0 倍（FP8 版本）
  相比 V2 on H100（FP16）：加速 1.3 倍（同精度，异步优化）

长序列能力：
  单卡 H100 80GB 可稳定训练 256K 长度
  支撑 100K~1M 上下文窗口的工程实现
```

#### 4.5 版本演进总结

```
V1 → V2 → V3 的核心变化：

  V1：解决 IO 瓶颈 → 减少 HBM 读写（N² → N）
  V2：解决并行调度 → 提高 Tensor Core 利用率
  V3：解决计算/访存重叠 → 异步流水线 + FP8

一句话：
  从"减少内存带宽瓶颈" → "提升并行调度与 GPU 利用率"
  → "实现计算与访存重叠，让 SM 始终忙碌"
```

---

### 五、PagedAttention：管理 KV Cache 的显存

#### 5.1 推理中的 KV Cache 问题

自回归推理（每步生成 1 个 token）时，需要缓存之前所有 token 的 K 和 V：

```
生成第 t 个 token 时：
  Q_t（1 个 token）与 所有 K_1, K_2, ..., K_t 做注意力
  → 需要缓存 K_1~K_{t-1} 和 V_1~V_{t-1}

传统管理方式：为每个请求预分配 max_seq_len 的连续空间

  请求 A（实际生成 128 token）：预分配 2048 → 浪费 1920
  请求 B（实际生成 1024 token）：预分配 2048 → 浪费 1024
  请求 C（实际生成 300 token）：预分配 2048 → 浪费 1748

问题：
  内部碎片：预留空间 > 实际使用 → 空间浪费
  外部碎片：释放后空闲空间分散 → 无法分配给新请求
  → 总空闲空间充足，但找不到连续的大块
```

#### 5.2 PagedAttention 的核心思想

借鉴操作系统**分页机制**：

```
将 KV Cache 切分为固定大小的 Page（如 16 token/Page）

  逻辑上：token 1~16 | 17~32 | 33~48 | ...
  物理上：Block 7 | Block 2 | Block 19 | ...（不要求连续！）

通过页表（Block Table）映射：
  请求 ID + Token 偏移 → 逻辑块索引 → Block Table → 物理块地址
```

#### 5.3 数值对比

```
请求 C：实际 300 token

传统方式（预分配 max_seq_len=2048）：
  分配 2048 个 token 的连续空间
  浪费：2048 - 300 = 1748 个 token

PagedAttention（Page 大小 = 16 token）：
  需要 ⌈300/16⌉ = 19 个 Page
  实际分配：19 × 16 = 304 个 token 的空间
  浪费：304 - 300 = 4 个 token

  浪费从 1748 降到 4 → 减少 99.8%！
```

#### 5.4 用餐类比

```
传统方式 = 按次收费的满汉全席
  不管吃多少，都按 2048 道菜的规模占位
  吃 300 道就走，剩下 1748 道只能倒掉（显存被强占且无法复用）

PagedAttention = 按需取餐的自助餐
  每 16 个 token 是一盘菜
  吃完一盘再取下一盘，空盘可以立即给其他人用（显存动态回收复用）
```

---

### 六、FlashAttention vs PagedAttention

两者是**不同维度的优化**，通常协同使用：

| 维度 | FlashAttention | PagedAttention |
|------|---------------|----------------|
| 优化目标 | 单次前向计算的 IO 复杂度 | 整个生成生命周期的显存管理 |
| 优化层级 | 算子级（微观） | 系统级（宏观） |
| 解决的问题 | "算得更快" | "存得更高效" |
| 影响范围 | 训练 + 推理 | 仅推理 |
| 核心机制 | Tiling + Online Softmax | 分页 + 页表映射 |
| 复杂度改进 | O(N²) → O(N) HBM 读写 | O(max_seq_len) → O(page_size) 浪费 |

```
实际推理框架（如 vLLM）中：

  FlashAttention → 负责每个请求内部的 Attention 计算（更快）
  PagedAttention → 负责多个请求之间的显存调度（更高效）

  两者协同：同时优化单请求延迟与整体吞吐量
```

---

## 🧠 本模块问题

请在下方回答以下问题后，输入 `提交作业` 提交。

**Q1**：标准 Attention 计算的 IO 瓶颈在哪里？请写出 N=4096 时 N×N 中间矩阵的大小，并解释为什么这会导致内存受限。

**Q2**：FlashAttention V1 如何解决上述 IO 瓶颈？请写出其核心思想（两句话以内），并解释 Online Softmax 的作用。

**Q3**：PagedAttention 解决了什么问题？请用"内部碎片"和"外部碎片"的概念，对比传统 KV Cache 管理方式和 PagedAttention 的显存利用率差异。

---

<!-- 学习者作答区（请在此处填写你的答案） -->

**A1**：

标准的 Attention 计算的 IO 瓶颈在于 attention 计算时多次 HBM 读取操作
这里 N = 4096，则 NxN 矩阵 FP32 则占用空间 4096 * 4096 * 4 Bytes = 64MB
即一个中间矩阵的 HBM 搬运就占了 64 MB，当 N 增大时，带宽占用将指数增长，从而导致内存受限。


**A2**：

FLashAttentionV1 通过 Tiling 算法和 online softmax，将逐个数据搬运改成了逐块数据搬运，并且通过 online softmax 方法解决了 softmax 在分块下计算的难题。

online softmax 通过维护三个变量：m_i(当前最大值), l_i（当前 row 的元素 - 最大值的 exp 和） 以及 o_i（softmax 的结果），根据新载入的块动态更新变量以及输出结果。


**A3**：

PagedAttention 解决了显存分配需要根据 max_seq_len 来预先分配，有很多 预留空间浪费，造成内部碎片。在显存使用释放后，因为显存分配碎片化，无法有效利用，导致外部碎片。

这是 KV cache 下造成的问题，但是 PagedAttention 将显存按照 16 token/page 按页划分，每次分配在表面分配了 max_seq_len ，实际上会按照需求的 token 数量分配一定 数量 的page 来分配 token，而在使用后也会按照 token 收回，并且分配的 page 不要求要连续，从而解决了外部碎片的问题。

---

<!-- 教师批改区（提交作业后由导师填写，请勿手动修改） -->

### 📝 批改结果

**Q1 批改**：基本概念正确——N×N 矩阵 FP32 下 64MB 的计算对了，也提到了 N 增大时带宽占用增长。但有两个问题：① 用词"指数增长"不准确，N×N 是**多项式**（O(N²)）增长，不是指数增长（指数增长是 2^N）；② 没有说明标准 Attention 需要**多次 HBM 读写**（写 S、读 S 写 P、读 P 写 O），只说了一次中间矩阵的搬运。应强调"仅 S 矩阵一次写回就是 64MB，再加上 softmax 和 P@V 的读写，总共 64~128MB 的 HBM 带宽占用"。 — 得分：**7/10**

<details>
<summary>📖 Q1 参考答案</summary>

标准 Attention 计算的 IO 瓶颈在于：N×N 中间矩阵 S = Q@K^T 需要写回 HBM，后续 softmax(S) 和 P@V 还需要反复从 HBM 读写这个大矩阵。

N=4096 时：
- S 矩阵：4096×4096 = 16,777,216 个元素
- FP32：16M × 4 字节 = 64 MB
- FP16/BF16：16M × 2 字节 = 32 MB

标准实现需要 3 次 HBM 读写往返：
① 计算 S = Q@K^T → 写回 HBM（32~64 MB）
② 计算 softmax(S) → 从 HBM 读 S → 写回 P（32~64 MB 读 + 写）
③ 计算 O = P@V → 从 HBM 读 P → 写回 O

仅 S 矩阵的读写就占用 64~128 MB 的 HBM 带宽，而 A100 HBM 带宽是 2 TB/s。当 N 从 4096 增到 16384 时，中间矩阵从 64 MB 增到 1 GB（16 倍增长，因为 O(N²)），远超 HBM 带宽的承载能力。

这是典型的"内存受限"问题——瓶颈不在计算量（FLOPs），而在中间矩阵的 HBM 读写。计算单元大部分时间在等数据从 HBM 搬运过来。

</details>

---

**Q2 批改**：核心思路正确——Tiling 分块 + Online Softmax 解决分块下 softmax 的难题，m_i、l_i、O_i 三个变量的维护机制理解到位。小问题：① 写的是"o_i（softmax 的结果）"，实际上是 O_i（输出矩阵的第 i 行），不是 softmax 的结果，softmax 的结果是中间量 P；② "逐个数据搬运改成了逐块数据搬运"表述不够精确——应该是"避免将 N×N 中间矩阵写回 HBM，只在 SRAM 中完成全部计算后写回最终结果 O"。Online Softmax 的解释较好，但可以补充一句"数学上与标准 softmax 完全等价"。 — 得分：**8/10**

<details>
<summary>📖 Q2 参考答案</summary>

**核心思想（两句话）**：
1. 把 Q、K、V 分成小块（Tile），在 SRAM 中完成 Q@K^T、softmax、P@V 的全部计算，N×N 中间矩阵永远不写回 HBM，只把最终结果 O 写回。
2. 通过 Online Softmax 算法，在分块处理过程中动态维护全局 softmax 归一化因子，数学上与标准 softmax 完全等价。

**Online Softmax 的作用**：
标准 softmax 需要整行的全局信息（P[i,j] = exp(S[i,j]) / Σ_k exp(S[i,k])），如果 S 被分块了，不同块的 exp 值无法直接累加。

Online Softmax 通过维护两个状态变量解决：
- m_i：当前行已处理部分的最大值
- l_i：当前行已处理部分的 exp 累加和

处理新块 j 时，通过 rescaling 公式将旧块的贡献按新块的最大值重新缩放：
- m_new = max(m_i, rowmax(S_ij))   → 更新全局最大值
- l_i = exp(m_i - m_new) × l_i + Σ exp(S_ij - m_new)  → 重新缩放旧累加和 + 加入新块
- O_i = exp(m_i - m_new) × O_i + exp(S_ij - m_new) @ V_j  → 重新缩放旧输出 + 加入新块贡献

关键性质：每处理一个块就更新一次，不需要等所有块处理完，且最终结果与标准 softmax 数学上完全等价。这使得分块计算成为可能。

</details>

---

**Q3 批改**：核心理解正确——内部碎片（预留 > 实际使用）和外部碎片（释放后空间分散）的概念都对了，16 token/page 按需分配、page 不要求连续的机制也理解到位。回答质量比前面几个模块有进步。小问题：① 没有用具体数字对比（如 1748→4 的浪费对比），这让概念不够落地；② "分配的 page 不要求要连续"应强调这正是解决**外部碎片**的关键——因为页表映射使得逻辑连续与物理连续解耦。 — 得分：**8/10**

<details>
<summary>📖 Q3 参考答案</summary>

**PagedAttention 解决的问题**：传统 KV Cache 管理需要为每个请求预分配 max_seq_len 的连续显存空间，导致严重的内部碎片和外部碎片。

**内部碎片**：预留空间 > 实际使用 → 空间浪费。
例如请求实际生成 300 token，但预分配了 max_seq_len=2048 的连续空间，浪费 1748 个 token。

**外部碎片**：请求释放后，空闲空间分散在显存中，虽然总空间充足，但找不到连续的大块来分配给新请求。

**PagedAttention 的解决方案**：
借鉴操作系统分页机制，将 KV Cache 切分为固定大小的 Page（如 16 token/Page），通过页表（Block Table）实现逻辑连续与物理连续的解耦：
- 逻辑上：token 1~16 | 17~32 | 33~48 | ...
- 物理上：Block 7 | Block 2 | Block 19 | ...（不要求连续！）

**显存利用率对比**（300 token 请求）：
- 传统方式：分配 2048 token → 浪费 1748 token
- PagedAttention：⌈300/16⌉ = 19 个 Page = 304 token → 浪费仅 4 token
- 浪费从 1748 降到 4，减少 99.8%

关键：Page 不要求连续，通过页表映射寻址 → 解决外部碎片；按需分配 Page → 解决内部碎片。空闲 Page 可立即复用给其他请求。

</details>

---

**综合评价**：平均 7.7/10。核心概念（IO 瓶颈、FlashAttention 分块思想、Online Softmax 机制、PagedAttention 分页管理）都理解正确。相比模块 3 和 4 的回答（平均 6.7/10），本次回答的完整度有所提升。注意两点：① 数学用语要精确（"多项式增长"而非"指数增长"）；② 尽量用具体数字支撑概念（如 1748→4 的对比），让抽象概念落地。

**批改时间**：2026-05-05
