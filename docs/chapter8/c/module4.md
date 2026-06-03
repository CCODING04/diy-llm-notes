# 第 8 章：分布式训练 — 模块 4：张量并行、序列/上下文并行与 3D 并行

> 📍 学习进度：第 8 章，第 4 / 5 模块
> 📅 生成时间：2026-05-13

---

## 学习目标

- 能解释张量并行（Tensor Parallelism, TP）为什么适合切分 Transformer 的大矩阵乘法
- 能区分 MLP TP、Attention TP、序列并行（SP）、上下文并行（CP）各自切分的维度和通信模式
- 能用具体数值估算 TP/PP/DP/CP 的组合关系：`总 GPU 数 = TP × PP × DP × CP`
- 能根据模型规模、序列长度和网络拓扑，给出合理的 3D 并行配置

---

## 核心内容

### 一、张量并行：沿隐藏维度切分单层计算

张量并行的核心不是“把层分给不同 GPU”，而是**每个 GPU 都参与同一层，但只持有这一层权重矩阵的一部分**。

```
数据并行（DP）：每张 GPU 有完整模型，切 batch
流水线并行（PP）：每张 GPU 有部分层，切 depth
张量并行（TP）：每张 GPU 有每层的一部分参数，切 hidden width

例：hidden_size = 8192，TP = 4
    每张 GPU 只负责约 8192 / 4 = 2048 个 hidden 通道相关的矩阵块
```

TP 的优势是可以把单层参数和单层矩阵乘法拆到多张 GPU 上；代价是**每一层附近都会发生 collective communication**，所以 TP 通常放在同节点高速互联内，例如 NVLink/NVSwitch。

![矩阵乘法分割示例](<../images/8-24-矩阵乘法分割示例.png>)

> 💡 **补充（官方文档 / Megatron Core）**：Megatron Core 的并行策略指南把 TP 描述为切分单层计算，建议在大 hidden dimension 或单层无法放入单 GPU 时使用，并且通常与 DP、PP 一起组合。该指南还建议使用 TP 时开启 Sequence Parallelism，以减少部分激活内存。
> 来源：https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html

---

### 二、MLP 的 TP：先列切，再行切

Transformer FFN/MLP 通常可以写成：

$$
H = \mathrm{GELU}(X W_1), \quad Y = H W_2
$$

设：

- `X`: `[B, S, d_model]`
- `W1`: `[d_model, 4d_model]`
- `W2`: `[4d_model, d_model]`
- TP = 2

#### 2.1 为什么 `W1` 按列切

`W1` 的输出维度是 `4d_model`。按列切 `W1`，每张 GPU 得到一部分 FFN 中间通道：

```
W1 = [ W1_0 | W1_1 ]

GPU0: H0 = GELU(X @ W1_0)   # [B, S, 2d_model]
GPU1: H1 = GELU(X @ W1_1)   # [B, S, 2d_model]

完整 H = [H0 | H1]
```

这样做的好处是：`GELU` 是逐元素函数，`H0` 和 `H1` 可以各自在本地完成，不需要先把完整 `H` 聚合回来。

#### 2.2 为什么 `W2` 按行切

`W2` 的输入维度是 `4d_model`。因为 `H` 已经被切成 `[H0 | H1]`，所以让 `W2` 按行切，刚好让每张 GPU 消化自己那部分 `H`：

```
W2 = [ W2_0 ]    # W2_0: [2d_model, d_model]
     [ W2_1 ]    # W2_1: [2d_model, d_model]

GPU0: Y0 = H0 @ W2_0    # [B, S, d_model]
GPU1: Y1 = H1 @ W2_1    # [B, S, d_model]

完整 Y = Y0 + Y1
```

这里最终需要的是**求和**，不是拼接：

$$
[H_0 \mid H_1]
\begin{bmatrix}
W_{2,0} \\
W_{2,1}
\end{bmatrix}
= H_0 W_{2,0} + H_1 W_{2,1}
$$

因此最后的通信是 `All-Reduce(sum)`，让每张 GPU 都拿到相同的完整 `Y`，继续进入下一层。

#### 2.3 带数字的维度推演

假设 `B=2`，`S=4`，`d_model=8`，TP=2：

```
X:       [2, 4, 8]
W1:      [8, 32]       -> 每卡 W1_i: [8, 16]
H_i:     [2, 4, 16]    -> 每卡只保存一半 FFN 中间激活
W2_i:    [16, 8]
Y_i:     [2, 4, 8]

All-Reduce:
Y = Y0 + Y1 = [2, 4, 8]
```

关键点：`All-Gather` 用于“拼接不同分片”，`All-Reduce` 用于“把同形状 partial result 求和”。MLP 的第二个矩阵乘法属于后者。

![MLP 张量并行示意](<../images/8-25-MLP示例.png>)

---

### 三、Attention 的 TP：head 可分，但不是完全无通信

多头注意力天然适合按 head 分片。设 8 个 attention heads，TP=2：

```
GPU0: Head 0-3
GPU1: Head 4-7
```

每张 GPU 可以独立计算自己负责的 `QK^T`、softmax 和 `softmax @ V`。这部分确实不需要 GPU 间通信，因为不同 head 之间互不依赖。

但完整 Attention block 不止 head 内计算，还包括输出投影 `W_o`：

```
局部 head 输出:
GPU0: O0 = Attention(Q0, K0, V0)
GPU1: O1 = Attention(Q1, K1, V1)

拼接后再输出投影:
O = [O0 | O1] @ W_o
```

Megatron 风格实现通常把 Attention 的 QKV 投影和输出投影也做列/行切分，因此最后仍会在合适位置发生 `All-Reduce` 或与后续层的通信融合。更准确的说法是：

```
Attention 的 head 内计算局部独立；
Attention block 作为完整层，仍需要通过 collective 通信恢复后续层所需的完整语义。
```

![张量并行的条件](<../images/8-26-张量并行的条件.png>)

---

### 四、课程代码：教学版 TP 前向演示

原课程在第 8 章后半部分给了一个简化 MLP 的 TP 前向示例。它不是完整 Transformer TP，也没有实现反向传播，只用于说明“沿 hidden 维度切参数，层间 all_gather 激活”的基本数据流。

来源：[chapter8_第八章分布式训练.md](../chapter8_第八章分布式训练.md) 中 `tensor_parallelism_main` 代码片段。

```python
def tensor_parallelism_main(rank: int, world_size: int, data: torch.Tensor, num_layers: int):
    setup(rank, world_size)
    data = data.to(get_device(rank))
    batch_size = data.size(0)
    num_dim = data.size(1)
    local_num_dim = int_divide(num_dim, world_size)

    params = [get_init_params(num_dim, local_num_dim, rank) for i in range(num_layers)]

    x = data
    for i in range(num_layers):
        x = x @ params[i]
        x = F.gelu(x)

        activations = [
            torch.empty(batch_size, local_num_dim, device=get_device(rank))
            for _ in range(world_size)
        ]
        dist.all_gather(tensor_list=activations, tensor=x, async_op=False)
        x = torch.cat(activations, dim=1)

    print(f"[tensor_parallelism] Rank {rank}: forward pass produced activations {summarize_tensor(x)}")
    cleanup()
```

适用范围：

- 这是 MLP 前向传播的教学版本，不是生产级 Megatron TP。
- 它使用 `All-Gather` 恢复完整 hidden activation，便于下一层继续计算。
- 它没有展示反向传播、通信计算重叠、bias/dropout/residual、LayerNorm，也没有展示 Attention 的 TP。

---

### 五、Sequence Parallelism 与 Context Parallelism：名字相近，但切分范围不同

这两个概念容易混淆。简单区分：

| 概念 | 切分维度 | 切分范围 | 主要目的 | Attention 是否需要特殊通信 |
|------|----------|----------|----------|-----------------------------|
| Sequence Parallelism (SP) | sequence | 通常只切 LayerNorm、Dropout 等部分激活 | 配合 TP 降低激活内存 | 通常不负责完整长上下文 attention |
| Context Parallelism (CP) | sequence | 切网络输入和几乎所有 activation | 训练长上下文，降低每卡 activation | 需要为 attention 交换/收集 K/V |
| Ring Attention | sequence block | 用环形传递 K/V block 计算注意力 | 极长上下文，通信与计算重叠 | 是，需要 K/V block 环形传递 |

#### 5.1 Sequence Parallelism：配合 TP 省激活

在 TP 中，某些操作需要完整 hidden 或完整 sequence 激活。如果每张 GPU 都保存完整激活，会浪费显存。SP 的思路是把可分片的 activation 沿 sequence 维度分散保存：

```
原始 activation: [B, S, H]
TP = 4 时，每张 GPU 不一定都保存完整 [B, S, H]

SP 后：
GPU0: [B, S/4, H]
GPU1: [B, S/4, H]
GPU2: [B, S/4, H]
GPU3: [B, S/4, H]
```

这特别适合 LayerNorm、Dropout 这类对 token 独立的操作，因为它们不需要跨 token 读取信息。

> 💡 **补充（官方文档 / Megatron Core）**：Megatron Core 文档明确建议使用 TP 时开启 `--sequence-parallel`，并说明 SP 会通过切分 LayerNorm 和 Dropout 等激活来降低 activation memory。
> 来源：https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html

#### 5.2 Context Parallelism：把长序列切到多组 GPU

CP 也是沿 sequence length 切，但它比 SP 更激进：CP 会切分网络输入和所有 activation，使每张 GPU 只处理一段上下文。

假设 `S = 8192`，CP=4：

```
GPU0: token 0-2047
GPU1: token 2048-4095
GPU2: token 4096-6143
GPU3: token 6144-8191
```

对 MLP、LayerNorm 这类 token-wise 操作，每张 GPU 只算自己的 token 片段即可。但 Attention 有跨 token 依赖：

```
某个 token 的 Q 需要看同一序列中所有 token 的 K/V
```

所以 CP 不能说成“每张 GPU 完全独立计算自己的 attention”。更准确地说：

```
CP 让每张 GPU 持有部分 Q 和部分 K/V；
attention 计算时，需要通过 all-gather、p2p、a2a 或 ring 等方式交换 K/V 信息。
```

> 💡 **补充（官方文档 / Megatron Core CP）**：Megatron Core 的 CP 文档说明，CP 沿 sequence length 切分输入和 activation；除 Attention 外的模块通常不需要特殊改动，但 Attention 中每个 token 的 Q 需要和同序列全部 K/V 交互，因此 CP 需要额外通信来获得其他 sequence chunk 的 K/V。
> 来源：https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/context_parallel.html

#### 5.3 Ring Attention：一种长上下文 attention 通信方式

Ring Attention 的核心是把 `Q/K/V` 按 sequence block 切分，并让 K/V block 沿环传递：

```
4 GPU, 4 个 sequence block:

初始：
GPU0: Q0, K0, V0
GPU1: Q1, K1, V1
GPU2: Q2, K2, V2
GPU3: Q3, K3, V3

第 1 轮：每张 GPU 用本地 Q 和本地 K/V 计算一部分 attention
第 2 轮：K/V 沿环移动一格，继续计算
第 3 轮：再移动一格
第 4 轮：每张 GPU 都看过所有 K/V block
```

Ring Attention 的关键不是近似 attention，而是用 blockwise attention 和 ring 通信重排计算，让长序列分布到多设备上，并尽量把 K/V 通信与 attention 计算重叠。

> 🌐 **补充（论文 / RingAttention, ICLR 2024）**：RingAttention with Blockwise Transformers 提出通过 blockwise computation 和环形传递 K/V block，把长序列分布到多设备上，并重叠通信与计算，用于接近无限上下文的训练和推理。
> 来源：https://proceedings.iclr.cc/paper_files/paper/2024/hash/1119587863e78451f080da2a768c4935-Abstract-Conference.html

![序列并行示意](<../images/8-29-序列并行.png>)

---

### 六、激活内存与重计算：不同并行策略下的量化分析

#### 6.1 符号定义

下图的表格来自 Megatron-LM 论文 *Reducing Activation Recomputation in Large Transformer Models*，分析了不同并行策略组合下**单层 Transformer 的激活内存占用**。

| 符号 | 含义 |
|------|------|
| `s` | 序列长度（sequence length） |
| `b` | 批次大小（batch size） |
| `h` | 隐藏维度（hidden dimension） |
| `t` | 张量并行度（tensor parallelism size） |
| `a` | 注意力头维度（attention head dimension） |

`sbh` 是所有公式的公共因子，代表了激活内存的"基础规模"，括号里的项则表示不同并行策略带来的内存开销变化。

![序列并行详细示意](<../images/8-30-序列并行2.png>)

#### 6.2 逐行解读（引用自 Megatron-LM 论文）

| 配置 | 每层激活内存公式 | 核心含义 |
|------|:-----------------|----------|
| **无并行** | `sbh(34 + 5·a·s/h)` | 基准：所有激活保存在单 GPU，34 个 `sbh` 级激活 + 5 个注意力相关项 |
| **仅张量并行（TP baseline）** | `sbh(10 + 24/t + 5·a·s/(h·t))` | 34 被拆为 `10`（无法被 TP 拆分的项）+ `24`（可被 t 均分）；注意力项 `5·a·s/h` 也被除以 t |
| **TP + 序列并行（SP）** | `sbh(34/t + 5·a·s/(h·t))` | SP 沿序列维度进一步切分，大部分开销都被除以 t，只剩注意力相关激活未被消除 |
| **TP + 选择性重计算** | `sbh(10 + 24/t)` | 用选择性重计算丢弃了所有注意力相关激活（`5·a·s/(h·t)` 项消失），反向传播时重新计算 |
| **TP + SP + 选择性重计算（三者结合）** | `sbh(34/t)` | **终极方案**：所有 `sbh` 级激活被 t 均分，注意力激活被重计算消除，内存与 t 成反比 |

#### 6.3 34 这个数字的来源

`34` 是标准 GPT 式 Decoder 层中，**所有形状为 `[b, s, h]` 的激活张量的数量之和**（来自论文的逐项推导）：

| 模块 | 包含的激活项 | 系数 |
|------|-------------|:---:|
| 输入与残差 | 层输入、残差连接、LayerNorm | 2 |
| 自注意力块 | Q/K/V 线性投影、注意力输出投影、dropout、残差、LayerNorm | 10 |
| MLP 块 | 两个线性层（中间维度 4h）、激活函数、dropout、残差、LayerNorm | 22 |
| **合计** | | **2 + 10 + 22 = 34** |

每一个系数代表反向传播时需要保存的一个 `[b, s, h]` 张量。

#### 6.4 公式之间的对应关系

- `10 + 24/t` 是 34 的拆分：`10` 是 TP 无法拆的固定项，`24` 是可被 t 均分的项
- `5·a·s/h` 是注意力相关的激活（与 head dim `a` 成正比），选择性重计算将其消除
- 最终 `34/t`：所有项通过 TP+SP 拆分到 t 台 GPU + 选择性重计算去掉注意力开销

#### 6.5 核心结论

1. **内存线性扩展**：通过 TP + SP + 选择性重计算，单 GPU 激活内存从 `O(sbh)` 降到 `O(sbh/t)` —— 每增加一台机器，单卡激活内存按比例下降
2. **技术取舍**：选择性重计算不是免费的，它增加反向传播计算量（约 30%），但大模型训练中"内存瓶颈"比"算力瓶颈"更致命
3. **二次项问题**：注意力 score 是 `O(b·n_heads·s²)`，随序列长度二次增长。现代实现用 FlashAttention 避免显式保存完整 attention score 矩阵

数值感受：

设 `B=1`，`S=8192`，`H=4096`，BF16 每元素 2 bytes。

```
一个 [B, S, H] activation:
1 × 8192 × 4096 × 2 bytes ≈ 64 MiB

32 层仅 sbh 类激活:
64 MiB × 32 = 2 GiB（无并行）
64 MiB × 32 / 4 = 512 MiB（TP=4 + SP + 选择性重计算）
```

如果显式存 attention score，`n_heads=32`：

```
[B, n_heads, S, S]
= 1 × 32 × 8192 × 8192 × 2 bytes ≈ 4 GiB（仅一层！）
```

缓解手段：
- FlashAttention：避免显式保存 attention score 矩阵
- Activation Recomputation / Gradient Checkpointing：少存激活，反向时重算
- SP/CP：把 sequence 维度切到多张 GPU，降低每卡激活

#### 6.6 实验验证：激活内存优化效果对比

下图是 Megatron-LM 论文中的实验结果，用真实数据验证了上述公式推导的结论。

**图表关键元素**：

| 元素 | 含义 |
|------|------|
| 横轴 | 4 种参数规模：22B、175B、530B、1T |
| 对比项 | 每个规模下两组：基线（传统方案）vs 当前工作（TP + SP + 选择性重计算） |
| 纵轴 | 单卡显存占用（GB），红色虚线为 80GB（A100 显存上限） |
| 堆叠柱状 | 蓝色：参数 + 优化器状态；绿色：激活内存 |

**数据趋势**：

1. **基线方案**：即使是最小的 22B 模型，传统方案总显存也超过 100GB，远超 80GB 上限；1T 模型基线方案总显存接近 170GB。激活内存（绿色）随模型规模爆炸增长，成为显存瓶颈的主要来源。

2. **优化方案**：显存下降几乎全部来自激活内存（绿色），蓝色的参数 + 优化器状态基本不变。所有模型规模的总显存都被控制在 80GB 以内：
   - 22B：约 105GB → 约 60GB
   - 1T：约 165GB → 约 65GB

3. **核心结论**：传统方案中激活内存占显存一半以上；TP + SP + 选择性重计算三者结合，使激活内存实现 `O(sbh/t)` 的线性扩展，直接验证了 6.2 节公式的推导。

![激活值内存使用分析](<../images/8-28-激活内存的使用.png>)

![激活值重计算策略](<../images/8-38-激活值的重新计算.png>)

**上图解读：激活重计算的"算力换内存"权衡**

| 元素 | 含义 |
|------|------|
| 横轴 | 批次大小（batch size） |
| 纵轴 | 吞吐量（每秒序列数），越高越好 |
| 橙色线 | 无激活重计算：反向传播直接用保存好的中间结果 |
| 蓝色线 | 有激活重计算：丢弃部分中间激活，反向时重新计算 |
| 底部标注 | `t=8, p=16`：张量并行度=8，流水线并行度=16 |

**核心趋势**：

1. **无重计算（橙色线）**：batch size 约为 8 时就停止增长 —— 激活内存随 batch size 线性增长，很快耗尽显存，无法继续增大 batch。吞吐量卡在约 4。

2. **有重计算（蓝色线）**：batch size 可以一直增到 256，吞吐量超过 8，是无重计算峰值的 2 倍。重计算大幅降低了显存占用，让模型能使用远更大的 batch size。

3. **"内存自我补偿"的本质**：重计算本身增加了反向传播的算力开销（约 30%），但它换来了更大的 batch size，让 GPU 的并行计算单元被"喂饱"，算力利用率大幅提升。最终整体吞吐量不仅没降，反而显著超过无重计算方案。这就是为什么说"激活重计算不是性能毒药，而是突破显存瓶颈的关键手段"。

> 💡 **补充（官方文档 / Megatron Core CP）**：Megatron Core CP 文档指出，长上下文 OOM 主要来自 activation memory 随 sequence length 增长；完整 activation recomputation 可缓解 OOM，但会带来约 30% 的额外开销。CP 通过按 CP degree 分摊 activation footprint，减少每卡内存压力。
> 来源：https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/context_parallel.html

---

### 七、3D 并行：把 DP、PP、TP 组合起来

经典 3D 并行通常指：

```
DP: Data Parallelism      切 batch
PP: Pipeline Parallelism  切 depth / layers
TP: Tensor Parallelism    切 width / hidden dimension
```

如果还训练长上下文，可以再加入 CP：

```
总 GPU 数 = TP × PP × DP × CP
```

映射经验：

```
TP：通信最频繁，优先放同节点 NVLink/NVSwitch 内
PP：阶段间传 activation，适合跨节点连接不同 stage
DP：每 step 同步梯度，频率相对低，适合扩展到更多节点
CP：长上下文时使用，注意 attention 的 K/V 通信模式
```

![3D 并行示意](<../images/8-34-3D并行.png>)

#### 7.1 并行配置的系统化推导步骤

确定并行配置不是凭经验猜，而是有一套可复用的计算框架。以 **128 GPU（16 节点 × 8 GPU）、dense 70B 模型** 为例，逐步推导。

**已知条件**：

| 项 | 值 |
|----|---|
| 参数量 P | 70B = 70 × 10⁹ |
| 单节点 GPU 数 | 8（NVLink 互联） |
| 总 GPU 数 | 128 |
| 单卡显存 | 80 GB（A100） |
| 精度 | BF16（2 bytes/param） |
| 优化器 | Adam（状态量 = 2 × 参数量，FP32 存储 = 4 bytes/param） |

---

**Step 1：估算每张 GPU 需要的模型内存**

不考虑激活，仅模型参数 + 优化器状态 + 梯度的单卡内存：

```
参数内存（BF16）：    P × 2 bytes = 70B × 2  = 140 GB
梯度内存（BF16）：    P × 2 bytes = 70B × 2  = 140 GB
优化器状态（FP32）：  P × 8 bytes = 70B × 8  = 560 GB  （Adam 一阶矩 + 二阶矩 + FP32 参数副本）
────────────────────────────────────────────
总计：840 GB
```

> 💡 简化公式：每参数约 `2 + 2 + 8 = 12 bytes`（BF16 混合精度 + Adam）。有些实现用 `2 + 2 + 4 = 8 bytes`（优化器状态用 FP16），具体取决于实现。

单卡需承载：`840 GB / N`，其中 `N` 是参与分摊的 GPU 数。

> 📎 **来源追溯**：12 bytes/param 的分类方式来自 Hugging Face Accelerate 文档（Megatron-LM 插件）。Megatron-LM 原始论文（Shoeybi et al., 2019）使用 16 bytes/param（额外计入 4B FP32 master weight），两种算法本质相同，只是分类方式不同。

---

**Step 2：确定 TP —— 单层矩阵能否放进单卡**

TP 的核心约束不是总内存，而是**单层权重矩阵能否被单卡放下**。同时 TP 通信密集，优先选择节点内 NVLink 能覆盖的 GPU 数。

70B 模型（以 LLaMA-2 70B 为例），单层结构如下：

```
输入 X: [B, S, h]  (h = 8192, intermediate_size = 28672 = 3.5h)
    │
    ├─────────────────────────────────────────────────────────┐
    │                   Self-Attention 块                      │
    │                                                         │
    │   Wq: [h, h]     Wk: [h, h]     Wv: [h, h]             │
    │   8192×8192       8192×8192       8192×8192              │
    │      │               │               │                   │
    │      └───────┬───────┘               │                   │
    │              │  Q,K,V: [B,S,h]       │                   │
    │              ▼                       │                   │
    │        Attention(Q,K,V)              │                   │
    │              │                       │                   │
    │              ▼                       │                   │
    │         Wo: [h, h]                   │                   │
    │         8192×8192                    │                   │
    │              │                       │                   │
    ├──────────────┼───────────────────────┘                   │
    │              ▼                                           │
    │         + 残差连接                                       │
    │              │                                           │
    │         LayerNorm                                        │
    │              │                                           │
    ├─────────────────────────────────────────────────────────┤
    │                    FFN / MLP 块                          │
    │                                                         │
    │   SwiGLU: Y = (X @ W1 ⊙ SiLU(X @ W3)) @ W2             │
    │                                                         │
    │   W1: [h, 3.5h]    W3: [h, 3.5h]     W2: [3.5h, h]     │
    │   8192×28672        8192×28672         28672×8192        │
    │      │                  │                  │             │
    │      ▼                  ▼                  │             │
    │   gate_proj         up_proj                │             │
    │      │                  │                  │             │
    │      └──── ⊙(SiLU) ────┘                  │             │
    │              │                             │             │
    │              ▼                             │             │
    │         down_proj ◄────────────────────────┘             │
    │              │                                           │
    ├──────────────┼───────────────────────────────────────────┘
    │              ▼
    │         + 残差连接
    │              │
    │         LayerNorm
    │              ▼
    │         输出: [B, S, h]
    ▼
```

单层参数量统计：

| 矩阵 | 形状 | 参数量 |
|------|------|--------|
| Wq | [h, h] = [8192, 8192] | 67.1M |
| Wk | [h, h] = [8192, 8192] | 67.1M |
| Wv | [h, h] = [8192, 8192] | 67.1M |
| Wo | [h, h] = [8192, 8192] | 67.1M |
| W1 (gate_proj) | [h, 3.5h] = [8192, 28672] | 234.9M |
| W3 (up_proj) | [h, 3.5h] = [8192, 28672] | 234.9M |
| W2 (down_proj) | [3.5h, h] = [28672, 8192] | 234.9M |
| **合计** | | **~973M ≈ 0.97B** |

> 💡 **LLaMA-2 70B 的 intermediate_size 推导**：`28672 = 3.5h`，不是 `8/3·h ≈ 21845`。Meta 源码（`model.py` 第 307-348 行）中 FFN 中间维度的计算过程：
>
> ```
> 4 × dim = 32768
> × 2/3  → int(2 × 32768 / 3) = 21845     ← 这是 8/3·h 的来源（7B/13B 的默认值）
> × 1.3  → int(1.3 × 21845) = 28398       ← 70B 特有的 ffn_dim_multiplier = 1.3
> 取整到 256 的倍数 → 28672                ← 最终值 = 3.5h
> ```
>
> 来源：Meta 官方仓库 [meta-llama/llama `model.py`](https://github.com/meta-llama/llama/blob/master/llama/model.py)；HuggingFace 配置 `meta-llama/Llama-2-70b-hf/config.json`
>
> Attention 四个矩阵各 `h² ≈ 67M`，FFN 三个矩阵各 `h × 3.5h ≈ 235M`。FFN 的单个矩阵是 Attention 单个矩阵的 3.5 倍，因此 TP 的首要目标是切分 FFN 的大矩阵。

TP 需要满足：`单层参数 / TP ≤ 单卡可承载的模型内存`。但更关键的是，**TP 越大，层内通信越频繁**，所以优先选节点内 GPU 数的因数：

| TP 值 | 每卡单层参数 | 层内通信 | 是否合理 |
|:---:|------------|---------|---------|
| 1 | 0.97B × 2B = 1.94 GB | 无 | 单层可放下，但总模型内存分摊不够 |
| 2 | 0.97 GB | 2 卡通信 | 可行 |
| 4 | 0.24 GB | 4 卡通信 | **推荐**：通信在 NVLink 内，延迟可控 |
| 8 | 0.12 GB | 8 卡通信 | 可行但通信开销大，收益递减 |

**结论**：`TP = 4`（节点内 8 卡取一半，留余量给通信重叠）。

> 📎 **来源追溯**：NVLink 优先规则来自 Megatron-LM (Shoeybi et al., 2019) 和 Hugging Face Accelerate 文档。"单层矩阵能放进单卡"是工程经验法则，由 TP 的设计逻辑（shards 单层权重）推断而来，非论文直接给出的公式。

---

**Step 3：确定 PP —— 每个 stage 放几层**

TP 确定后，模型参数和优化器状态被 TP 张量并行分摊到 4 卡。但单个 TP 组（4 卡）仍需承载完整模型。PP 进一步按层切分：

```
70B 模型层数 L ≈ 80（LLaMA-2 70B 为 80 层）

PP = 4 时：
  每个 stage 的层数 = 80 / 4 = 20 层
  每个 stage 的参数 = 70B / 4 = 17.5B
  每卡参数（经 TP=4 分摊）= 17.5B / 4 = 4.375B
  每卡模型内存 = 4.375 × 12 bytes ≈ 52.5 GB
```

52.5 GB 已经接近 80 GB 上限，留约 27.5 GB 给激活内存 —— 这在中等序列长度（4K-8K）下通常够用。

**PP 的约束**：PP 会产生 pipeline bubble（气泡），`bubble 比例 ≈ (PP - 1) / (PP + micro_batch - 1)`。PP 越大气泡越大，所以不宜过大：

| PP 值 | bubble 比例（micro_batch=8） | 每卡模型内存 |
|:---:|:---:|:---:|
| 2 | 1/9 ≈ 11% | 105 GB（超出 80 GB） |
| 4 | 3/11 ≈ 27% | 52.5 GB |
| 8 | 7/15 ≈ 47% | 26.3 GB |

PP=2 内存不够；PP=8 气泡太大。**PP=4 是平衡点**。

> 📎 **来源追溯**：bubble 公式 `(PP-1)/(PP+M-1)` 来自 GPipe (Huang et al., 2019) 和 PipeDream (Narayanan et al., SOSP 2019)。"选最小 PP 使模型能放进单卡"的策略在 Sahani's Medium guide 中有明确描述。

---

**Step 4：确定 CP —— 激活内存是否超出预算**

TP=4、PP=4 确定后，检查激活内存是否在预算内：

```
每卡激活内存（无 CP，无重计算）= sbh(34 + 5·a·s/h) / t
  t = 4（TP）
  设 b=1, s=8192, h=8192, a=128（head_dim）
  
  sbh = 1 × 8192 × 8192 × 2 bytes ≈ 128 MiB
  34 + 5 × 128 × 8192 / 8192 = 34 + 640 = 674
  
  每层激活 ≈ 128 MiB × 674 / 4 ≈ 21.5 GiB
  20 层（PP=4）≈ 430 GiB  ← 远超 27.5 GB 预算！
```

即使开启选择性重计算（去掉 `5·a·s/h` 项）：

```
每层激活 = 128 MiB × 34 / 4 ≈ 1.09 GiB
20 层 ≈ 21.8 GiB  ← 在 27.5 GB 预算内
```

**结论**：8K 序列长度下，选择性重计算后激活内存可控，**不需要 CP（CP=1）**。

如果序列长度增加到 32K：

```
s = 32768，sbh = 1 × 32768 × 8192 × 2 ≈ 512 MiB
选择性重计算后：每层 = 512 MiB × 34 / 4 ≈ 4.35 GiB
20 层 ≈ 87 GiB  ← 超出 27.5 GB 预算
```

此时需要进一步优化。有两种重计算策略，区别在于**保留多少激活**：

```
选择性重计算（Selective）：
  只丢弃注意力 score（去掉 5·a·s/h 项），保留其余 34 个 [b,s,h] 激活
  公式：sbh(34/t)
  每层：512 MiB × 34 / 4 ≈ 4.35 GiB

完全重计算（Full）：
  只保留每层的输入激活（1 个 [b,s,h]），其余全部反向时重算
  公式：sbh(2/t)  （2 = 输入 + 残差连接，最少需要保存的量）
  每层：512 MiB × 2 / 4 ≈ 0.25 GiB
```

> 💡 选择性重计算是"省一部分"，完全重计算是"几乎全省"。代价是反向传播计算量增加更多（约 30-40%），但大模型训练中内存瓶颈通常比算力瓶颈更致命。

32K 场景下各方案对比：

| 方案 | 每层激活 | 20 层总计 | 是否可行 |
|------|---------|----------|---------|
| 选择性重计算（无 CP） | 4.35 GiB | 87 GiB | 超出预算 |
| 选择性重计算 + CP=2 | 2.18 GiB | 43.5 GiB | 仍超出 |
| 选择性重计算 + CP=4 | 1.09 GiB | 21.8 GiB | 可行，但 DP 降至 4 |
| **完全重计算 + CP=2** | 0.13 GiB | **2.5 GiB** | **可行，DP=8** |

**折中方案**：`CP=2` + 完全重计算 → 激活内存仅 2.5 GiB，DP 仍可保持 8，吞吐量损失最小。这也是为什么说"完全重计算可以进一步降低激活内存"——它把公式从 `34/t` 降到 `2/t`。

> 📎 **来源追溯**：激活内存公式 `sbh(34 + 5as/h)` 来自 Korthikanti et al., *Reducing Activation Recomputation in Large Transformer Models*, MLSys 2023 (arXiv:2205.05198)，为论文 Equation (1)。CP/SP 对激活内存的分摊效果在 Megatron Core CP 文档中有说明。

---

**Step 5：DP = 总 GPU / (TP × PP × CP)**

约束关系：`总 GPU = TP × PP × CP × DP`。本例中总 GPU = 128 是已知的硬件约束（16 节点 × 8 GPU），因此：

```
8K 上下文：  DP = 128 / (4 × 4 × 1) = 8
32K 上下文：DP = 128 / (4 × 4 × 2) = 4
```

反过来，如果先确定了所需的 DP（例如需要 DP=16 才能撑起足够的 global batch size），则可以推算所需总 GPU 数：

```
总 GPU = TP × PP × CP × DP = 4 × 4 × 1 × 16 = 256（需要 256 张 GPU）
```

> 📎 **来源追溯**：`总 GPU = TP × PP × CP × DP` 是 3D/4D 并行的通用约束，所有并行框架（Megatron-LM, DeepSpeed, FSDP）均确认。Sahani's Medium guide 和 Minjia Zhang 的 Megatron-LM 课程 slides 中均有明确使用。

---

**最终配置汇总**：

| 场景 | TP | PP | CP | DP | 总 GPU | 每卡模型内存 | 每卡激活内存 |
|------|:--:|:--:|:--:|:--:|:------:|:----------:|:----------:|
| 8K 上下文 | 4 | 4 | 1 | 8 | 128 | ~52 GB | ~22 GB（选择性重计算） |
| 32K 上下文 | 4 | 4 | 2 | 4 | 128 | ~52 GB | ~22 GB（CP=2 + 选择性重计算） |

**推导顺序总结**：

```
Step 1: 估算总模型内存 → 确定"每卡模型内存预算"
Step 2: TP = max(单层参数能放进单卡, 节点内因数) → 优先 NVLink 内
Step 3: PP = 使每卡模型内存 ≤ 预算，同时 bubble 可接受
Step 4: CP = 使每卡激活内存 ≤ 剩余预算（配合选择性重计算）
Step 5: DP = 总 GPU / (TP × PP × CP)
```

> 💡 **补充（官方文档 / Megatron Core 配置表）**：Megatron Core 文档给出的生产配置示例中，LLaMA-3 70B 在 64 GPU 上可用 `TP=4, PP=4, CP=2`；GPT-3 175B 在 128-512 GPU 上常见 `TP=4, PP=8, CP=1`。实际配置仍需根据 batch size、sequence length、显存和互联拓扑调参。
> 来源：https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html

#### 7.2 策略选择小结

```
模型能放进单卡？
  ├─ 能：先用 DP / DDP，必要时加 ZeRO/FSDP
  └─ 不能：
      ├─ 单层矩阵太大：加 TP
      ├─ 层数太多或每卡参数太多：加 PP
      ├─ 长上下文 activation 爆显存：加 CP 或 activation recomputation
      └─ 需要更高吞吐：增加 DP，但注意 global batch size 上限
```

![其他并行策略](<../images/8-31-其他的并行策略.png>)

![并行策略总结](<../images/8-32-并行策略总结.png>)

![张量并行的最优配置](<../images/8-37-张量并行的最优解.png>)

---

### 八、补充：Transformer Engine 与 FP8

Transformer Engine 是 NVIDIA 为 Transformer 训练/推理提供的高性能组件库，在 Hopper 架构 GPU（如 H100）上支持 FP8。更准确的说法不是“自动在 FP8 和 FP16 间随意切换”，而是：

- H100 支持 FP8 Tensor Core 路径；
- Transformer Engine 提供 FP8 相关 kernel、scaling、recipe 和 PyTorch API；
- 典型用法是在合适的 Transformer 层中启用 FP8 autocast，同时保留必要的高精度状态以维持稳定性。

> 🌐 **补充（官方文档 / Transformer Engine）**：NVIDIA Transformer Engine 文档说明，H100 引入 FP8 datatype 支持，TE 提供面向 Transformer 的优化 building blocks 和类似 automatic mixed precision 的 API；FP8 包含 E4M3 和 E5M2 两种格式，分别偏向精度和动态范围。
> 来源：https://docs.nvidia.com/deeplearning/transformer-engine-releases/release-0.5.0/user-guide/examples/fp8_primer.html

---

## 🧠 本模块问题

请在下方回答以下问题后，输入 `提交作业` 提交。

**Q1**：MLP 张量并行中，`W1` 为什么适合按列切分，`W2` 为什么适合按行切分？请用矩阵形状说明：为什么最后需要 `All-Reduce(sum)`，而不是 `All-Gather`？

**Q2**：请区分 Sequence Parallelism、Context Parallelism 和 Ring Attention。要求说明三者分别切分什么、主要解决什么问题、Attention 阶段是否需要跨 GPU 通信。

**Q3**：假设你有 `128 GPU = 16 节点 × 8 GPU`，要训练一个 dense 70B 模型。请给出两套配置：一套用于 8K 上下文，一套用于 32K 长上下文。每套配置都写出 `TP × PP × CP × DP = 128`，并解释为什么这么分配。

---

<!-- 学习者作答区（请在此处填写你的答案） -->

**A1**：

MLP 张量并行的步骤是

H = GELU(X@W1), Y = H@W2

假设有 2 GPU 来进行 Tensor Parallelism
W1 按照列切分，将 W1 分为 [W1_0 | W1_1]
则 X@W1 可以拆分为 [X@W1_0 | X@W1_1]
因为 GELU 可以使用 pointwise 的 tanh 来近似
所以 H = GELU(X@W1) 可以分开在 GPU0 和 GPU1 计算，再 concatenate
即 [H_0 | H_1] = GELU(X@[W1_0, W1_1])
而 step2 Y = H@W2 可以转换为 [H_0 | H_1] @ W2
将 W2 按照行切分，可以得到 
H @ W2 = H_0 @ W2_0 + H_1 @ W2_1
在 GPU0 和 GPU1 上分别计算的结果，然后相加即可得到完整的 MLP 结果
所以最后一步 所有 GPU 上都要同步相加的结果，自然是 All-Reduce 了


**A2**：

Sequence Parallelism 是 序列并行，适用于上下文特别长的场景，对于 [Batch, Sequence, head_dim], 切分 Sequence 的维度，把可分片的 activation 沿 sequence 维度分散保存。

Context Parallelism 也是沿着 Sequence 维度切分，但是比 SP 更激进，CP 会切分网络输入和所有 activation，使每张 GPU 只处理一部分上下文，如果需要 attention 计算时，再通过通信交换 K、V 信息。

Ring Attention 应该是 CP 中一种交换 K、V 信息的一种通信方案，向右发送数据，等到 N-1 步之后，所有设备都拥有了各自的信息（这里 N 指节点内分布存储 K、V 信息的 GPU）


**A3**：

70B 模型，如果用 BF16 计算参数，Optimizer 部分需要FP32，那么其占用的总内存数需要

70 B * ( 2 + 2 + 2 * 4) Bytes = 840 GB 对于 80GB 显存的 A100 来说显然无法满足

先做 TP 部分并行，根据 70B llama 模型 head_dim 是 8192，
那么整个 attention block 的 参数量大致应该是
4 * h * h + 3 * 3.5 * h * h = 14.5 * h * h = 14.5 * 8192 * 8192 ≈ 973M 参数量
因此整个 attention block 参数占用内存 0.97B * 2Bytes ≈ 1.94G
整个模型 840GB 肯定放不进一个 GPU 的，所以 tensor 部分需要分层
TP = 4 时，节点内 8 卡取一半，留余量给通信重叠

再做 PP 部分并行，70B 模型大约 80 层
PP = 4 时，每个 stage 有 20 层
每个 stage 的参数量是 70B / 4 = 17.5B
每个卡参数分担下（TP=4）则是 17.5B/4 = 4.375B
每个 GPU 的模型内存占用是 4.375B * 12 Bytes = 52.5 GB，满足 80GB 显存
根据 气泡率计算公式 (PP - 1) / (M + PP - 1) , 如果 batch 一定（M恒定），那么 PP 越大气泡率越高，所以 PP 不宜太大，这里选择 4

CP 部分并行，考虑 8K 和 32K 两种序列长度

对于 8K 长度，s = 8192，则 sbh = 1 * 8292 * 8192 * 2 Bytes = 128 MB

如果 无 CP， 则 每层激活 占用是 sbh * (34 + 5as/h)/t = sbh * ( 34 + 5 * 128 * 8192 / 8192)/4
(a 这里选择 128，即 attention head dim 是 128)
即 128MB * 674/4 = 21.0625 GB，但是这里每个 stage 有 20 层
则总共需要 421.25 GB 超过单卡显存剩余大小( 80 - 52.5 = 27.5 GB)
因此只能加入选择重计算，则
sbh * (34/t) = 128MB * 34 / 4 = 1.0625GB，20 层大小为 21.25GB 满足要求
因此 8K 数据长度不需要 CP，只做 选择性重计算，可以满足显存需求

对于 32K 长度，s = 32768，则 sbh = 1 * 32768 * 8292 * 2 Bytes = 512 MB
如果只做选择重计算，那么每层激活占用显存大小是
sbh * (34/t) = 4.25 GB, 20 层需要 85 GB 大小，无法满足要求
只能使用 完全重计算，sbh * (2/t) = 0.25 GB，20 层是 5GB 大小，满足需求
但是如果 CP = 1 的话，DP 只能为 4，如果 DP 为 8 则显存大小不够，所以 32K 
使用 CP 为 2， 则 DP 为 128 /（4 * 4 * 2） = 4

因此这里对于 8K 上下文，选择 
DP = 8, TP = 4, PP = 4, CP = 1，需要 GPU 128 刚好满足要求
对于 32K 上下文，选择
DP = 4, TP = 4, PP = 4, CP = 2,

---

<!-- 教师批改区（提交作业后由导师填写，请勿手动修改） -->

### 📝 批改结果

**Q1 批改**：逻辑链完整，从 W1 列切 → GELU 逐元素 → W2 行切 → 求和 → All-Reduce，覆盖了核心推理。矩阵形状推导正确。有一点需要修正：你提到"GELU 可以使用 pointwise 的 tanh 来近似"，这个说法不准确——GELU 本身就是逐元素函数（element-wise），不需要"近似"才能分开计算，直接就可以在各 GPU 上独立完成。tanh 只是 GELU 的一种数值近似实现方式，与"能否分开计算"无关。 — 得分：**9/10**

<details>
<summary>📖 Q1 参考答案</summary>

**W1 按列切分的原因**：

W1 的输出维度是 `4d_model`（或 SwiGLU 中的 `8/3·h`）。按列切分后：

```
W1 = [W1_0 | W1_1]，每卡持有 [d_model, 2d_model]
GPU0: H0 = GELU(X @ W1_0)   # [B, S, 2d_model]
GPU1: H1 = GELU(X @ W1_1)   # [B, S, 2d_model]
```

关键：GELU 是逐元素函数（element-wise），不依赖其他通道的数据，所以 H0 和 H1 可以**完全独立**地在各自 GPU 上计算，无需通信。

**W2 按行切分的原因**：

H 已经被切成 `[H0 | H1]`，W2 按行切刚好匹配：

```
W2 = [W2_0; W2_1]，每卡持有 [2d_model, d_model]
GPU0: Y0 = H0 @ W2_0   # [B, S, d_model]
GPU1: Y1 = H1 @ W2_1   # [B, S, d_model]
```

矩阵乘法的分配律：`[H0|H1] @ [W2_0; W2_1] = H0@W2_0 + H1@W2_1`

**为什么是 All-Reduce 而不是 All-Gather**：

- 最终结果是**求和**（`Y = Y0 + Y1`），不是拼接
- All-Gather 会把 `[Y0, Y1]` 拼成 `[B, S, 2d_model]`，维度错误
- All-Reduce(sum) 让每张 GPU 都拿到相同的 `Y = Y0 + Y1 = [B, S, d_model]`，继续进入下一层

**常见误解**：认为"列切 → 输出需要拼接 → All-Gather"。实际上列切 W1 的输出确实是拼接的（`H = [H0|H1]`），但这发生在 W2 的输入端，不是最终输出端。W2 行切后，最终输出是求和，所以用 All-Reduce。

</details>

---

**Q2 批改**：CP 和 Ring Attention 的描述基本正确。但 SP 的描述有明显偏差：你说 SP "适用于上下文特别长的场景"，这是把 SP 和 CP 混淆了。SP 的核心定位是**配合 TP 使用**，切分的是 LayerNorm、Dropout 等 TP 无法拆分的激活，而不是为了解决长上下文问题。Ring Attention 的"向右发送数据"说法也不够准确——K/V block 沿环形拓扑传递，方向不限，核心是 blockwise attention + 通信计算重叠。 — 得分：**7/10**

<details>
<summary>📖 Q2 参考答案</summary>

| 概念 | 切分什么 | 解决什么问题 | Attention 是否需要跨 GPU 通信 |
|------|----------|-------------|---------------------------|
| **SP** | LayerNorm、Dropout 等激活，沿 sequence 维度切分 | 配合 TP 减少激活内存（TP 的 All-Reduce 需要完整激活，SP 把非 TP 部分的激活沿 sequence 分散） | 否，SP 不参与 attention 计算 |
| **CP** | 网络输入和几乎所有 activation，沿 sequence 切分 | 训练长上下文，降低每卡 activation 内存 | 是，需要交换 K/V（通过 all-gather、p2p 或 ring） |
| **Ring Attention** | Q/K/V 按 sequence block 切分 | 极长上下文的 attention 计算，通信与计算重叠 | 是，K/V block 沿环形传递，每轮计算一部分 attention |

**关键区别**：
- SP 是 TP 的附属优化，不独立使用；切分范围小（仅 LayerNorm/Dropout 激活）
- CP 是独立的并行维度，切分范围大（输入 + 所有 activation）
- Ring Attention 是 CP 中 attention 通信的一种具体实现方式，不是独立的并行维度

**SP 为什么配合 TP**：TP 的 All-Reduce 需要每卡持有完整 `[B, S, H]` 激活，但 LayerNorm/Dropout 这些非 TP 操作其实不需要完整 sequence。SP 把这些激活沿 sequence 分散到 TP 组内的各卡，节省显存。通信发生在 TP 边界（All-Reduce 前后用 Reduce-Scatter/All-Gather 转换）。

</details>

---

**Q3 批改**：推导过程完整，Step 1-5 的计算逻辑正确，8K 和 32K 两个场景都给出了合理配置。两个小问题：① 第 798 行和 808 行 `s = 8192` 写成了 `8292`（笔误）；② 32K 场景的推理逻辑有跳跃——先说"只能使用完全重计算"，然后突然跳到"CP=2, DP=4"，没有解释为什么选择 CP=2 而不是继续用完全重计算 + CP=1。正确思路应该是：先评估 CP=2 + 选择性重计算是否可行（85/2=42.5GB 仍超出），再退到 CP=2 + 完全重计算（5/2=2.5GB，可行），这样 CP=2 的选择才有依据。 — 得分：**8/10**

<details>
<summary>📖 Q3 参考答案</summary>

**Step 1：估算模型内存**

```
70B × 12 bytes/param = 840 GB（BF16 参数 + 梯度 + Adam FP32 状态）
```

单卡 80GB 无法承载，必须并行分摊。

**Step 2：确定 TP = 4**

70B 单层参数约 0.97B（含 Attention 4×h² + FFN 3×h×3.5h），单层权重约 1.94GB。TP 需在节点内 NVLink 覆盖范围内，取节点内 8 卡的一半：**TP = 4**。

**Step 3：确定 PP = 4**

70B 约 80 层。PP=4 时每 stage 20 层，每卡模型内存 = 17.5B/4 × 12 bytes ≈ 52.5GB。Bubble 率 = (4-1)/(M+4-1)，PP 不宜过大。**PP = 4**。

**Step 4：确定 CP（按序列长度）**

剩余显存预算 = 80 - 52.5 = 27.5 GB。

**8K 场景**：sbh = 128 MiB
- 选择性重计算：每层 = 128 × 34/4 ≈ 1.09 GiB，20 层 ≈ 21.8 GiB ≤ 27.5 GiB ✓
- **CP = 1**（不需要 CP）

**32K 场景**：sbh = 512 MiB
- 选择性重计算：每层 = 512 × 34/4 ≈ 4.35 GiB，20 层 ≈ 87 GiB >> 27.5 GiB ✗
- CP=2 + 选择性重计算：87/2 ≈ 43.5 GiB >> 27.5 GiB ✗
- CP=2 + 完全重计算：每层 = 512 × 2/4 ≈ 0.25 GiB，20 层 ≈ 5 GiB，5/2 ≈ 2.5 GiB ≤ 27.5 GiB ✓
- **CP = 2** + 完全重计算

**Step 5：DP = 总GPU / (TP × PP × CP)**

| 场景 | TP | PP | CP | DP | 总 GPU | 每卡模型内存 | 每卡激活内存 |
|------|:--:|:--:|:--:|:--:|:------:|:----------:|:----------:|
| 8K | 4 | 4 | 1 | 8 | 128 | 52.5 GB | 21.8 GB（选择性重计算） |
| 32K | 4 | 4 | 2 | 4 | 128 | 52.5 GB | 2.5 GiB（完全重计算） |

</details>

---

**综合评价**：三道题的核心知识点都掌握了，Q1 的 MLP TP 推导尤其扎实。主要薄弱点是 Q2 中 SP 的定位（容易和 CP 混淆）和 Q3 中推理链的完整性（跳跃了中间判断步骤）。建议复习第五节 SP 部分，重点理解 SP 与 TP 的配合关系。

**批改时间**：2026-05-14
