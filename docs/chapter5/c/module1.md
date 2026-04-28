# 第 5 章：混合专家模型 — 模块 1：MoE 核心概念与路由机制

> 📍 学习进度：第 5 章，第 1 / 4 模块
> 📅 生成时间：2026-04-22（修订：2026-04-24）

---

## 学习目标

- 理解 MoE 的核心思想：容量大但计算稀疏
- 掌握 Top-K 路由的两种模式 TC（Token Choice）和 EC（Expert Choice）
- 理解负载均衡问题及三种独立解法：Auxiliary Loss、容量控制、动态 Bias
- 了解哈希路由（如 LSH）作为非学习式路由的设计思路

---

## 核心内容

### 一、MoE 的核心思想

MoE（Mixture of Experts，混合专家模型）解决的核心矛盾：**模型规模 vs 计算成本**。

传统稠密模型中，每个 token 都经过全部参数的计算。MoE 则将单个 FFN 替换为 **N 个并行专家**，通过路由机制每次只激活 **k 个**（k << N），从而：

- **参数容量**：等于所有专家参数之和（非常大）
- **实际计算量**：等于 k 个专家的计算量（与 k 成正比，与 N 无关）

```
稠密模型:                          MoE 模型:
━━━━━━━━━━━━━━━━━━━━━━           ━━━━━━━━━━━━━━━━━━━━━━

  x ──→ [FFN (大)] ──→ out         x ──→ 路由器 (W_g)
                                            │
                                       ┌────┼────┐
                                    E₀   E₂   E₇  ← 只激活 k=3 个
                                    │    │    │
                                    └────┴────┘
                                      加权求和
                                         │
                                        out

参数量: d×4d×2 = 8d²              参数量: N × (d×4d×2) = 8Nd²
FLOPs:  ~8d²                      FLOPs:  ~8kd²  (k << N)
```

> **关键认知**：MoE 的优势在**大规模**时最明显。小规模或资源受限环境下，MoE 可能不如对应的稠密模型。

### 二、路由机制详解

当前主流路由方式是基于可学习门控得分的 **Top-K 路由**，核心公式：

$$y = \sum_{i \in \mathcal{T}} G_i(x) \cdot E_i(x)$$

其中 $\mathcal{T}$ 是 Top-K 选出的专家索引集合。

**路由计算两步走**：

```
输入 x (d维)
    │
    ↓ 打分: h(x) = x · W_g        W_g: [d, N]，输出 [N] 维得分向量
    │
    ↓ 稀疏化: 保留分数最高的 k 个，其余置零
    │         对 k 个得分做 softmax 归一化
    ↓
  G(x): [N] 维权重向量（只有 k 个非零）
```

$W_g$ 是路由器中的**可学习线性投影层**，将 token 特征映射为 N 维得分向量，表示 token 与各专家的匹配程度。

对应的路由器代码（来自 [Top-K TC.py](<../Top-K TC.py>)）：

```python
class TC_MoE(nn.Module):
    def __init__(self, dim, num_experts, k):
        super().__init__()
        self.num_experts = num_experts
        self.k = k
        # 路由器：线性投影层，将 token 特征映射为专家得分
        self.router = nn.Linear(dim, num_experts)
        # N 个并行的专家 FFN
        self.experts = nn.ModuleList([Expert(dim) for _ in range(num_experts)])
```

```python
# forward 中的路由计算
gate_scores = F.softmax(self.router(x), dim=-1)  # [B, E] — softmax 归一化
topk_scores, topk_idx = gate_scores.topk(self.k, dim=-1)  # [B, k] — 保留 top-k
```

#### 各模型的 Top-K 选择

| 模型 | Top-K | 专家总数 | 总参数 / 激活参数 | 备注 |
|------|:-----:|:--------:|:-----------------:|------|
| Switch Transformer | 1 | 8~2048 | 7.4B~1.6T / 1.1B~1.6B (Fedus et al., 2022) | 极简路由降低通信 |
| GShard / Grok / Mixtral | 2 | 8 | 46.7B / 14.3B (Mixtral 8×7B) | 最主流配置 |
| Qwen1.5-MoE / DBRX | 4 | 60 | 14.3B / 2.7B (Qwen) | 更多专家更精细 |
| DeepSeek-V3 | 8 (1共享+7路由) | 256 | 671B / 37B | 极致细粒度专家 |

> 来源：Stanford CS336 Lecture 4 (Tatsu Hashimoto)；Switch Transformer 有多个变体（Switch-Base 到 Switch-C），从 8 专家到 2048 专家不等

**MoE 的实际效果**（DeepSeek 小规模实验，来源：CS336 课件）：
- Dense 0.2B vs MoE 2.0B(0.2B activated)：**FLOPs 相同**，但 Pile Loss 从 2.060 降到 1.881
- Qwen1.5-MoE-A2.7B（2.7B 激活）性能接近 Qwen1.5-7B（7B 全激活），GSM8K 上 MoE 反超（61.5 vs 47.5）

#### 噪声门控（Noisy Gating）

Shazeer et al. (2017) 提出的完整噪声门控公式，在打分阶段加入可学习的噪声以增强专家多样性：

$$H(x)_i = (x \cdot W_g)_i + \text{StandardNormal}() \cdot \text{Softplus}((x \cdot W_{\text{noise}})_i)$$

$$G(x) = \text{Softmax}(\text{KeepTopK}(H(x), k))$$

其中 $\text{KeepTopK}(v, k)_i$ 定义为：保留 top-k 位置的原值，其余设为 $-\infty$。

```
噪声门控 vs 简单门控:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

简单门控 (Mini LLM 代码):
  logits = x · W_g
  noise_std = Sigmoid(x · W_noise)     ← 用 Sigmoid 映射到 [0,1]
  logits = logits + randn · noise_std

完整噪声门控 (Shazeer 2017):
  noise_scale = Softplus(x · W_noise)   ← 用 Softplus 映射到 [0, +∞)
  logits = x · W_g + randn · noise_scale

区别: Sigmoid 有上界(1), Softplus 无上界 → Softplus 给噪声更大自由度
```

> 💡 **补充（PDF 课件 / CS336 Lecture 4）**：Mini LLM+MoE.py 中使用 `Sigmoid` 控制噪声标准差，而 Shazeer 原始论文使用 `Softplus`。两者作用相同（数据依赖的噪声幅度），但 Softplus 允许更大的噪声范围，理论上能更充分地探索专家空间。

#### 2.1 TC 模式（Token Choice）— "学生找导师"

每个 token 在专家维度（列维度）上选择 Top-K 专家。

```
gate_scores: [B_total, N]    ← 每个 token 对每个专家的得分
                 ↓
对每行（每个 token）取 Top-K 列（专家）

Token 0:  [0.1, 0.3, 0.8, 0.05, ...]  → 选 Expert 2, Expert 1
Token 1:  [0.4, 0.1, 0.2, 0.7,  ...]  → 选 Expert 3, Expert 0
Token 2:  [0.1, 0.3, 0.8, 0.05, ...]  → 选 Expert 2, Expert 1  ← 重复选热门
...
```

TC 模式的路由代码（来自 [Top-K TC.py](<../Top-K TC.py>)）：

```python
def forward(self, x, tokens=None, verbose=False):
    B, D = x.shape
    gate_scores = F.softmax(self.router(x), dim=-1)  # [B, E]
    topk_scores, topk_idx = gate_scores.topk(self.k, dim=-1)  # [B, k]

    out = torch.zeros_like(x)
    # 遍历每个 Top-K 位置
    for i in range(self.k):
        expert_ids = topk_idx[:, i]            # 当前位置的专家索引 [B]
        expert_weight = topk_scores[:, i]       # 当前位置的权重 [B]
        expert_output = torch.zeros_like(x)
        for e_id, expert in enumerate(self.experts):
            mask = (expert_ids == e_id).float().unsqueeze(1)  # 属于该专家的 token
            if mask.sum() == 0:
                continue
            expert_output += expert(x * mask)  # 只把属于该专家的 token 送入
        out += expert_output * expert_weight.unsqueeze(1)  # 加权累加
    return out
```

**实验结果**（10 专家, k=2, 33 tokens）：
> 专家负载：`[13, 13, 16, 14, 9, 6, 20, 19, 18, 4]` — 严重不均衡！

**优点**：每个 token 都会被处理，语义完整性好
**缺点**：少数"热门"专家被过度使用，"冷门"专家闲置 → **负载不均衡**

#### 2.2 EC 模式（Expert Choice）— "导师挑学生"

每个专家在 token 维度（行维度）上选择 Top-K token。

```
gate_scores: [B_total, N]
       ↓ transpose
scores_T: [N, B_total]       ← 每个专家对每个 token 的得分
       ↓
对每行（每个专家）取 Top-K 列（token）

Expert 0: 选 Token 3, Token 7       ← 每个专家固定选 k 个
Expert 1: 选 Token 0, Token 5
Expert 2: 选 Token 1, Token 3       ← Token 3 被多个专家选
...
```

EC 模式的关键代码（来自 [Top-K EC.py](<../Top-K EC.py>)）：

```python
def forward(self, x, tokens=None, verbose=False):
    B_total, D = x.shape
    gate_scores = F.softmax(self.router(x), dim=-1)  # [B*L, E]

    # EC 关键区别：转置后每个专家选 Top-K token
    scores_T = gate_scores.transpose(0, 1)            # [E, B*L]
    topk_scores, topk_idx = scores_T.topk(min(self.k, B_total), dim=-1)  # [E, k]

    # 构建 dispatch_weights: 每个 token 对每个专家的权重
    dispatch_weights = x.new_zeros((B_total, self.num_experts))
    for e in range(self.num_experts):
        for t_idx, s in zip(topk_idx[e].tolist(), topk_scores[e].tolist()):
            dispatch_weights[t_idx, e] = s

    # 专家计算
    out = torch.zeros_like(x)
    for e_id, expert in enumerate(self.experts):
        mask = (dispatch_weights[:, e_id] > 0).float().unsqueeze(1)
        if mask.sum() == 0:
            continue
        expert_out = expert(x * mask)
        out += expert_out * dispatch_weights[:, e_id].unsqueeze(1)
    return out
```

**实验结果**（10 专家, k=2, 33 tokens）：
> 每个专家恰好处理 2 个 token — 负载完美均衡
> 但有 token 如 '混'、'合'、'模'、'型' 从未被任何专家选中 — **语义丢失**

**优点**：天然负载均衡，专家能力均匀发展
**缺点**：部分 token 可能完全未被处理，信息缺失

#### 2.3 TC vs EC 对比

```
              TC (Token Choice)           EC (Expert Choice)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━
视角:    token 选专家                  专家选 token
类比:    学生找导师                    导师挑学生
负载:    不均衡（热门过载）             完美均衡（每个专家固定 k）
语义:    完整（每个 token 都处理）       有丢失（部分 token 被丢弃）
适用:    大多数生产模型                 研究实验、需要均衡场景

核心权衡: 信息完整性  vs  负载均衡
```

> **可能的误解澄清**：EC 模式下每个专家选 k 个 token，不代表每个 token 恰好被选 k 次。一些 token 可能被多个专家选中，一些可能完全没人选——这正是语义丢失的来源。

![TC模式](<../images/5.1.png>)
*图5.1 词元选择模式（TC）*

![EC模式](<../images/5.2.png>)
*图5.2 专家选择模式（EC）*

### 三、负载均衡策略

TC 模式的负载不均衡是 MoE 工程中最核心的挑战。本节介绍三种独立的解法，分别解决不同层面的问题：

```
问题 1：训练层面的路由坍缩 → Auxiliary Loss（本节 3.1）
问题 2：推理层面的专家过载 → 容量控制 + Token 丢弃（本节 3.2）
问题 3：辅助损失对主目标的干扰 → 无辅助损失动态均衡（本节 3.3）
```

#### 3.1 辅助负载均衡损失（Auxiliary Loss）

**来源**：Switch Transformer (Fedus et al., 2022)，用于防止路由坍缩——所有 token 涌向同一个专家。

**公式**：

$$\mathcal{L}_{\text{aux}} = \alpha \cdot N \cdot \sum_{i=1}^{N} f_i \cdot P_i$$

```
L_aux = α × N × Σᵢ(fᵢ × Pᵢ)

α  → 超参数，通常 0.01（控制均衡约束的强度）
N  → 专家总数
fᵢ → 专家 i 实际收到的 token 比例（不可微）
Pᵢ → 路由器分配给专家 i 的概率总和（可微）
```

**完整计算过程**（以 4 个专家、6 个 token、Top-1 为例）：

```
步骤 1：计算路由概率（softmax 输出）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

假设路由器对 6 个 token 的 softmax 输出为：

Token 0: [0.40, 0.30, 0.20, 0.10]  → Top-1 → Expert 0
Token 1: [0.35, 0.25, 0.25, 0.15]  → Top-1 → Expert 0
Token 2: [0.10, 0.70, 0.10, 0.10]  → Top-1 → Expert 1
Token 3: [0.45, 0.15, 0.30, 0.10]  → Top-1 → Expert 0
Token 4: [0.05, 0.10, 0.15, 0.70]  → Top-1 → Expert 3
Token 5: [0.30, 0.20, 0.40, 0.10]  → Top-1 → Expert 2

路由结果：
  Expert 0 ← Token 0, 1, 3（3 个 token）
  Expert 1 ← Token 2（1 个 token）
  Expert 2 ← Token 5（1 个 token）
  Expert 3 ← Token 4（1 个 token）
  → 不均衡！Expert 0 占了一半
```

```
步骤 2：计算 fᵢ（实际分配比例）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

f₀ = 3/6 = 0.500  ← Expert 0 收到 3 个 token
f₁ = 1/6 = 0.167
f₂ = 1/6 = 0.167
f₃ = 1/6 = 0.167

⚠️ fᵢ 来自 argmax（离散选择），不可微分，无法直接反向传播
```

```
步骤 3：计算 Pᵢ（路由器概率总和）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

P₀ = 0.40 + 0.35 + 0.10 + 0.45 + 0.05 + 0.30 = 1.650
P₁ = 0.30 + 0.25 + 0.70 + 0.15 + 0.10 + 0.20 = 1.700
P₂ = 0.20 + 0.25 + 0.10 + 0.30 + 0.15 + 0.40 = 1.400
P₃ = 0.10 + 0.15 + 0.10 + 0.10 + 0.70 + 0.10 = 1.250

✓ Pᵢ 来自 softmax 输出（连续值），可微分，可以反向传播
```

```
步骤 4：计算 L_aux
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

fᵢ × Pᵢ 的乘积：
  f₀ × P₀ = 0.500 × 1.650 = 0.825  ← 最大！因为 f₀ 高且 P₀ 也高
  f₁ × P₁ = 0.167 × 1.700 = 0.284
  f₂ × P₂ = 0.167 × 1.400 = 0.234
  f₃ × P₃ = 0.167 × 1.250 = 0.209

L_aux = α × N × Σ(fᵢ × Pᵢ)
      = 0.01 × 4 × (0.825 + 0.284 + 0.234 + 0.209)
      = 0.01 × 4 × 1.552
      = 0.0621
```

```
步骤 5：理想均衡情况对比
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

如果每个专家恰好收到 1.5 个 token（fᵢ = 1/4 = 0.25 对所有 i），
且路由器概率均匀（Pᵢ = 1.5 对所有 i）：

  L_aux = 0.01 × 4 × 4 × (0.25 × 1.5)
        = 0.01 × 4 × 1.5
        = 0.06    ← 最小值

当前 L_aux = 0.0621 > 0.06
→ 梯度推动路由器降低 f₀ 和 P₀ → 减少 Expert 0 的"吸引力"
→ 直到各专家的 fᵢ 趋近均匀
```

**为什么用 fᵢ × Pᵢ 而不是直接惩罚 |fᵢ - 1/N|？**

```
fᵢ 不可微（argmax 的结果）→ 无法反传梯度
Pᵢ 可微（softmax 输出）→ 可以传梯度

fᵢ × Pᵢ 的巧妙之处：
  ① fᵢ 作为权重，放大了"被频繁选中专家"的 Pᵢ 的梯度
  ② Pᵢ 作为可微项，将梯度传导回路由器的 W_g
  ③ 路由器学到：降低热门专家的得分 → token 分散到其他专家

直觉：如果 Expert 0 被很多 token 选中（f₀ 大），
     同时路由器也给它很高的概率（P₀ 大），
     那么 f₀×P₀ 就很大 → 贡献大 → 路由器被惩罚 → 降低 Expert 0 的分数
```

**L_aux 与 Router Z-loss 的区别**（第 2 模块有详细讨论）：

```
                Auxiliary Loss              Router Z-loss
                ━━━━━━━━━━━━━━              ━━━━━━━━━━━━━━
目的：           负载均衡                    数值稳定性
                让每个专家收到               防止 logits 过大
                差不多的 token              导致 exp() 溢出

公式：          α × N × Σ(fᵢ × Pᵢ)        λ × log²(Σexp(zᵢ))

解决问题：       路由坍缩                    计算溢出
                （所有 token 涌向            （FP16 下 z > 11
                 同一个专家）                 就溢出）

超参数：         α ≈ 0.01                   λ ≈ 0.001
```

> 🌐 **补充（PDF 课件 / CS336）**：负载均衡损失（Load Balancing Loss, LBL）的消融实验显示：无 LBL 时，训练不稳定且验证 loss 更高；极端情况下出现**专家坍塌**（单个专家处理近 100% 的 token）。LBL 不仅提升硬件利用率，还实际**提升学习质量**。

#### 3.2 容量控制 + Token 丢弃

Auxiliary Loss 在训练层面推动均衡，但无法保证每次 forward 都均衡。容量控制是一种**硬限制**机制，直接在推理层面防止专家过载。

```
容量计算公式:

  capacity = (tokens_per_batch / num_experts) × capacity_factor

  capacity_factor 通常为 1.0~1.25
  capacity_factor = 1.0 时，每个专家恰好处理平均数量的 token
  capacity_factor > 1.0 时，留一些余量
```

当某专家被分配的 token 数 > capacity 时，执行 **Token Dropping 四步流水线**（来源：CS336 课件）：

```
① Routing:
   每个 token 计算得分 → 选定 Top-K 专家
   gate_scores: [N_tokens, N_experts] → topk_idx: [N_tokens, k]

② Permutation (重排):
   将 token 按专家分组重排
   Expert 0: [token_3, token_7, token_12, ...]
   Expert 1: [token_0, token_5, ...]
   ⚠️ 如果某专家的 token 数 > capacity，多余的 token 被丢弃！

③ Computation:
   每个专家只处理分配给它的 token（容量内）
   被丢弃的 token 仅通过残差连接传到下一层

④ Un-permutation (还原):
   将专家输出按原始顺序放回
   被丢弃位置保留残差值（未经专家处理）
```

**关键洞察**（来自 CS336 课件）：在批处理推理中，**其他用户的查询可能导致你的 token 被丢弃**！因为容量限制是按整个 batch 计算的，某个专家可能被 batch 中其他序列的 token 占满了容量。

#### 3.3 无辅助损失动态均衡（DeepSeek-V3）

Auxiliary Loss 存在一个副作用：它直接修改路由器的梯度，可能干扰主训练目标。

DeepSeek-V3 提出了替代方案：给每个专家加**可学习的 bias**，基于历史负载统计动态调整路由分数，无需 aux-loss。

```
路由得分调整:

  adjusted_logits = x · W_g + bias

  bias 初始化为 0
  每步训练后统计各专家的 token 数
  → 负载高的专家：bias 减小 → 降低被选概率
  → 负载低的专家：bias 增大 → 提高被选概率
```

```
三种策略的对比:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

                    Auxiliary Loss    容量控制      动态 Bias
                    ━━━━━━━━━━━━━    ━━━━━━━━━━━   ━━━━━━━━━━
作用层面:           训练（梯度）      推理（硬限制）  训练（梯度）
对主目标的干扰:      有（修改梯度）     无             无
能否完全消除不均衡:  否（只能减轻）     是（强制丢弃）  接近完全均衡
实现复杂度:         低（加一个 loss）  中（需管理丢弃） 中（需维护 bias）
典型模型:           Switch            GShard         DeepSeek-V3
```

> 🌐 **补充（Web Search）**：DeepSeek-V3（2024 年底发布，671B 总参数，37B 激活参数）采用了**无辅助损失的负载均衡策略**（Auxiliary Loss-Free Load Balancing），通过对每个专家加动态 bias 来平衡负载，避免了传统 aux-loss 对模型性能的负面影响。此外还使用 FP8 量化降低通信开销。来源：[DeepSeek-V3 技术报告](https://arxiv.org/pdf/2412.19437)

#### 3.4 共享专家 + 路由专家

另一种从架构层面缓解负载均衡问题的设计（DeepSeekMoE 提出）：

```
传统 MoE:                          共享专家 MoE（并行架构）:
━━━━━━━━━━━━━━━                    ━━━━━━━━━━━━━━━━━━━━━━━━

  x → 路由器 → E₀~Eₙ (全路由)       x ──┬── 共享专家（必经，无路由）──→ s
                                         │
                                         └── 路由器 → E₀~Eₙ (路由专家)──→ r
                                                                            ↓
                                                                   output = s + r

  共享专家和路由专家的输入都是同一个 x（并行计算，非串行）
  → 共享专家学习通用知识，路由专家只需学习"增量"
  → 即使路由不完美，共享专家兜底 → 语义不丢失
```

### 四、哈希路由（非学习式路由）

一个值得深思的发现：**复杂的可学习路由器并非绝对必要**。

哈希路由通过固定哈希函数将输入映射到专家，天然具备负载均衡，无需训练路由器。

#### LSH（局部敏感哈希）路由

```
LSH 路由原理:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

x (d维嵌入)
    │
    ↓ 随机投影: projections = x @ A^T    A: [n_hashes, d] 随机矩阵
    │
    ↓ 符号量化: signs = (projections > 0).long()   → 0/1 比特串
    │
    ↓ 编码: hash = signs · [2⁰, 2¹, 2², ...]      → 整数哈希值
    │
    ↓ 取模: expert_id = hash % num_experts           → 专家索引

特点:
  ✅ 不需要训练（A 是固定的随机矩阵）
  ✅ 概率性负载均衡（哈希函数的均匀分布性）
  ✅ 弱语义保持（相似 token 更可能落入同一桶 — LSH 的局部敏感性）
  ⚠️ 语义灵活性弱于可学习路由
```

LSH 公式（随机投影 + 标量量化版本）：

$$h_i(x) = \left\lfloor \frac{a_i^\top x + b_i}{\epsilon} \right\rfloor$$

其中 $a_i$ 是随机投影方向，$b_i$ 是随机偏置，$\epsilon$ 是桶宽度（控制每个桶的 token 容量）。

**核心洞察**：哈希路由能取得相当竞争力，说明 MoE 的架构优势可能**在很大程度上源于稀疏激活 + 参数容量扩张本身**，而非路由器的精妙设计。

![哈希路由](<../images/5.3.png>)
*图5.3 哈希路由*

### 五、路由机制总览对比

| 路由方式 | 是否可学习 | 负载均衡 | 语义能力 | 通信开销 | 典型用途 |
|---------|:---------:|:--------:|:--------:|:--------:|---------|
| **Top-K TC** | 是 | 差（需辅助策略） | 强 | 较高 | 主流生产模型 |
| **Top-K EC** | 是 | 好（天然均衡） | 有丢失 | 中等 | 研究实验 |
| **哈希路由** | 否 | 好（天然均衡） | 弱 | 低 | 大规模推理、轻量实验 |

**可学习路由 vs 哈希路由的实证对比**（来源：DeepSeek MoE 论文 / CS336 课件）：

```
模型配置: 0.2B 激活参数, FLOPs 相同
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Dense 0.2B:        Pile Loss = 2.060  ← 基线
Hash Layer 2.0B:   Pile Loss = 1.9xx  ← 哈希路由，参数多但计算量不变
Switch 2.0B:       Pile Loss = 1.881  ← 可学习路由，进一步优化

结论: 稀疏激活+大参数量是 MoE 的核心驱动力（Hash 已大幅领先 Dense），
      但可学习路由能在此基础上进一步提升（Switch 优于 Hash）。
```

> 💡 **补充（Context7 / PyTorch）**：PyTorch 在分布式通信方面提供了针对 MoE 的优化原语，如 `torch.ops.symm_mem.all_to_all_vdev_2d`，用于 MoE 模型中 token 在专家间的分发和收集，支持非连续的偏移和填充，从 rank-major 重排为 expert-major 顺序。

---

## 🧠 本模块问题

请在下方回答以下问题后，输入 `提交作业` 提交。

**Q1**：TC 和 EC 两种路由模式分别沿得分矩阵（gate_scores: `[B_total, N]`）的哪个维度执行 Top-K 选择？请写出两者的选择方向，并解释为什么 EC 天然负载均衡而 TC 不行。

**Q2**：在 Mini LLM 的 MoE 层中，`capacity_factor` 参数的作用是什么？当某个专家被分配的 token 数超过容量限制时，超出的 token 会发生什么？这对模型有什么影响？

**Q3**：哈希路由（如 LSH）不需要训练，为何仍能在一些基准上展现出与可学习路由相当的竞争力？这说明 MoE 架构的核心优势来自哪里？

---

<!-- 学习者作答区（请在此处填写你的答案） -->

**A1**：
TC（token choice） 沿着得分矩阵 gate_scores 列向量进行 Top-K 选择，每个 token 选择 top k 个专家；
EC（Expert choice）沿着得分矩阵 gate_scores 的行向量进行 Top-K 选择，每个 expert（专家）选择 top k 个 token。
EC 天然负载均衡是因为 每个 expert 都能选到 k 个 token（可能会有重复）；但是 TC 是每个 token 对应专家，有可能多个 token 选择一个专家，而另外几个 expert 没有 token 选择。


**A2**：
`capactity_factor` 是为了控制 expert 的负载均衡，保证某个 expert 不要有过多的 计算。
当某个专家被分配的 token 数量超过 限制时，剩余的 token 不会进入 expert 计算（被丢弃），直接进入下一层。
这会使得该部分 token 特征没有被有效提取，进入下一层 attention 时没法获得有效的注意力。


**A3**：
哈希路由不需要训练，其按照一定随机性将 token 均匀得分配给每个 expert，说明 MOE 的核心不是来自多个专家之间的知识聚焦，而是通过类似 dropout 的稀疏激活，来使得模型有限可表达得参数更鲁棒，并且总体参数量会上升，代来了更多表达容量的可能性。
这也就是 MOE 的核心优势。

---

<!-- 教师批改区（提交作业后由导师填写，请勿手动修改） -->

### 📝 批改结果

**Q1 批改**：

✅ TC "每个 token 选择 top k 个专家" — 方向正确，TC 沿专家维度（列）为每个 token 选 Top-K
✅ EC "每个 expert 选择 top k 个 token" — 方向正确，EC 沿 token 维度（行）为每个专家选 Top-K
✅ EC 天然均衡的原因正确：每个专家固定选 k 个 token
✅ TC 不均衡的原因正确：多个 token 可能选同一个专家，其他专家闲置
⚠️ "沿着列向量/行向量"表述稍模糊——更精确的说法是：TC 对**每行**在**列维度**上取 Top-K，EC 对**每列**（转置后的每行）在**行维度**上取 Top-K

— 得分：**7/10**

**Q2 批改**：

✅ capacity_factor 控制专家负载 — 正确
✅ 超出容量的 token 被丢弃、不进入专家计算 — 正确
✅ 指出 token 直接进入下一层 — 正确
⚠️ "进入下一层 attention 时没法获得有效的注意力" — 表述不够精确。被丢弃的 token 仍通过**残差连接**保留原始隐藏状态，下一层的 attention 仍然会计算它，只是该 token 的表征**未经专家非线性变换**，语义丰富度不如正常处理的 token。不是"没法获得注意力"，而是"输入给注意力的特征较弱"

— 得分：**7/10**

**Q3 批改**：

✅ 哈希路由均匀分配 token — 正确
✅ 核心优势来自参数量扩展而非路由精妙 — 方向正确
⚠️ "类似 dropout 的稀疏激活" — 这个类比不太准确。Dropout 是**随机丢弃神经元**做正则化（防过拟合），MoE 稀疏激活是**按需选择专家**做效率优化（保持计算量不变）。两者虽然都涉及"稀疏"，但目的完全不同
👍 抓住了关键洞察：参数容量扩张是 MoE 的根本驱动力。DeepSeek 实验数据也验证了这一点——Hash 路由（非学习式）已经大幅领先同等 FLOPs 的 Dense 模型

— 得分：**7/10**

**综合评价**：21/30。三个问题都抓住了核心要点，说明对 MoE 路由机制的理解到位。主要改进方向：
1. **术语精确性**：矩阵维度描述用"对每行在列维度上取 Top-K"比"沿列向量"更清晰
2. **残差机制**：被丢弃的 token 不是"消失"了，而是只走了残差路径（保留原始表征），与"完全不存在"有本质区别
3. **类比准确性**：Dropout vs MoE 稀疏激活的目的不同（正则化 vs 效率优化），类比的"相似面"和"不同面"都要看清

**批改时间**：2026-04-23
