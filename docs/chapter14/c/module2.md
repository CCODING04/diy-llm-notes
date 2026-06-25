# 第 14 章：可验证奖励的强化学习 — 模块 2：GRPO 训练循环、长度偏差与 Dr.GRPO 修正

> 📍 学习进度：第 14 章，第 2 / 4 模块
> 📅 生成时间：2026-06-17

---

## 学习目标

- 理解 GRPO 的完整训练循环：从采样到优势计算到参数更新
- 掌握 GRPO 长度偏差的数学根源——`1/|o_i|` 和 `1/std` 如何引入系统性偏差
- 理解 Dr.GRPO 的无偏修正方案及其在 verl 框架中的配置

---

## 核心内容

### 1. GRPO 训练循环全景

模块 1 我们分析了 `compute_pg_loss`——GRPO 的损失计算。但它只是整个训练循环的一环。完整的 GRPO 训练流程是：

```
For each iteration:
  ① 采样: 对每个 prompt q，从 π_old 采样 G 个回答
  ② 打分: 用可验证奖励函数 R(q, o_i) 计算每个回答的奖励
  ③ 归一化: 组内 z-score → 标量优势 A_i
  ④ 扩展: 将标量 A_i 复制到每个 token 位置
  ⑤ 前向+损失: compute_pg_loss 计算 policy_loss + kl_penalty
  ⑥ 更新: 反向传播，更新 π_θ
```

步骤 ①~④ 对应的是优势构建阶段，步骤 ⑤~⑥ 对应的是参数更新阶段。下面我们深入步骤 ②~④ 的代码实现。

#### 1.1 奖励计算与 z-score 归一化（步骤 ②~③）

来自 [nano-aha-moment](https://github.com/McGill-NLP/nano-aha-moment/blob/main/nano_r1_script.py) 的优势构建代码：

```python
# 1. 数据校验与分组
assert len(all_generations) == len(all_finish_reasons)
assert len(all_generations) == len(samples) * GENERATIONS_PER_SAMPLE
# GENERATIONS_PER_SAMPLE = G，每个 prompt 采样 G 个回答
# 总回复数 = 样本数 × G

groups = [
    list(range(i, i + GENERATIONS_PER_SAMPLE))
    for i in range(0, len(all_generations), GENERATIONS_PER_SAMPLE)
]
# groups = [[0,1,2], [3,4,5], ...]  当 G=3 时

# 2. 初始化存储
all_query_token_ids, all_responses_token_ids, all_advantages = [], [], []
all_rewards = []
stats = {"response_lengths": [], "rewards": [], "non_stop_rate": []}

# 3. 核心循环：逐组处理
for sample, group_indices in zip(samples, groups):
    finish_reasons = [all_finish_reasons[i] for i in group_indices]
    response_token_ids = [all_generations[i] for i in group_indices]
    responses = tokenizer.batch_decode(response_token_ids, skip_special_tokens=False)

    # 对组内每个回答调用 compute_reward（用户自定义——如数学判分、代码测试）
    rewards_and_metrics = [compute_reward(resp, sample, EOS_TOKEN) for resp in responses]
    rewards, reward_metrics = zip(*rewards_and_metrics)

    # 4. 关键步骤：z-score 归一化
    rewards = np.array(rewards)
    advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-4)
    # A_i = (r_i - μ) / (σ + 1e-4)

    # 5. 将标量优势扩展到每个 token
    per_token_advantages = [
        [adv] * len(resp) for adv, resp in zip(advantages, response_token_ids)
    ]
    # 例如: adv=+1.414, 回答长度=50 → [1.414, 1.414, ..., 1.414] (50个)

    # 6. 收集结果
    all_query_token_ids.extend([sample["input_ids"]] * GENERATIONS_PER_SAMPLE)
    all_responses_token_ids.extend(response_token_ids)
    all_advantages.extend(per_token_advantages)
    stats["rewards"].extend(rewards)
    stats["non_stop_rate"].extend([fr != "stop" for fr in finish_reasons])
    stats["response_lengths"].extend([len(ids) for ids in response_token_ids])

# 返回 episodes 字典，供 compute_pg_loss 使用
episodes = {
    "all_query_token_ids": all_query_token_ids,
    "all_response_token_ids": all_responses_token_ids,
    "all_advantages": all_advantages,
}
```

> 此代码片段展示了 GRPO 训练循环的数据准备阶段。`compute_reward` 是用户根据任务自定义的函数。该片段不是可直接独立运行的完整训练脚本，依赖外部的 `samples`、`GENERATIONS_PER_SAMPLE` 等变量。

##### 案例：nano-aha-moment 的 `compute_reward` 实现

nano-aha-moment 的任务是**算数游戏**：给定若干数字和一个目标值，模型需要用四则运算得出目标值（类似 24 点）。它的 `compute_reward` 由两个子奖励组合而成：

```python
def format_reward_func(completion: str, EOS_TOKEN: str) -> float:
    """检查输出是否符合 <think>...</think>\n<answer>...</answer> 格式"""
    allowed_pattern = r"^[\d+\-*/().\s]+$"  # 答案区只允许数学字符

    completion = "<think>" + completion  # 补全前缀方便正则匹配
    if completion.endswith(EOS_TOKEN):
        completion = completion[: -len(EOS_TOKEN)]

    regex = r"^<think>([^<]*(?:<(?!/?think>)[^<]*)*)<\/think>\n<answer>([\s\S]*?)<\/answer>$"
    match = re.search(regex, completion, re.DOTALL)

    if match is None:
        return 0.0                    # 格式完全错误
    else:
        answer_content = match.group(2).strip()
        if not re.match(allowed_pattern, answer_content):
            return 0.5                # 格式对，但答案含非法字符
        else:
            return 1.0                # 格式完全正确


def equation_reward_func(completion: str, nums: List[int], target: int) -> float:
    """检查答案表达式是否数学上正确"""
    match = re.search(r"<answer>(.*?)<\/answer>", "<think>" + completion)
    if match is None:
        return 0.0

    equation = match.group(1).strip()
    used_numbers = [int(n) for n in re.findall(r"\d+", equation)]

    if sorted(used_numbers) != sorted(nums):   # 必须使用给定数字各一次
        return 0.0
    if not re.match(r"^[\d+\-*/().\s]+$", equation):
        return 0.0                             # 防代码注入

    # 安全 eval: 禁用所有内置函数，只允许纯数学表达式
    result = eval(equation, {"__builtins__": None}, {})
    if abs(float(result) - float(target)) < 1e-5:
        return 1.0
    return 0.0


def compute_reward(completion, sample, EOS_TOKEN):
    """总奖励 = 格式奖励 + 等式奖励，满分 2.0"""
    format_reward = format_reward_func(completion, EOS_TOKEN)
    equation_reward = equation_reward_func(completion, sample["nums"], sample["target"])
    reward = format_reward + equation_reward
    return reward, {"format_reward": format_reward, "equation_reward": equation_reward}
```

三层奖励信号的含义：

| 场景 | format | equation | total | 含义 |
|------|--------|----------|-------|------|
| 没按格式输出 | 0 | 0 | **0** | 连基本格式都没学会 |
| 格式对，但答案含非法字符 | 0.5 | 0 | **0.5** | 学会了格式骨架，数学还不会 |
| 格式对，表达式不成立 | 1 | 0 | **1** | 格式 OK，推理能力尚不足 |
| 完全正确 | 1 | 1 | **2** | 完美 |

**设计意图**：格式奖励是**稠密信号**——模型很快学会 `<think>/<answer>` 结构；等式奖励是**稀疏信号**——只有数学严格正确才给分。两者组合让模型先学会"怎么说"，再学会"说什么对"。这正是 DeepSeek R1 的 **accuracy + format 双重奖励**的微型翻版。

##### 不同任务对应不同的 `compute_reward`

`compute_reward` 是 GRPO 与具体任务之间的**适配层**——算法框架不变，但打分逻辑随任务变化：

| 任务类型 | 验证方式 | `compute_reward` 的核心逻辑 | 典型奖励范围 |
|---------|---------|--------------------------|------------|
| 数学推理（GSM8K, MATH） | 字符串匹配/表达式等价 | 提取 `\boxed{...}` 与标准答案比对 | 0/1 |
| 代码生成（HumanEval, LiveCodeBench） | 执行测试用例 | 沙箱运行代码，检查通过率 | 0 ~ 1（通过比例） |
| 算数游戏（nano-aha-moment） | eval 表达式 | 格式检查 + 等式验证 | 0 ~ 2 |
| 形式化证明（MiniF2F） | 证明助手验证 | Lean/Coq 编译通过 | 0/1 |
| 通用对话（RLHF 变体） | 奖励模型打分 | Reward Model 前向传播 | 连续值 |

关键原则：
- 奖励函数必须是**确定性**的（同一输入同一输出），这是 RLVR 的前提
- 如果同时有多个奖励维度（格式 + 正确性），**拆成子奖励分别记录**——便于监控训练中哪个维度在改善、哪个在退化
- `compute_reward` 的复杂度直接影响训练吞吐——数学字符串匹配是 O(1) 级别，代码沙箱执行可能需要秒级

#### 1.2 代码与工业实现的差异

| 代码中做了什么 | 简化了什么 | 工业部署怎么做 |
|--------------|-----------|--------------|
| 单进程 `for` 循环逐组处理 | 省略了分布式采样 | 多 GPU 并行 rollout，每张卡独立采样 |
| `np.array(rewards)` | 使用 numpy 做简单统计 | 大规模部署用 torch 分布式 all-reduce 收集全局统计量 |
| `rewards.std() + 1e-4` | 固定数值稳定常数 | 可能用动态 epsilon 或分层归一化 |
| `[adv] * len(resp)` 创建 Python list | 直接在 Python 中复制 | 用 `torch.repeat` 或广播在 GPU 上复制 |
| `compute_reward` 是同步调用 | 假设打分很快 | 代码执行需沙箱、超时控制、并行评测 |

#### 1.3 GRPO 的端到端数据流

```
Prompt q
   │
   ├─→ π_old 采样 → o_1 (50 tokens) ─→ R(q,o_1) = 1 ─┐
   ├─→ π_old 采样 → o_2 (30 tokens) ─→ R(q,o_2) = 0 ─┤
   ├─→ π_old 采样 → o_3 (45 tokens) ─→ R(q,o_3) = 1 ─┤
   └─→ π_old 采样 → o_4 (35 tokens) ─→ R(q,o_4) = 0 ─┘
                                                      │
                                              r = [1, 0, 1, 0]
                                              μ = 0.5, σ = 0.5
                                                      │
                              A = [+1.0, -1.0, +1.0, -1.0]  (z-score)
                                                      │
                    ┌─────────────────────────────────┘
                    │  扩展到每个 token:
                    │  o_1: [1.0×50]   o_2: [-1.0×30]
                    │  o_3: [1.0×45]   o_4: [-1.0×35]
                    │
                    ▼
            compute_pg_loss(policy_model, batch, ...)
                    │
                    ├─→ logps = compute_token_log_probs(...)  [B, seq-1]
                    ├─→ kl_penalty = exp(ref_logratio) - 1 - ref_logratio
                    ├─→ policy_loss = -logps * advantages
                    ├─→ loss = (policy_loss + β·kl_penalty).sum() / total_response_len
                    │
                    ▼
              反向传播 → 更新 π_θ
```

---

### 2. GRPO 实验结果：RFT vs GRPO+OS vs GRPO+PS

在进入长度偏差分析之前，先看 GRPO 在数学推理基准上的实际表现（来自 DeepSeekMath 论文，arxiv 2402.03300）：

![GRPO与其他训练方法在两个数学推理基准测试上的模型性能对比](<../images/14-5-grpo与其他训练方法在两个数学推理基准测试上的模型性能对比.png>)

- **GSM8K**（小学数学应用题）和 **MATH**（高中数学竞赛题）
- 四条曲线代表不同的训练方法：

| 方法 | 全称 | 核心机制 |
|------|------|---------|
| RFT | Reinforcing Fine-Tuning | 只奖励最终正确答案，在线采样后一次性 SFT |
| Online RFT | Online RFT | RFT 的在线版本，训练过程中持续采样更新 |
| **GRPO+OS** | GRPO + Online Sampling | 标准 GRPO，组内 z-score + 在线采样 |
| **GRPO+PS** | GRPO + Process Supervision | GRPO + 过程监督——不仅奖励最终答案，也奖励正确解题步骤 |

**关键结论**：
- GRPO 显著优于 RFT（橙色/蓝色线远超紫色线）——组内归一化确实有效
- 过程监督（PS）带来额外增益——蓝色线略高于橙色线，但差距不大（说明 GRPO 本身已足够好）
- GRPO 曲线比 RFT 更平滑——反映了算法稳定性

---

### 3. GRPO 的长度偏差：`1/|o_i|` 的数学陷阱

#### 3.1 偏差从哪里来？

GRPO 的原始目标函数（DeepSeekMath 论文）中，每个回答 `o_i` 的损失在 token 维度上除以了响应长度 `|o_i|`：

$$J_{GRPO}(\theta) = \mathbb{E}\left[\frac{1}{G}\sum_{i=1}^{G} \frac{1}{|o_i|}\sum_{t=1}^{|o_i|} \left(\min(r_{i,t} \hat{A}_i, \text{clip}(r_{i,t}, 1-\epsilon, 1+\epsilon) \hat{A}_i) - \beta D_{KL}\right)\right]$$

这个看起来无害的 `1/|o_i|`，在对 token 级梯度求和后产生了系统性的**长度相关偏差**。

#### 3.2 正优势情况（正确答案）

当 `A_i > 0`（回答正确）：
- 梯度 ∝ `A_i / |o_i|`
- 如果 **|o_i| 较小** → `A_i/|o_i|` 更大 → **更大的梯度更新**
- 模型学到：**正确回答越短越好**

```
两个都正确的回答:
  o_A: 30 tokens, A=+1.0 → 每个 token 梯度 ≈ 1.0/30 = 0.033
  o_B: 60 tokens, A=+1.0 → 每个 token 梯度 ≈ 1.0/60 = 0.017

→ o_A 的梯度是 o_B 的 2 倍
→ 模型被更强地鼓励生成 30-token 的回答
```

这本身不是坏事——简洁的正确回答确实更可取。但问题出在负优势情况。

#### 3.3 负优势情况（错误答案）—— "越错越长"

当 `A_i < 0`（回答错误）：
- 梯度 ∝ `A_i / |o_i|`（负值，表示惩罚）
- 如果 **|o_i| 较大** → `A_i/|o_i|` 的绝对值**更小** → **惩罚被稀释**

```
两个都错误的回答:
  o_A: 20 tokens, A=-1.0 → 每个 token 梯度 ≈ -1.0/20 = -0.050 (强惩罚)
  o_B: 100 tokens, A=-1.0 → 每个 token 梯度 ≈ -1.0/100 = -0.010 (弱惩罚)

→ o_B 的惩罚只有 o_A 的 1/5
→ 模型学到：如果答错了，说得越长惩罚越轻！
```

这就是 **"越错越长"（longer-when-wrong）现象**的数学根源。模型在训练中逐渐学会：宁可生成一个冗长的错误回答，也不要生成一个简短的错误回答——因为前者受到的惩罚更小。

#### 3.4 `1/std` 的问题难度偏差

GRPO 的另一个归一化因子是标准差 `std({r_1, ..., r_G})`：

$$A_i = \frac{r_i - \text{mean}}{\text{std}}$$

这个 `std` 引入了**问题难度相关的偏差**：

**数值推演**：
```
问题 A（难，模型几乎都错）:
  G=4, r = [0, 0, 0, 1]  (只有1个碰巧对)
  mean = 0.25, std = 0.433
  A = [-0.577, -0.577, -0.577, +1.732]
  → |A_max| = 1.732

问题 B（易，模型几乎都对）:
  G=4, r = [1, 1, 1, 0]  (只有1个粗心错)
  mean = 0.75, std = 0.433
  A = [+0.577, +0.577, +0.577, -1.732]
  → |A_max| = 1.732
```

两种情况 std 相同（因为是线性平移），但当奖励不是纯 0/1 而是有连续值时（如代码通过部分测试用例），std 会因问题类型而异。过于简单的问题（所有回答都得分高，std 小）会被人为放大优势，过于难的问题（所有回答都得分低，std 也小）同样被放大——导致模型把过多的更新量花在"极端难度"的问题上，而非"刚好在能力边界"的问题上。

> 🌐 **补充（Web Search / Dr.GRPO 论文）**：Dr.GRPO 论文（"Understanding R1-Zero-Like Training: A Critical Perspective", arxiv 2503.20783）指出，GRPO 的 STD 归一化导致"too-easy or too-hard problems are upweighted"——这两类问题的区分度最低，不应获得最大的更新权重。理想情况下，模型应该把最多的学习信号分配给**中等难度**的问题（回答有对有错，区分度最高）。

---

### 4. Dr. GRPO：无偏修正

#### 4.1 Dr. GRPO 做了什么？

[Dr. GRPO](https://arxiv.org/abs/2503.20783)（Sea AI Lab, 2025）的核心改动很简单——从 GRPO 目标中**移除两个归一化项**：

| 归一化项 | GRPO 原始 | Dr. GRPO | 论文命名的偏差 | 移除原因 |
|---------|----------|----------|-------------|---------|
| 长度归一化 `1/len(o_i)` | ✅ 使用 | ❌ 移除 | **Response-level length bias** | 短回答每 token 梯度大、长回答每 token 梯度小。正确短回答被强力强化（好），但错误长回答的惩罚被稀释，导致模型"越错越长" |
| 标准差归一化 `1/std` | ✅ 使用 | ❌ 移除 | **Question-level difficulty bias** | 太易/太难的问题 std 接近 0 → 优势被放大 → 获得不成比例的高更新权重。模型在"最没有学习价值"的问题上浪费最多算力 |

Dr. GRPO 的无偏梯度形式：

$$\nabla_\theta J_{Dr.GRPO} = \mathbb{E}\left[\frac{1}{G}\sum_{i=1}^{G} \sum_{t=1}^{|o_i|} \nabla_\theta \log \pi_\theta(o_{i,t}|q, o_{i,<t}) \cdot (r_i - \bar{r}) - \beta \nabla_\theta D_{KL}\right]$$

与原始 GRPO 的关键区别：
- 优势：`(r_i - mean(r))` 代替 `(r_i - mean(r)) / std(r)`
- Token 聚合：每个 token 的损失**不除以** `|o_i|`，而是累计后统一归一化

#### 4.2 数值对比

```
GRPO 原始:  loss_i = (1/|o_i|) × Σ_t policy_loss_t
                       ↑ 逐回答长度归一化 → 偏差来源！

Dr. GRPO:   loss_i = Σ_t policy_loss_t     （不除以 |o_i|）
            total_loss = (1/G) × Σ_i loss_i
                       ↑ 对 G 个回答取平均，不再按回答长度"打折"每个 token
```

核心变化：`1/|o_i|`（逐回答长度归一化）被移除。回答 o_i 无论长 30 token 还是 90 token，其内部每个 token 的梯度贡献不再被自身长度稀释。外层 `1/G` 是标准的 MC 平均——对 G 条采样轨迹取均值，是策略梯度估计量的标准形式。

> 📝 **实现备注（verl 框架）**：verl 的实际实现 `loss_agg_mode="seq-mean-token-sum-norm"` 使用 **`1/Σ|o_i|`**（总 token 数归一化）代替 `1/G`：
> ```
> total_loss = Σ_i loss_i / (Σ_i |o_i|)   ← verl 的实际实现
> ```
> 两种归一化的共同点：都**不包含** `1/|o_i|`（逐回答长度归一化），因此都消除了 Dr. GRPO 指出的偏差。区别仅在外层缩放：
> - `1/G`（论文形式，4.1 公式）：每个**回答**等权重贡献 → 回答级公平
> - `1/Σ|o_i|`（verl 形式）：每个 **token** 等权重贡献 → token 级公平，跨 batch 数值更稳定

#### 4.3 verl 框架中的 Dr. GRPO 配置

[verl](https://github.com/verl-project/verl)（Volcano Engine Reinforcement Learning）是字节跳动开源的 LLM 强化学习训练框架，支持 PPO、GRPO、Dr. GRPO、DAPO 等多种 RL 算法。它提供了统一的 `AlgoConfig` 接口来配置算法超参数，是当前工业界部署 GRPO 训练最主流的选择之一。

以下是啟用 Dr. GRPO 的具体配置：

```yaml
# Dr. GRPO 的关键配置项
actor_rollout_ref.actor.loss_agg_mode: "seq-mean-token-sum-norm"
  # "seq-mean-token-sum-norm": 先按序列平均，再按 token 求和，最后用总 token 数归一化
  # 等价于移除 |o_i| 的偏差

algorithm.norm_adv_by_std_in_grpo: False
  # 关键！设为 False 以移除 std 归一化

actor_rollout_ref.actor.use_kl_loss: False
  # Dr. GRPO 不使用独立的 KL loss 项
  # 原因：移除 1/|o_i| 和 1/std 偏差后，梯度本身干净稳定，PPO clip 已
  # 足够约束更新幅度。再加 KL loss 反而与 clip 冗余掣肘。公式中 β·D_KL
  # 保留是为了数学完备性，实际 β 通常设为 0。
```

> 💡 **补充（Context7 / verl）**：`loss_agg_mode: "seq-mean-token-sum-norm"` 是 Dr. GRPO 的核心配置。verl 还支持可选的 `loss_scale_factor` 参数（设为常量如最大响应长度），确保不同 batch 间的归一化一致。

#### 4.4 Dr. GRPO 的效果

![Dr.GRPO与标准的GRPO的数学公式与性能对比](<../images/14-6-Dr-GRPO与标准的GRPO的数学公式与性能对比.png>)

左图：Dr. GRPO 的公式（移除了长度和标准差归一化）
右图：训练过程中奖励（Reward）与输出长度（Output length）的关系

**关键发现**：
- 原始 GRPO：随着训练进行，输出长度持续增长（尤其错误回答）
- Dr. GRPO：有效抑制了长度膨胀，token 效率显著提升
- 性能（奖励）不受影响甚至略好——说明被移除的那些归一化项本质上是有害的

---

### 5. 常见误解澄清

**误解 1**："GRPO 和 Dr. GRPO 是完全不同的算法"

✅ 正确理解：Dr. GRPO **不是**一个全新的算法——它只是 GRPO 的**无偏版本**。核心结构（组内采样、z-score 优势、PPO clip、KL 惩罚）完全相同，区别仅在于去掉了两个归一化项。可以理解为 GRPO v1 → GRPO v2。

**误解 2**："去掉 `1/|o_i|` 后模型会倾向于生成更长的回答"

✅ 正确理解：恰恰相反。原始 GRPO 的 `1/|o_i|` 对错误长回答有"免死金牌"效应（惩罚被稀释），去掉后模型反而不敢随便写长错误回答。Dr. GRPO 的 token 效率更高，不是因为强制输出变短，而是因为**不再对长错误回答给予隐性奖励**。

**误解 3**："std 归一化只是数值稳定性技巧，去掉与否无所谓"

✅ 正确理解：std 归一化改变了不同难度问题的**相对权重**。去掉它不仅不影响数值稳定性（有 1e-4 保护），还能让模型把更多学习信号分配给"能力边界"上的问题——恰好是学习效率最高的区域。

---

## 🧠 本模块问题

**Q1**：GRPO 的 `1/|o_i|` 归一化对正确回答鼓励简洁、对错误回答稀释惩罚——这构成了一个不对称的偏差。请分析：如果训练数据中错误回答的比例远高于正确答案（初始模型能力弱），这个偏差会导致什么恶性循环？

**Q2**：Dr. GRPO 移除了 `1/std` 归一化。假设我们有三个问题——问题 A（std=0.1，很简单的判断题）、问题 B（std=0.3，中等难度的计算题）、问题 C（std=0.2，很难的证明题但模型输出格式高度一致导致方差小）。在原始 GRPO 下，哪个问题会获得最大的更新权重？Dr. GRPO 移除 std 后，这种权重分配发生了什么变化？是好是坏？

**Q3**：nano-aha-moment 的代码中用 `[adv] * len(resp)` 将标量优势扩展到每个 token。结合模块 1 中讨论的"同享优势"问题，试分析：如果要在代码层面做一个最小修改来区分"推理的关键步骤 token"和"填充/格式 token"的优势值，你会怎么改？为什么这个修改在工程实践中其实很困难？

---

<!-- 学习者作答区（请在此处填写你的答案） -->

**A1**：
GRPO 的 `1/|o_i|` 归一化，对错误答案，会让模型通过生成更多的 token，来平均（压低）损失。
而如果初始模型能力弱，回答错误比例远高于正确答案，那么模型通过 GRPO 采样得到的数据大多数错误数据，而我们优化的目标会变成让模型生成更多的无用的 token。
模型生成越来越长的错误答案只会让 模型能力变得更差，从而恶性循环。



**A2**：

在原始 GRPO 下，对于 std=0.1（很简单的问题） 以及 std=0.2（很难得证明题但模型输出格式高度一致） 更容易获得最大的权重更新。因为这个时候经过 1/std 归一化，权重 scale 到较大的尺度。

Dr.GRPO 移除 std 后，没有了 1/std 的 scale，相当于直接通过 r - r_mean 来计算 advantage。权重更依赖回答本身获得的分数多少以及这个问题 难易（均值）来做判断，而不再通过 1/std 进行二次 scale，不会收到 难题/容易题目 更容易获得奖励的偏见，是好事。


**A3**：

我们需要一个信号告诉我们
什么是 "推理的关键步骤 token"
而什么又是  "填充/格式 token"

填充/格式，很容易理解，可以通过正则或者规格化标签来满足要求，但是 "推理的关键步骤 token" 需要 得到 token 和 正确答案的 attention，这个 attention 足够高，那么它就是关键步骤 token。

如果是我的话，我会将得到正确答案的 最后一次token输出的最后一层 attention 拿来做 关联性分析，找到跟 "答案" 部分 sofmax(QK_t) 最高的 index 来作为 关键 token

这个修改困难在于需要缓存最后一层的 全局注意力，这是一个非常大的量级，并且还需要通过 argmax 找到跟答案相关的 token index。


---

<!-- 教师批改区（提交作业后由导师填写，请勿手动修改） -->

### 📝 批改结果

**Q1 批改**：方向正确——准确识别了"错误回答通过生成更多 token 稀释惩罚 → 模型变本加厉生成冗长错误 → 进一步削弱学习信号"这条恶性链路。但缺少两个层次：(1) 梯度层面的自强化机制——`∂L/∂θ ∝ 1/|o_i|`，长度每翻倍惩罚就减半，这是指数级的稀释；(2) 定量视角——若初始模型 80% 回答错误，意味着 80% 的训练步在强化"冗长但错误"的策略。另外，KL 惩罚在长回答上的累积效应也应提及。——得分：7/10

<details>
<summary>📖 Q1 参考答案</summary>

**"越错越长"的恶性循环——完整链路**：

**第一环：梯度稀释（微观机制）**

```
模型能力弱 → 大部分回答错误 → 对每个 prompt 的 G 个回答:
  假设 G=4, 3 个错误, 1 个正确

  错误回答: o_wrong, |o_wrong|=100 tokens, A_i=-1.0
    每个 token 梯度 ∝ -1.0/100 = -0.01  ← 几乎无惩罚
  正确回答: o_right, |o_right|=30 tokens, A_i=+1.0
    每个 token 梯度 ∝ +1.0/30 = +0.033   ← 信号强度是惩罚的 3.3 倍

  → 单步净效果: 鼓励正确短回答, 几乎不惩罚错误长回答
```

**第二环：策略退化（宏观效应）**

```
训练步 1-100:  错误率 80%, 平均错误长度 60 tokens
训练步 100-200: 错误率 75% (略有改善), 但错误长度增长到 90 tokens
                 ↑ 改善是假象——模型只是学会了"安全地犯错"
训练步 200-300: 错误率 72%, 错误长度增长到 130 tokens
                 ↑ 惩罚被进一步稀释, 改善速度越来越慢
```

**第三环：恶性自强化**

```
模型学到: "如果没把握做对 → 说得越长越好"
  → 生成越来越冗长的推理过程
  → 更多 token 被训练成"冗长但错误"的模式
  → 1/|o_i| 进一步稀释惩罚
  → 模型更没有动力改进推理质量
  → 回到第一步
```

**综合效应**：

```
初始状态: 模型 20% 正确, 平均长度 40 tokens
500步后:  模型 30% 正确, 平均长度 150 tokens  ← "伪进步"
         → 正确率提升 10%, 但长度膨胀 275%
         → token 效率下降 73% (每个正确回答消耗的 token 数翻倍)

训练曲线上看:
  reward ↗ 缓慢上升 (虚假的进步)
  length ↗↗↗ 持续膨胀 (真实的退化)
  token-efficiency ↘↘↘ (被长度稀释的实际学习效率)
```

**为什么 RLHF 中这个问题不那么严重？**
- RLHF 的 reward model 会给冗长回答低分（人类标注者不喜欢啰嗦）
- RLVR 的规则验证器只看最终答案对错，对"过程有多啰嗦"无感知
- 因此 RLVR 场景下 `1/|o_i|` 的长度偏差尤其危险——没有"啰嗦惩罚"信号来制衡

</details>

---

**Q2 批改**：正确判断了 std 越小、`1/std` 放大越厉害的方向，答出了"A 和 C 受影响更大"这一关键结论。移除后的分析方向对（不再被 std 偏见主导），但有两处可深化：(1) 缺少三个问题权重的定量计算——A(std=0.1) 的权重是 B(std=0.3) 的 3 倍，C(std=0.2) 是 B 的 1.5 倍，算出来更清晰；(2) "权重更依赖回答本身获得的分数和均值"表述模糊——移除 std 后优势退化为 `r - r̄`，本质上是**绝对分数差**，而非"依赖难度"。——得分：6.5/10

<details>
<summary>📖 Q2 参考答案</summary>

**原始 GRPO 下的权重分析**：

在原始 GRPO 中，更新强度 ∝ `(r_i - r̄) / std`。即使 `(r_i - r̄)` 相同，`1/std` 这个乘数在不同问题间差异巨大：

```
问题 A (std=0.1, 简单判断题):
  |A_max| ∝ 1/0.1 = 10.0  → 更新权重 ×10

问题 B (std=0.3, 中等计算题):
  |A_max| ∝ 1/0.3 = 3.33  → 更新权重 ×3.33

问题 C (std=0.2, 难证明题):
  |A_max| ∝ 1/0.2 = 5.0   → 更新权重 ×5.0
```

**权重排序**：A (10.0) > C (5.0) > B (3.33)

这说明最简单的题（A）获得了最大的更新权重——恰好是"最没有学习价值"的题（模型已经会了，不需要再学）。

**Dr. GRPO 移除 std 后**：

```
所有问题: |A_max| ∝ |r_i - r̄|  ← 退化为线性差
  A: 判断题, 标准答案固定, r_i ∈ {0, 2}, r̄ ≈ 1.0~2.0
     → |A_max| ≈ 1.0  (不再是 10.0!)
  B: 计算题, 部分对/错, r_i ∈ {0, 1, 2}, r̄ ≈ 0.5~1.5
     → |A_max| ≈ 1.5  (现在 B 的权重最高!)
  C: 证明题, 格式一致但难, r_i ∈ {0, 1}, r̄ ≈ 0.3~0.7
     → |A_max| ≈ 0.7
```

**权重重分配**：B (1.5) > A (1.0) > C (0.7)

中等难度的 B 现在获得最大权重——恰好在模型的能力边界上，学习效率最高。

**为什么这是好事？**

| | 原始 GRPO (有 std) | Dr. GRPO (无 std) |
|---|---|---|
| 权重主导因素 | 问题类型（std 大小） | 回答质量差异（|r-r̄|） |
| 学习信号分配 | 偏简单题和极难题 | 偏中等难度题（能力边界） |
| 极端情况 | std≈0 的问题权重爆炸 | 无此风险 |

核心洞察：**模型进步最快的区域是"刚好在能力边界"的问题（有对有错，区分度最高）。** 原始 GRPO 的 `1/std` 恰好把最大权重分配给了这个区域的反面——区分度最低的极端难度题。

</details>

---

**Q3 批改**：方案有创意——用最终 token 的 attention 来追溯关键推理步骤，确实是一种可行的试探方向。准确指出了内存瓶颈（缓存全局注意力矩阵）。但遗漏了三个更根本的困难：(1) attention 高 ≠ 推理重要——语法虚词、格式 token 可能也有高 attention；(2) 这个方案要求训练时存储生成时的中间激活，与现有 GRPO 的 rollout-then-train 流程冲突；(3) 工业界的标准解法是**过程奖励模型（PRM）**——训练一个独立模型来评判每个推理步骤的质量，而非手工硬编码规则。你的方案本质上是 PRM 的一个轻量级近似。——得分：7/10

<details>
<summary>📖 Q3 参考答案</summary>

**方案层面——最小代码修改**：

```python
# 原始 (所有 token 同享优势):
per_token_advantages = [adv] * len(resp)

# 最小修改 (加权优势):
token_weights = compute_token_importance(resp, answer)  # 某种重要性评分
per_token_advantages = [adv * w for w in token_weights]
```

**你的方案本质**：用 `compute_token_importance` = attention 关联度

**工程实践中的四重困难**：

**困难 1：没有可靠的 token 重要性真值标签**

```
已知: 最终答案对/错 (标量 reward)
未知: 哪个 token 对最终答案贡献了多少

任何启发式规则 (attention, gradient, etc.) 都是猜测:
  - attention 高 ≠ "推理关键": "the", "is" 的 attention 可能也很高
  - gradient 大 ≠ "重要": token 可能在 loss 曲面的陡峭区域碰巧梯度大
  - 位置靠后的 token ≠ "贡献大": 可能只是格式输出
```

**困难 2：与 GRPO rollout 流程的架构冲突**

```
现有流程:
  ① Rollout: 生成 G 个回答 (只存 token ids, 不存中间激活)
  ② Reward: 打分
  ③ Train: 用 token ids + rewards 训练

你的方案需要:
  ① Rollout: 生成 G 个回答 + 保存每层 attention 矩阵
    → 若 model=7B, layers=32, heads=32, seq=1000:
      单次生成存储 = 32×32×1000×1000×2bytes ≈ 2GB/回答
      G=4 → 8GB/问题 → 完全不可行
```

**困难 3：工业界标准解法是 PRM（过程奖励模型）**

```
PRM 方法 (DeepSeekMath 的 Process Supervision):
  训练一个独立模型，对每个推理步骤打分
  Step 1: "思考：需要勾股定理" → PRM score: 0.15
  Step 2: "设 a²+b²=c²"       → PRM score: 0.30
  Step 3: "代入 a=3, b=4"     → PRM score: 0.65
  Step 4: "c=5"               → PRM score: 0.95

然后用 PRM 分数作为 token-level advantage，而非标量复制。

代价: 需要训练和维护一个额外的 PRM 模型 (同规模，显存翻倍)。
      这恰好是 GRPO 设计初衷想要避免的 ("不要额外的网络")。
```

**困难 4：注意力方案的退化风险**

```
如果模型学到: "把 attention 集中在几个 token 上 → 获得更高加权优势"
  → 模型可能操纵 attention pattern 来骗优势，而非真正改进推理
  → 本质上是新的 exploit 路径
```

**总结：为什么工业界还在用"同享优势"？**

不是因为想不到区分 token 重要性的方法，而是因为**所有区分方法的额外成本（PRM 训练/注意力缓存/规则标注）都超过了 GRPO 去掉价值网络所节省的成本**。"同享优势"是精度和成本的 tradeoff，在目前的技术条件下，这个 tradeoff 仍然偏向前者。

</details>

---

**综合评价**：

| 题目 | 得分 | 核心问题 |
|------|------|---------|
| Q1 | 7.0/10 | 方向正确，缺少梯度层面的定量分析和 KL 交互效应 |
| Q2 | 6.5/10 | 结论对，缺少三个问题的定量权重计算 |
| Q3 | 7.0/10 | 方案有创意，指出了内存困难，但遗漏了 PRM 等更深层的工程障碍 |
| **总分** | **20.5/30** | |

**亮点**：Q1 的恶性循环直觉很好，Q3 的 attention 思路有研究价值。

**建议复习**：
1. Q1 — 可以更关注"梯度级"的分析习惯，不只是"发生了什么"，更要追问"每次梯度更新时 ∂L/∂θ 到底多大"
2. Q2 — 养成"先算数字再下结论"的习惯——三个 std 值算完权重，结论一目了然
3. Q3 — 了解 PRM（过程奖励模型）作为 token-level 信用分配的标准解法，可以回顾 DeepSeekMath 论文 4.1.3 节

**批改时间**：2026-06-18
