# 第 9 章：缩放定律（Scaling Laws） — 模块 3：案例研究与实践

> 📍 学习进度：第 9 章，第 3 / 4 模块
> 📅 生成时间：2026-05-19

---

## 学习目标

- 理解扩散模型同样遵循缩放定律的实证结论
- 掌握 Cerebras-GPT 如何首次公开验证 μP 在大模型缩放中的有效性
- 理解 MiniCPM 的 WSD 学习率调度如何降低 Chinchilla 分析成本
- 掌握 DeepSeek 不用 μP、直接拟合缩放定律的务实路线
- 了解 LLaMA 3、Hunyuan、MiniMax 等模型的缩放策略差异
- 能对比不同团队的缩放策略，理解"没有唯一正确方案"

---

## 核心内容

### 一、扩散模型的缩放法则

在此之前，Scaling Laws 主要在**自回归模型**（如 GPT）上被研究。[Likelihood-Based Diffusion Language Models](https://arxiv.org/abs/2305.18619) 验证了扩散模型是否也遵循同样的规律。

![扩散模型缩放法则](<../images/9-30-扩散模型的缩放法则1.png>) ![扩散模型缩放法则2](<../images/9-30扩散模型的缩放法则2.png>)

> **图片讲解**：
> - **左图**：自回归模型的 IsoFLOP 曲线（固定计算预算，扫描不同模型大小，找最优点）
> - **中图**：扩散模型的 IsoFLOP 曲线（同样方法）
> - **右图**：将左图和中图的所有最优点（星标）连起来，在双对数坐标下呈**直线**
>
> **核心结论**：扩散模型也严格遵循幂律缩放。只要增加算力，就能精准预测扩散模型能达到的效果。IsoFLOP 中 "Iso" 意为"相等"——固定总 FLOPs，寻找最优 (N, D) 平衡点。

> 📎 **来源追溯**：Li et al. (2023), *Likelihood-Based Diffusion Language Models* (arXiv:2305.18619)。

> 💡 **补充（Web Search / 扩散模型缩放最新进展）**：
> - **Masked Diffusion Model 缩放律**（NeurIPS 2024, arXiv:2410.18514）：首次建立了 Masked Diffusion Models 的缩放律，缩放速率与自回归模型相当，计算开销较小。
> - **Uniform-state diffusion 超越 masked diffusion**（ICML 2026, arXiv:2602.15014）：发现 uniform-state diffusion language models 在计算和数据维度上的缩放性能都优于 masked diffusion 和自回归模型。这一发现挑战了"masked diffusion 是扩散语言建模未来"的观点。
> - **实践启示**：perplexity 在同一扩散族内有参考价值，但跨族比较可能误导。GSM8K 实验中，1.7B 参数的 uniform-state diffusion 尽管 perplexity 更差，但实际推理准确率超过了自回归和 masked diffusion 模型。

---

### 二、Cerebras-GPT：首次公开验证 μP

[Cerebras-GPT](https://arxiv.org/abs/2304.03208) 遵循 Chinchilla 法则（20 tokens/param），训练了从 0.1B 到 13B 的 7 个模型，**首次公开验证了 μP 在大模型缩放中的有效性**。

#### 2.1 更稳定的缩放

![μP 更稳定的扩展规律](<../images/9-31-使用mμP展现了更稳定的扩展规律.png>)

> **图片讲解**：对比标准参数化（SP）和 μP 在不同模型规模下的训练 loss。μP 的曲线更平滑、更一致，说明超参数迁移后训练更稳定。

#### 2.2 更可预测的缩放（μTransfer）

![μP 更可预测的扩展规律](<../images/9-32-使用mμP展现了更可预测的扩展规律.png>)

> **图片讲解**：横轴 = 训练 FLOPs，纵轴 = 相对于 SP（Standard Parameterization，标准参数化，即不使用 μP 的默认做法）拟合缩放律的 loss 偏差。μP 的偏差更小且更稳定——**从 40M 小模型调好的超参数，直接迁移到所有规模，预测精度依然很高**。

核心流程（μTransfer）：

```
步骤 1：在 40M 小模型上做随机超参数搜索
         → 找到最优超参数：
           η_base = 6e-3  （基础学习率，实际 lr = η_base / r）
           σ_base = 0.08  （基础初始化标准差，实际方差 = σ_base² / r）
           m_emb  = 10    （嵌入层缩放因子，嵌入维度与词表相关需单独缩放）

步骤 2：通过 μP 缩放规则，直接将超参数迁移到 0.1B ~ 13B 所有规模
         → 不需要为每个大模型重新调参

步骤 3：验证结果——大模型性能与缩放律预测高度吻合
```

#### 2.3 SP vs μP 实现细节

![SP 和 μP 实现细节](<../images/9-33-SP和mμP的详细实现细节比较.png>)

> **图片讲解**：表格对比标准参数化（SP）和 μP 在初始化方差、学习率、输出层缩放等维度的具体差异。μP 的关键改动：矩阵参数的学习率和初始化方差都除以宽度缩放比例 $r$，向量参数保持不变。

![小模型超参数搜索](<../images/9-34-在小模型对三个关键超参数进行随机超参数搜索.png>)

> **图片讲解**：在 40M 小模型上对三个关键超参数的搜索结果。左图：基础学习率 η_base；中图：基础权重初始化标准差 σ_base；右图：嵌入层缩放因子 m_emb。每个点代表一次训练 run，最终选定最优值。

> 📎 **来源追溯**：Dey et al. (2023), *Cerebras-GPT: Open Compute-Optimal Language Models Using μP* (arXiv:2304.03208)。

> 💡 **补充（Web Search / μP 验证流程）**：Cerebras 实践指南提供了两个具体的 μP 验证步骤：
> 1. **Coordinate Check**：训练不同宽度的模型各 10 步，验证激活值的幅度是否宽度不变（width-invariant）。如果 μP 实现正确，不同宽度模型的激活值应该在同一数量级。
> 2. **μTransfer Test**：验证最优学习率是否跨宽度稳定——这是最终的集成测试。
> 3. **代理模型协议**：使用 hidden_size=256 的小模型，训练 20 tokens/param，对 4 个超参（base_init_std, base_lr, embedding_multiplier, output_logit_multiplier）做随机搜索。
>
> 注意：如果代理模型的 batch size 低于临界 batch size，迁移到大模型（通常 at/above 临界 batch size）时学习率可能不是最优。代码参考：[nanoGPT-mup](https://github.com/EleutherAI/nanoGPT-mup)。

---

### 三、MiniCPM：μP + WSD 调度器

MiniCPM 由面壁智能（清华 NLP 实验室孵化）推出，以 2B 模型打败当时的一众 7B 模型。

![MiniCPM 性能对比](<../images/9-35-MiniCPM与其他SOTA模型的性能比较.png>)

> **图片讲解**：MiniCPM 2B 在多个基准上超越 LLaMA 2 7B、Mistral 7B 等更大模型，展示了"小模型 + 精细缩放"的威力。下表为图中各评测数据集的说明：

**评测数据集速查**：

| 数据集 | 能力类型 | 内容简介 |
|--------|---------|---------|
| C-Eval | 中文知识 | 国内最权威的中文综合知识评测，覆盖中小学到大学多学科选择题 |
| C-MMLU | 中文推理 | MMLU 的中文翻译版，考察中文语境下的逻辑推理和跨学科知识 |
| MMLU | 英文综合 | 全球通用的英文综合能力标准，57 个学科选择题，参数量越大分数越高（符合缩放定律） |
| HumanEval | 代码生成 | OpenAI 开源的 164 道 Python 函数题，给注释写代码验证正确性 |
| MBPP | 代码基础 | 1000 道基础 Python 题，难度低于 HumanEval |
| GSM8K | 初等数学 | 小学数学应用题，文字逻辑 + 基础算数推理 |
| MATH | 高等数学 | 初高中竞赛级难题，缩放收益最慢，需要极大数据量 + 高质量数学语料 |
| BBH | 高级推理 | 高难度多步推理基准，包含逻辑题、反事实推理，衡量模型高级推理能力 |
| ARC-e/c | 科学常识 | ARC-e 为中小学基础科学题，ARC-c 为科学推理难题 |
| HellaSwag | 常识推理 | 给场景开头选最合理后续，考察世界常识和日常逻辑 |
| Avg/Avgen | 英文汇总 | 英文任务平均分（MMLU+BBH+ARC+HellaSwag 等） |
| Avgchn | 中文汇总 | 中文任务平均分（C-Eval+C-MMLU） |

**图表分析**：
1. **参数量缩放规律**：同系列模型 7B < 13B < 30B，分数整体上升，符合幂律缩放
2. **架构优势**：Mistral-7B 用更好的架构 + 预训练数据，7B 反超 LLaMA 2 13B
3. **中文优化**：Qwen、MiniCPM 中文原生预训练，中文指标（C-Eval、C-MMLU）碾压 LLaMA 系列
4. **数学/代码是硬指标**：MATH、HumanEval 提升最难，需要专门数据，缩放收益比通用能力慢

#### 3.1 μP 验证

![MiniCPM μP 稳定扩展](<../images/9-36-使用muP稳定扩展.png>)

MiniCPM 与 CerebrasGPT 得到了相似的 μP 超参数：scale_emb ≈ 10-12, scale_depth ≈ 1.4, init_std ≈ 0.08-0.1, lr ≈ 0.01。这说明 μP 的缩放规则具有**跨团队的一致性**。

#### 3.2 WSD 学习率调度器

**问题**：Chinchilla 分析需要对每个 (N, D) 组合从头训练，成本为 O(m×n)。如何降低成本？

**WSD（Warmup-Stable-Decay）调度器**：

![学习率策略比较](<../images/9-41-学习率策略比较.png>)

> **图片讲解**：
> - **Cosine**（传统）：学习率从 warmup 升到峰值，再用余弦曲线衰减到 0。一旦开始训练就无法中途评估"训练到一半"的效果。
> - **WSD**：三阶段——Warmup（升）→ Stable（恒定）→ Decay（快速衰减）。关键优势：**共享同一个 Stable 阶段**，只需在不同时间点插入短 Decay（约 10% 步数），就能模拟不同数据量的训练效果。

```
WSD 的核心优势：

传统 Cosine：想评估"训练到 40N tokens"的效果 → 必须从头跑一次
WSD：训练到 80N tokens（Stable 阶段）
     → 在 40N 处插入 4N Decay → 得到 40N 的效果
     → 在 60N 处插入 6N Decay → 得到 60N 的效果
     → 一次训练，多个数据点！
```

![WSD 衰减阶段 loss 急剧下降](<../images/9-42-模型训练损失在WSD的衰减阶段突然下降.png>)

> **图片讲解**：Stable 阶段 loss 缓慢下降；进入 Decay 阶段后 loss **急剧下降**，在很短时间（约 10% 步数）内达到甚至低于 Cosine 的最终 loss。

> 💡 **补充（Web Search / WSD 理论与实践）**：
> - **"河谷地形"理论**（arXiv:2410.05192）：Stable 阶段的恒定学习率让模型在 loss 地形的平坦"谷底"广泛探索；Decay 阶段的快速衰减引导模型收敛到谷底最深处。这解释了为什么 Decay 阶段 loss 会急剧下降。
> - **WSD-S 变体**（Stanford/Tengyu Ma 组）：简化版 WSD，复用之前 checkpoint 的 Decay 阶段，只保留一个主分支。性能与独立调优的 Cosine 调度器相当。
> - **实践细节**：Decay 阶段约占总步数 5-10%（MiniCPM 用约 5%）；Decay 后**不能再回到 Stable 阶段**（MiniCPM GitHub issue #116 确认）；Cosine/Linear/Exponential 衰减均可，Cosine 最常用。
> - **2024 年现状**：WSD 已成为现代 LLM 训练的事实标准（MiniCPM、DeepSeek-V3 等均采用），因为它天然支持持续训练和领域适应。

#### 3.3 最优批次大小公式

![三种规模模型的批次大小曲线](<../images/9-38-三种不同规模模型使用不同批次大小进行训练的损失曲线.png>)

> **图片讲解**：三个子图分别对应 9M、30M、170M 参数模型。X 轴 = Batch Size，Y 轴 = 已处理 token 数。红色曲线连接每个数据量下的最优 batch size。

![连接最优批次大小](<../images/9-39-连接最有批次大小.png>)

拟合得到批次大小与 loss 的关系：

$$ bs = \frac{1.21 \times 10^9}{L^{6.24}} $$

**变量说明**：$bs$ = 最优批次大小，$L$ = C4 验证集损失。含义：**想要更低的 loss，就需要更大的 batch**。指数 6.24 说明 batch size 对 loss 非常敏感——loss 降低一点，最优 batch 就要增大很多。

#### 3.4 学习率稳定性验证

![MiniCPM 学习率稳定性](<../images/9-40-MiniCPM使用mμP保持了学习率的稳定性.png>)

> **图片讲解**：从 0.04B 到 2.1B（扩大 50 倍），所有规模模型的最优学习率都集中在 **0.01 附近**，完美验证了 μP 的学习率迁移性。

#### 3.5 缩放分析结果

![WSD 缩放实验结果](<../images/9-44-使用WSD在三种任务上进行扩展实验的结果.png>)

> **图片讲解**：3 行 6 子图，覆盖 5 类训练语料（Code、英文 Wikihow、中文 Wikihow、Ultratext、中文 Yayi）+ 全局平均。
> - **上排（3 张线图）**：Loss 随算力的缩放趋势。横轴 = 算力（对数尺度），纵轴 = 测试 Loss。代码、英文、中文三大领域全部严格遵循幂律缩放。
> - **中排/下排（热力图）**：横轴 = 非嵌入层参数量，纵轴 = 总算力（FLOPs）。颜色：红色 = Loss 高（效果差），蓝色 = Loss 低（效果好）。黑色等高线 = 相同 Loss 的参数-算力配比，黑色散点 = 实际训练的模型。

![WSD fit 结果](<../images/9-45-使用WSD在三种任务上进行扩展实验的fit结果.png>)

> **图片讲解**：此图是图 9-44 热力图的拟合结果，6 个子图分别对应 5 类训练语料 + 全局平均。每个子图给出 Chinchilla 联合缩放公式的拟合参数、$K^2$、$\eta$ 和最优配比。
>
> **各领域拟合公式与参数对比**：
>
> | 领域 | 拟合公式 | $K^2$ | $\eta$ | $D_{\text{opt}}/N_{\text{opt}}\|_{C=10^{21}}$ |
> |------|---------|-------|--------|----------------------------------------------|
> | Code | $\frac{3.32 \times 10^{-2}}{N^{0.37}} + \frac{2.17 \times 10^{-1}}{D^{0.34}} + 0.17$ | 0.01 | -0.05 | 194.93 |
> | English (Wikihow) | $\frac{8.08 \times 10^{-2}}{N^{0.26}} + \frac{2.97 \times 10^{-1}}{D^{0.18}} + 0.27$ | 0.02 | -0.18 | — |
> | Chinese (Wikihow) | $\frac{5.14 \times 10^{-2}}{N^{0.35}} + \frac{3.90 \times 10^{-1}}{D^{0.18}} + 0.40$ | 0.01 | -0.33 | 833.95 |
> | Ultratext | $\frac{7.54 \times 10^{-2}}{N^{0.30}} + \frac{2.92 \times 10^{-1}}{D^{0.30}} + 0.25$ | 0.01 | -0.00 | 95.60 |
> | Chinese (Yayi) | $\frac{1.53 \times 10^{-1}}{N^{0.19}} + \frac{3.76 \times 10^{-1}}{D^{0.17}} + 0.35$ | 0.01 | -0.05 | — |
> | **Average** | $\frac{7.15 \times 10^{-2}}{N^{0.29}} + \frac{3.00 \times 10^{-1}}{D^{0.23}} + 0.31$ | 0.01 | -0.10 | **191.87** |
>
> **公式变量说明**：$N$ = 非嵌入参数量，$D$ = 训练 token 数，$A/N^{\alpha}$ = 参数不足误差，$B/D^{\beta}$ = 数据不足误差，$C$ = 不可约损失下限。
>
> **$K^2$**：拟合常数（来自最优比例公式 $N_{\text{opt}}/D_{\text{opt}} = K^2 \cdot (C/6)^{\eta}$），多数子图 $K^2 \approx 0.01$，拟合优度极高。
>
> **$\eta$（倾斜系数）的分领域含义**：
>
> | $\eta$ 值 | 含义 |
> |-----------|------|
> | Code: -0.05 | 更吃模型，优先堆参数量 |
> | English: -0.18 | 偏向数据 |
> | Chinese Wikihow: -0.33 | 最吃数据，优先堆高质量中文数据 |
> | Ultratext: -0.00 | 参数/数据几乎均衡 |
> | Chinese Yayi: -0.05 | 偏向参数 |
>
> **关键结论**：在 $C = 10^{21}$ FLOPs 预算下，中文（Wikihow）最优 $D/N = 833.95$，代码为 $194.93$，全局平均为 $191.87$。中文更吃数据、代码更吃模型。
>
> **为什么代码 Loss 最低（"简单"）但代码任务最难？**
>
> 这里的 Loss = **自回归预测损失（NLL Loss）**，衡量的是"预测下一个 token 的难度"，**不是任务本身的难度**：
> - **代码**：语法严格（关键字、缩进、括号匹配），上下文约束极强，序列熵极低 → 单步预测容易 → NLL Loss 低。但代码需要**几十步长程逻辑连贯**，一步错步步错 → 全局任务极难（HumanEval 分数极低）
> - **中文**：语法自由、歧义多，上下文约束弱，序列熵极高 → 单步预测困难 → NLL Loss 高。但日常中文**长程逻辑简单** → 全局任务容易（C-Eval 分数高）
>
> 一句话：**NLL Loss 衡量"下一个 token 好不好猜"，不是"任务好不好做"**。

> 📎 **来源追溯**：Hu et al. (2024), *MiniCPM: Unveiling the Potential of Small Language Models with Scalable Training Strategies* (arXiv:2404.06395)。

---

### 四、DeepSeek：不用 μP 的务实路线

[DeepSeek LLM](https://arxiv.org/abs/2401.02954) 代表了另一种技术路线：**不使用 μP，直接拟合缩放定律来指导超参选择**。

![DeepSeek 性能对比](<../images/9-47-DeepSeek与其他SOTA模型的性能比较.png>)

#### 4.1 直接拟合最优超参

![批次大小和学习率组合](<../images/9-48-给定预算下批次大小和学习率的组合.png>)

> **图片讲解**：在固定计算预算下，扫描不同 batch size + learning rate 组合的泛化误差。左图（177M FLOPs/token）的最优深色区域在**右下角**（大 batch + 小 lr），右图（另一模型规模）的最优区域在**左下角**（大 batch + 大 lr）。两个热力图的最优区域方向不同，但都呈现**大 batch 更优**的规律。原文强调：泛化误差在大范围 batch size 和 learning rate 组合下都保持稳定，近最优性能可在较宽参数空间内实现。

![不同预算下的最优超参趋势](<../images/9-49-不同计算预算下最优批次大小和最优学习率的变化趋势.png>)

> **图片讲解**：
> - (a) 批次大小缩放曲线：在 log-log 坐标下呈线性 → 幂律关系
> - (b) 学习率缩放曲线：数据点有聚集，呈"近乎最优的宽泛区间"——现实中不存在完美精确的最优值

DeepSeek 的方法：**不依赖 μP 的理论保证，而是用经验数据直接拟合** batch size 和 learning rate 与计算预算的幂律关系。

#### 4.2 多步学习率调度器

![不同学习率调度器对比](<../images/9-50-不同学习率调度器对训练损失的影响.png>)

> **图片讲解**：
> - (a) Cosine vs 多步调度器：处理完 100B tokens 后，两者最终 loss 非常接近
> - (b) 不同阶段比例对性能影响有限

**为什么选择多步调度器？** 便于**持续训练**（Continual Training）——可以在已有模型基础上继续训练，重复利用之前的训练成果。Cosine 调度器一旦开始就无法灵活续训。

#### 4.3 IsoFLOP 分析

![DeepSeek Scaling Law](<../images/9-51-计算预算-模型规模和数据规模之间的Scaling-Law.png>)

> **图片讲解**：
> - (a) 每条虚线 = 固定计算预算（1e17 ~ 3e20 FLOPs），U 形曲线的最低点 = 该预算下的最优模型规模
> - (b) 最优模型规模 $M_{\text{opt}} \propto C^a$（幂律关系，$a$ 是模型缩放指数）
> - (c) 最优数据规模 $D_{\text{opt}} \propto C^b$（幂律关系，$b$ 是数据缩放指数）

#### 4.4 预测精度验证

![DeepSeek 预测验证](<../images/9-52-DeepSeek在不同训练计算预算下在验证集上的性能表现.png>)

> **图片讲解**：蓝色星形点 = 7B 和 67B 大模型的实际性能，虚线 = 小模型实验拟合的缩放曲线。**两者高度吻合**——小模型实验可以准确预测计算量大 1000 倍的大模型性能。

> 📎 **来源追溯**：DeepSeek-AI (2024), *DeepSeek LLM: Scaling Open-Source Language Models with Longtermism* (arXiv:2401.02954)。

---

### 五、其他模型的缩放策略

#### 5.1 LLaMA 3（2024）

![LLaMA 3 IsoFLOP 曲线](<../images/9-53-Llama3的IsoFLOPs的Scaling-Law曲线.png>)

LLaMA 3 使用 IsoFLOP 分析确定最优模型大小和数据量，采用 **39:1** 的 token/param 比例（远超 Chinchilla 的 20:1）。

![LLaMA 3 下游任务预测](<../images/9-54-对ARC-Challenge的Scaling-law预测.png>)

> **图片讲解**：左图 = FLOPs vs NLL（负对数似然），右图 = NLL vs 准确率。通过两步映射（FLOPs → NLL → Accuracy），可以**预测大模型在下游任务上的表现**，而不需要实际训练大模型去跑 benchmark。

> 📎 **来源追溯**：Meta (2024), *The Llama 3 Herd of Models* (arXiv:2407.21783)。

#### 5.2 Hunyuan-1（2024）：MoE 缩放

![Hunyuan MoE Scaling Law](<../images/9-55-Hunyuan混合专家模型的Scaling-Law.png>)

> **图片讲解**：
> - 左图：不同计算预算下，训练 loss 与**激活参数**（非总参数）的关系
> - 右图：激活参数与最优计算预算的缩放关系

MoE 模型的特殊性：总参数量很大但每次推理只激活一部分。Hunyuan 发现最优比例约为 **96 tokens per activated parameter**——注意是**激活参数**，不是总参数。

#### 5.3 MiniMax-01（2025）：架构缩放

![MiniMax Scaling Laws](<../images/9-56-MinMax-Scaling-Laws.png>)

> **图片讲解**：三种注意力机制（Softmax Attention、Lightning Attention、Hybrid-lightning）在不同计算预算下的 Loss、参数量和数据量缩放规律。**不同架构有不同的缩放曲线**——MiniMax 为每种注意力机制单独拟合了缩放定律。

MiniMax 的创新：不是选择"最好的架构"然后缩放，而是**为每种架构建立自己的缩放定律**，在给定预算下选择最优架构。

> 📎 **来源追溯**：MiniMax (2025), *MiniMax-01: Scaling Foundation Models with Lightning Attention*。

---

### 六、缩放策略对比总结

```
团队            μP?    调度器        D/N 比例       核心方法
────────────────────────────────────────────────────────────
Cerebras-GPT    ✅     Cosine        20:1           Chinchilla + μP 验证
MiniCPM         ✅     WSD           192:1          μP + WSD 高效分析
DeepSeek        ❌     多步 LR       经验拟合        直接拟合 bs/lr 缩放律
LLaMA 3         ❌     Cosine        39:1           IsoFLOP 分析
Hunyuan         —      —             96:1(激活)      MoE 专用缩放
MiniMax         —      —             —              架构级缩放定律
```

**关键洞察**：
1. **μP 不是必须的**：DeepSeek 不用 μP 也能得到好的缩放律，代价是需要更多实验
2. **D/N 比例在持续增长**：从 Chinchilla 的 20:1 到 MiniCPM 的 192:1，"过度训练"是大趋势
3. **调度器选择影响分析成本**：WSD 比 Cosine 更适合 Chinchilla 风格分析（一次训练多个数据点）
4. **不同架构需要不同的缩放律**：MiniMax 证明了这一点

> 💡 **补充（Web Search / Chinchilla 复现实验）**：Epoch AI 在 2024 年对 Chinchilla 进行了独立复现（arXiv:2404.10102），发现最优比例约为 **25.6 tokens/param**，略高于原论文的 20:1。这表明最优 D/N 比例可能随数据质量等因素变化，20:1 是一个经验近似值而非理论常数。DeepSeek 的经验公式也给出了最优学习率与计算预算的关系：$\eta_{\text{opt}} = 0.3118 \times C^{-0.0238}$。

---

## 🧠 本模块问题

请在下方回答以下问题后，输入 `提交作业` 提交。

**Q1**：MiniCPM 使用 WSD 调度器将 Chinchilla 分析成本从 O(m×n) 降低到 O(m+n)。请解释 WSD 的三个阶段（Warmup、Stable、Decay），以及为什么"共享 Stable 阶段"能实现这种成本降低。

**Q2**：DeepSeek 没有使用 μP，而是直接拟合 batch size 和 learning rate 的缩放定律。这种"经验路线"相比 μP 的"理论路线"有什么优缺点？

**Q3**：对比 Cerebras-GPT（20:1）、LLaMA 3（39:1）、MiniCPM（192:1）的 D/N 比例差异。为什么不同团队得出的比例差异如此之大？这说明 Chinchilla 的 20:1 结论有什么局限性？

---

<!-- 学习者作答区（请在此处填写你的答案） -->

**A1**：

WSD 有三个阶段，Warmup、Stable 以及 Decay。其中 Stable 阶段保持学习率不变，这部分过程是可以共享的。
即训练数据量从 10n  40n 到 80n，都可以在 stable 这个过程找到自己的位置，并从可以共享的部分继续，然后再做 decay。
这大大降低了 Chinchilla 的分析成本，因为不再像 Cosine 调度器一样，需要从头开始训练。


**A2**：

Deepseek 的“经验路线” 通过大量小模型实验 拟合  batch size 和 learning rate，能更有效地找到 tokens/params 之间的比例关系，对要训练的大模型的性能有更好地评估（更准确）。

但是因为需要自己进行大量小模型实验来拟合，增加了前期实验的时间。


**A3**：

差异大是因为 训练数据不同、模型架构不同 造成的
说明 Chinchilla 的 20：1 只是针对特定模型、特定数据的经验比例，对于不同架构、不同数据来看，还是需要拟合自己的 D/N 来更好的预估 scale 的曲线。


---

<!-- 教师批改区（提交作业后由导师填写，请勿手动修改） -->

### 📝 批改结果

**Q1 批改**：核心方向正确，但**机制描述不够精确**——缺少"一次训练、多次 Decay"的关键机制。 — 得分：**7/10**

- ✅ WSD 三阶段（Warmup/Stable/Decay）描述正确
- ✅ "Stable 阶段可共享"——方向正确
- ⚠️ 缺少关键机制：WSD 的核心不是"在 Stable 中找到位置再 Decay"，而是**一次完整的 Stable 阶段训练完成后，在不同 checkpoint 处插入短 Decay（约 5-10% 步数），模拟不同数据量的训练效果**。即 O(m+n) 而非 O(m×n)：m 个模型大小 × n 个数据量，传统方法需 m×n 次完整训练；WSD 只需训练 m 个模型各 1 次 Stable，再在 n 个 checkpoint 各做 1 次短 Decay → 总共 m+n 次

<details>
<summary>📖 Q1 参考答案</summary>

**WSD 三阶段**：
1. **Warmup**：学习率从 0 线性上升到峰值
2. **Stable**：学习率保持恒定，占训练的大部分时间（约 90% 步数）
3. **Decay**：学习率快速衰减到 0，占约 5-10% 步数

**为什么能降低成本**：

Cosine 调度器需要预先知道总步数，衰减曲线贯穿整个训练。想评估"训练到 40N tokens"的效果，必须从头跑一次完整训练。

WSD 的 Stable 阶段共享机制：
- 训练 1 个模型，跑完完整 Stable 阶段（如 80N 步）
- 在 40N 步 checkpoint 处插入 4N 步 Decay → 得到 40N tokens 的性能
- 在 60N 步 checkpoint 处插入 6N 步 Decay → 得到 60N tokens 的性能
- **一次训练，多个数据点**

传统 Cosine：m 个模型大小 × n 个数据量 = **O(m×n) 次完整训练**
WSD：m 个模型各 1 次 Stable + n 个 checkpoint 各 1 次短 Decay = **O(m+n) 次**
</details>

---

**Q2 批改**：结论正确，但**缺少对比 μP 的关键差异**。 — 得分：**6/10**

- ✅ "经验路线通过大量小模型实验拟合"——正确
- ✅ "增加了前期实验时间"——正确但过于简略
- ⚠️ 缺少 μP 对比：μP 的核心优势是**理论保证**——通过数学推导确定学习率的缩放规则，不需要额外实验即可跨规模迁移。DeepSeek 的经验路线需要**为每个计算预算单独做扫描实验**，没有理论外推能力
- ⚠️ 缺少 μP 的局限：μP 只处理**模型宽度**引起的学习率变化，不处理 batch size 和数据量的变化——DeepSeek 的方法直接拟合了这些维度

<details>
<summary>📖 Q2 参考答案</summary>

**μP "理论路线"**：
- 优点：有数学推导保证，小模型调好的学习率可直接迁移到大模型（宽度维度），无需额外实验
- 局限：只处理**模型宽度**引起的学习率变化，不处理 batch size、数据量的变化

**DeepSeek "经验路线"**：
- 优点：① 直接拟合 batch size 和 learning rate 的幂律关系，覆盖了 μP 不处理的维度；② 不依赖 μP 的理论假设，适用于任何架构；③ 给出了具体公式（如 $\eta_{\text{opt}} = 0.3118 \times C^{-0.0238}$），可直接计算
- 缺点：需要为每个计算预算做扫描实验，前期成本更高；没有理论外推能力，无法从一个小规模直接预测所有大规模

**实际选择**：两者并不互斥。μP 解决"模型变宽时 lr 怎么调"，DeepSeek 的方法解决"不同计算预算下 bs 和 lr 怎么调"——维度不同。MiniCPM 则结合了两者：用 μP 处理宽度，用 WSD 高效分析数据-模型配比。
</details>

---

**Q3 批改**：核心结论正确，但**缺少具体机制解释**。 — 得分：**7/10**

- ✅ "训练数据不同、模型架构不同"——正确
- ✅ "Chinchilla 20:1 是特定模型/特定数据的经验比例"——正确
- ⚠️ 缺少具体机制：为什么数据质量影响 D/N 比？因为 Chinchilla 公式 $L = E + AN^{-\alpha} + BD^{-\beta}$ 中的系数 $A$、$B$ 和指数 $\alpha$、$\beta$ 都取决于数据分布。高质量数据 → $B$ 更小（数据项贡献小）→ 更多参数收益更大 → D/N 偏低；低质量/多样化数据 → $B$ 更大 → 更多数据收益更大 → D/N 偏高
- ⚠️ 缺少 Chinchilla 复现实验的佐证：Epoch AI 2024 年复现得到 25.6:1（非 20:1），说明即使在同一数据集上，拟合方法和实验规模也会影响结果

<details>
<summary>📖 Q3 参考答案</summary>

**D/N 比例差异的具体原因**：

1. **数据质量与多样性**：Chinchilla 公式 $L = E + AN^{-\alpha} + BD^{-\beta}$ 中的 $B$ 和 $\beta$ 取决于数据分布
   - 高质量/单一数据 → $B$ 小（数据项贡献小）→ 增加参数收益更大 → D/N 偏低
   - 低质量/多样化数据 → $B$ 大（数据项贡献大）→ 增加数据收益更大 → D/N 偏高
   - 这解释了为什么中文（数据多样性高）D/N = 833.95，代码（语法约束强）D/N = 194.93

2. **模型架构**：不同架构的缩放指数 $\alpha$、$\beta$ 不同
   - MoE 模型（Hunyuan）：总参数大但激活参数小 → 用激活参数衡量 → 96:1
   - Dense 模型（Cerebras-GPT）：标准参数 → 20:1

3. **训练调度器**：WSD vs Cosine 影响缩放律拟合结果
   - WSD 可以高效探索更多 (N, D) 组合 → 拟合更精确 → 可能发现更高的最优比例

4. **Chinchilla 复现实验**：Epoch AI (2024) 复现得到 **25.6:1**，说明即使在同一数据集上，拟合方法和实验规模也会影响结果。20:1 不是理论常数，而是经验近似值。

**结论**：Chinchilla 的 20:1 结论的局限性在于——它是**特定数据、特定架构、特定实验规模**下的经验最优。实际部署中，D/N 比例需要根据数据质量、架构类型、推理成本等因素重新拟合。
</details>

---

**综合评价**：三个问题的核心结论都正确，但普遍存在"只给结论、缺少机制"的问题。Q1 缺少"一次训练多次 Decay"的关键机制；Q2 缺少 μP 和 DeepSeek 方法在维度上的差异对比；Q3 缺少 Chinchilla 公式中系数 $A$、$B$ 如何受数据质量影响的分析。建议在回答"为什么"类问题时，先给出数学/机制层面的解释，再给结论。

**批改时间**：2026-05-19
