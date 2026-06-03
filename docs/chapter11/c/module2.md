# 第 11 章：数据工程 — 模块 2：智能数据筛选

> 📍 学习进度：第 11 章，第 2 / 4 模块
> 📅 生成时间：2026-05-26

---

## 学习目标

- 理解 KenLM 在数据筛选中的作用及"好模型+坏模型"集成策略
- 掌握 FastText 分类器的核心原理：n-gram 哈希化 + 平均池化
- 理解 DSIR 重要性重采样的数学思想和工作流程
- 能够区分三种筛选方法各自适用的场景

---

## 一、回顾：从"粗筛"到"精筛"

模块 1 中，启发式规则过滤和困惑度过滤是**粗筛**——它们回答的是"这段文本是不是正常的自然语言"。但训练一个高质量的 LLM，仅仅"正常"是不够的——你还需要判断：

```
这段文本：
  - 对模型学习有帮助吗？（教育价值）
  - 属于我想要的领域吗？（领域匹配）
  - 和目标数据分布接近吗？（分布对齐）
```

这三个问题分别对应本节要讲的三种方法：

| 方法 | 回答的问题 | 核心技术 |
|------|-----------|---------|
| KenLM | "这段文本像不像高质量语料？" | n-gram 语言模型 + 困惑度对比 |
| FastText | "这段文本属于哪个类别？" | n-gram 哈希化 + 线性分类 |
| DSIR | "这批数据与目标分布有多接近？" | 重要性重采样 + 特征空间降维 |

---

## 二、KenLM：最轻量的"质量裁判"

### 2.1 什么是 KenLM？

KenLM 是一个高效的 **n-gram 语言模型**库（Heafield, EMNLP 2011）。它的核心思想非常朴素：

```
n-gram 语言模型：
  "I love machine ____" → 下一个词是什么？

  统计大量文本中 "I love machine" 后面出现过的词：
    "learning" 出现了 800 次  → P("learning" | "I love machine") = 0.8
    "translation" 出现了 150 次 → P("translation" | "I love machine") = 0.15
    "gun" 出现了 0 次          → P("gun" | "I love machine") = 0（在数据中从未见过）

  5-gram 模型：回看最近 4 个词来预测第 5 个词
```

**为什么 KenLM 而不是 GPT-2 来做质量评估？**

| | KenLM (5-gram) | GPT-2 (Transformer) |
|---|---|---|
| 模型大小 | 几百 MB（概率表） | 数百 MB~GB（神经网络权重） |
| 推理速度 | CPU 上每秒处理数万行 | 需要 GPU，慢数百倍 |
| 可解释性 | 完全透明（查概率表） | 黑箱 |
| 适用场景 | 大规模数据快速打分 | 需要深层语义理解的任务 |

CCNet 选择 KenLM 而非 Transformer LLM 来做数据过滤，正是因为处理 Common Crawl 级别的数据量（数十亿文档），速度是第一位的。

> 🌐 **补充（Web Search / NVIDIA NeMo Curator, 2025）**：NVIDIA 的数据处理框架 NeMo Curator 中，KenLM 困惑度被列为"三大经典预训练质量信号之一"，与 Gopher 规则和 C4 规则并列。Dolma（Allen AI 的数据集，用于 OLMo 训练）也使用 KenLM 作为核心过滤组件。

### 2.2 "好模型 + 坏模型"集成策略

单一 KenLM 模型（只训练在 Wikipedia 上）有一个盲区：它只知道"像不像 Wikipedia"，但不知道"像不像垃圾"。

> 🌐 **补充（Web Search / Rethinking KenLM, arXiv 2409.09613, 2024.09）**：最新研究提出了一种改进方案——**同时训练两个 KenLM 模型**：

```
Good KenLM（好模型）：
  训练数据：Wikipedia、高质量书籍、学术论文
  → 用于计算 "好困惑度" PPL_good
  → 数值越低 = 越像高质量语料

Bad KenLM（坏模型）：
  训练数据：垃圾邮件、SEO 水货、广告页面、低质量论坛
  → 用于计算 "坏困惑度" PPL_bad  
  → 数值越低 = 越像垃圾

最终质量分数 = PPL_good / PPL_bad
  （好困惑度低且坏困惑度高 → 分数低 → 质量高）
```

**为什么这个比值比单一 PPL 更准？**

用一段垃圾 SEO 文本来演示：

```
文本： "Buy now!!! Best deals!!! Click here!!! Cheap!!! Limited offer!!!"

单一模型（只在 Wikipedia 训练）：
  PPL = 350 → "嗯，这个不太像 Wikipedia……但也不太正常……不好说"
  
双模型：
  PPL_good = 350（不像好文本——对）
  PPL_bad  = 15 （非常像垃圾——对！这是关键增量的信息）
  比值 = 350/15 = 23.3 → 明确是垃圾！
```

---

## 三、FastText：轻量级文本分类器

### 3.1 核心思想：词袋 + n-gram + 哈希化

FastText 是 Facebook 在 2016 年提出的高效文本分类方法（Joulin et al., "Bag of Tricks for Efficient Text Classification", EACL 2017）。尽管论文已经发表近十年，它在 LLM 数据工程中仍然有重要地位——因为它可以在 **CPU 上每秒分类 2000+ 条文本**。

FastText 之所以快，是因为它把文本分类**极度简化**为三个操作：

```
输入文本："I love this product"
    ↓
① n-gram 分词：["I", "love", "this", "product",           ← 1-gram（单词）
                 "I love", "love this", "this product"]    ← 2-gram（词对）
    ↓
② 哈希化：hash("I") % 2000000 → 1234567
          hash("love") % 2000000 → 891234
          hash("I love") % 2000000 → 567890
          ...（每个 n-gram 映射到一个固定范围的整数）
    ↓
③ 嵌入 + 平均 + 分类：
    每个哈希值 → 查 Embedding 表 → 得到向量
    所有向量 → 求平均 → 文档表示向量
    文档向量 → 线性分类器 → 输出类别
```

**关键洞察——为什么可以忽略词序？**

在情感分析中，"I love this" 和 "this I love" 意思确有不同。但在文档级别的数据筛选任务中（判断一篇文章是否是教科书质量），单个词的统计特征（如"therefore"、"theorem"、"proof"的出现频率）已经足够做出准确判断。牺牲词序换来的速度优势（数百倍）让 FastText 成为处理**数十亿文档**的首选。

### 3.2 n-gram 哈希化：FastText 最精妙的设计

为什么要把 n-gram 哈希化而不是直接存词？考虑一个真实场景：

```
英语词汇量：约 100 万
如果存所有 2-gram 组合：100万 × 100万 = 1 万亿种可能
  但实际语料中只会出现极小一部分
  用一个巨大的 Embedding 矩阵存所有可能的 2-gram → 内存爆炸
```

**哈希化的解决方案**：

```
不管实际 n-gram 有多少种，统一映射到固定数量的"桶"（buckets）

例如：2,000,000 个桶
  hash("the cat") % 2,000,000 = 某整数
  hash("quantum entanglement") % 2,000,000 = 另一整数

碰撞容忍：不同 n-gram 可能映射到同一个桶
  → 这在实践中影响很小（桶足够多时碰撞率极低）
  → 而且有轻微的正则化效果（相似模式被映射到相近空间）
```

> 🌐 **补充（Web Search / FastText 原始论文）**：原论文使用 2,000,000 个桶处理 bigram，在处理超过 10 亿词的语料时效果最佳。当 n-gram 范围扩展到 5-gram 时，桶数增加到 1 亿。课程代码中为了演示将桶数设为 5，但实际工业部署是这个数量的数百万倍。

### 3.2.1 哈希值如何映射到 Embedding 表？

你可能会问："hash 只是一个随机整数，跟 Embedding 表有什么关系？"

**答案比想象的更简单——哈希值就是表的下标**：

```
Embedding 表本质上是一个矩阵：shape = [num_buckets, embed_dim]

  下标 0  → [0.23, -0.15, 0.78, ..., 0.41]   ← 16维向量
  下标 1  → [-0.33, 0.62, -0.19, ..., 0.05]
  下标 2  → [0.71, 0.08, -0.54, ..., -0.22]
  ...
  下标 4  → [0.12, -0.88, 0.33, ..., 0.67]
```

```
具体流程：

  "I love"  
    → hash("I love") = 8972341056321  
    → 8972341056321 % 5  =  1      ← 取模得到下标
    → Embedding[1] = [-0.33, 0.62, -0.19, ..., 0.05]  ← 直接查表
```

**那这个表是怎么来的？**

`nn.Embedding(num_buckets, embed_dim)` 创建时，矩阵中的每个向量是**随机初始化**的（和所有神经网络的权重一样）。在训练过程中，通过反向传播不断更新这些向量——让对分类有帮助的 n-gram 对应的向量"漂移"到有利于决策的方向。

用一个比喻来理解：

```
Embedding 表 = 一本有 5 页的"笔记本"，每页有 16 行空白（随机初始化的向量）

训练前：
  第 0 页：随机涂鸦
  第 1 页：随机涂鸦
  ...

训练过程中（通过反向传播）：
  "love" 映射到下标 1 → 第 1 页被反复修改 → 逐渐被"写满"正面情感的编码
  "hate" 映射到下标 3 → 第 3 页被反复修改 → 逐渐被"写满"负面情感的编码

最终：
  同一个桶中的不同 n-gram 共享同一个向量（碰撞），
  但桶足够多时，有区分力的 n-gram 大概率落在不同桶中
```

**关键洞察**：`hash()` 函数在不同 Python 进程中结果不同，因此 Embedding 表**不能跨进程/跨次运行复用**。课程代码的训练和推理在同一个脚本中完成，避开了这个问题。工业 FastText 使用 FNV-1a 等确定性哈希算法，保证跨平台一致性。

### 3.3 完整代码解析

**Step 1：n-gram 生成**

```python
# 来自 docs/chapter11/FastText.py

def get_ngrams(tokens, n):
    """
    生成 n-gram 词组。

    例子：tokens=["I","love","this"], n=2
    返回：["I", "love", "this",       ← 1-gram（单个词）
           "I love", "love this"]     ← 2-gram（相邻词对）
    """
    ngrams = []
    for i in range(len(tokens)):
        for j in range(1, n + 1):
            if i + j <= len(tokens):
                ngrams.append(" ".join(tokens[i:i + j]))
    return ngrams
```

Python 内置的 `hash()` 函数在不同进程中结果不同（这是有意为之的安全特性），但课程代码中所有操作在同一个 Python 进程内进行，所以一致性由进程内保证。

**Step 2：哈希化**

```python
def hash_ngrams(tokens, num_buckets, ngram):
    ngrams = get_ngrams(tokens, ngram)
    # Python 内置 hash() + 取模 → 映射到 [0, num_buckets-1]
    return torch.tensor([hash(g) % num_buckets for g in ngrams], dtype=torch.long)
```

**Step 3：批处理——处理不同长度的句子**

```python
def collate_fn(batch):
    """
    一个 batch 中可能有不同长度的句子，需要填充到相同长度。

    例：batch 中有两句：
      "I love it"         → 5 个 n-gram → [hash1, hash2, hash3, hash4, hash5]
      "This is terrible"  → 7 个 n-gram → [hash1, hash2, ..., hash7]
      
    填充后（填 0 补到 max_len=7）：
      [hash1, hash2, hash3, hash4, hash5, 0, 0]
      [hash1, hash2, hash3, hash4, hash5, hash6, hash7]
    """
    max_len = max(len(x[0]) for x in batch)
    padded = []
    for hashed_ids, label in batch:
        pad_len = max_len - len(hashed_ids)
        padded_ids = F.pad(hashed_ids, (0, pad_len), value=0)
        padded.append(padded_ids)
    return torch.stack(padded), torch.tensor(labels)
```

**Step 4：模型结构——FastText 的本质**

```python
# 来自 docs/chapter11/FastText.py

class FastTextClassifier(nn.Module):
    def __init__(self, num_buckets, embed_dim, num_classes):
        super().__init__()
        # 嵌入层：将每个哈希值（0~num_buckets-1）映射为 embed_dim 维向量
        self.embedding = nn.Embedding(num_buckets, embed_dim)
        # 线性分类器：将平均向量映射到分类标签
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        # x 形状: [Batch, Seq_Len]，每个元素是哈希值

        # Step A：查嵌入表 → [Batch, Seq_Len, Embed_Dim]
        embedded = self.embedding(x)

        # Step B：平均池化——FastText 的核心操作
        # 把所有 n-gram 的嵌入向量求平均 → [Batch, Embed_Dim]
        # ⚠️ 注意：这里丢失了所有词序信息！
        # 但这也是 FastText 能这么快的原因——没有 RNN、没有 Attention
        avg_embedded = embedded.mean(dim=1)

        # Step C：线性分类 → [Batch, Num_Classes]
        logits = self.fc(avg_embedded)
        return logits
```

**FastText 模型的参数规模对比**：

```
课程演示代码：
  num_buckets = 5       → Embedding: 5×16 = 80 个参数
  embed_dim = 16
  分类层：16×2 = 32 个参数
  总参数：~112 个 → 极简演示

真实工业部署（FastText 原论文）：
  num_buckets = 2,000,000  → Embedding: 2M×100 = 2 亿参数
  embed_dim = 100
  但通过哈希技巧，Embedding 矩阵不需要预先构造完整的 1 万亿维
```

### 3.4 完整训练流程（数值推演）

课程代码在 6 条样本上训练 1000 轮，演示完整的学习过程：

```
数据：6 条影评 (3 正 3 负)
  1. "I love this product"  → 正面 (1)
  2. "This is terrible"     → 负面 (0)
  3. "Amazing experience"   → 正面 (1)
  4. "I hate it"            → 负面 (0)
  5. "Pretty good"          → 正面 (1)
  6. "Worst ever"           → 负面 (0)

超参数：H=16（嵌入维度）, K=2（二分类）, lr=0.03, num_buckets=5

训练 1000 轮后测试：
  输入："I hate this product"
  输出：正面=0.0012, 负面=0.9988 → ✅ 正确判断为负面
```

> ⚠️ **课程代码注意事项**：① 代码依赖 `hash()` 的进程内一致性——重启 Python 后同一文本的哈希值会变化，因此不能跨进程复用模型；② `num_buckets=5` 仅为演示目的——实际 FastText 论文使用 200 万到 1 亿桶来降低碰撞率；③ 6 条训练样本太少，模型在此规模上只能学会简单的词汇-情感关联，完全没有泛化能力。

> 🌐 **补充（Web Search / HuggingFace kenhktsui/llm-data-textbook-quality-fasttext-classifier-v2, 2024）**：有研究者专门训练了一个"教科书质量分类器"——用 FastText 判断一段网页文本的"教育价值"。该分类器在 CPU 上每秒可处理 2000+ 条文本，可直接用于预训练数据的在线过滤。关键发现：合成数据（如 Cosmopedia）的教育价值普遍高于真实网页数据；FineWeb 和 Dolma 是真实数据中教育价值最高的开源数据集。

---

## 四、DSIR：重要性重采样——"让数据分布对齐目标"

### 4.1 问题的提出

前面的方法（KenLM、FastText）都是在回答二元问题："这段文本好不好？"但真实的数据工程中有一个更精细的需求：

```
问题场景：
  我有一个庞大的通用语料库（如 The Pile，800GB 文本，分布 q）
  我有一段目标数据（如 PubMed 医学论文，100MB，分布 p）
  
  目标：从 800GB 中选出 10GB，使得这 10GB 的数据分布
        尽可能接近 p（医学领域），而非 q（通用领域）
  
  如果用 KenLM/FastText，我只能选出"高质量的文本"，
  但这些"高质量"可能包含小说、新闻、维基百科……
  没有一项是"分布对齐"的操作
```

**DSIR（Data Selection with Importance Resampling）**正是为解决这个问题而设计的（Xie et al., NeurIPS 2023）。

### 4.2 核心思想：重要性重采样

DSIR 的数学框架来自统计学中的**重要性采样**（Importance Sampling）：

```
标准重要性采样：
  想从分布 p 中采样，但只能从分布 q 中采样
  → 每采一个样本，赋予权重 w = p(样本) / q(样本)
  → 权重高的样本 = 在 p 中更常见、在 q 中更稀有 = "更像目标分布"

DSIR 的做法：
  ① 在"Bag of hashed n-grams"特征空间中估计 p 和 q
  ② 计算每个文档的 w_i = p(x_i) / q(x_i)
  ③ 按权重 w_i 做重要性重采样 → 选出 K 个文档
```

### 4.3 完整工作流程

```
┌─────────────────────────────────────────────────┐
│                   DSIR 完整流程                    │
├─────────────────────────────────────────────────┤
│                                                 │
│  Step 1: 特征化（Featurization）                  │
│    原始文本 → n-gram 分词 → 哈希化 → 稀疏特征向量    │
│    例如："The patient showed..."                │
│    → ["the", "patient", "showed",               │
│       "the patient", "patient showed"]          │
│    → hash → {桶12: 1, 桶45: 2, 桶89: 1, ...}    │
│                                                 │
│  Step 2: 估计分布                                │
│    对原始数据（q）和目标数据（p），各自统计          │
│    每个哈希桶中的 n-gram 出现频率                   │
│                                                 │
│  Step 3: 计算重要性权重                            │
│    对每个文档 x_i:                               │
│      w_i = p_hat(x_i) / q_hat(x_i)              │
│    （p_hat 和 q_hat 是从 n-gram 特征估计的代理分布） │
│                                                 │
│  Step 4: 重要性重采样                              │
│    按权重 w_i 从原始数据中采样 K 个文档              │
│    → 选中数据的分布 ≈ 目标分布 p                    │
│                                                 │
└─────────────────────────────────────────────────┘
```

### 4.4 为什么"哈希 n-gram 特征"就够了？

你可能会疑惑：n-gram 特征丢失了词序信息，能准确表示文档的分布特征吗？

**直觉解释**：一篇医学论文和一篇影评在**词汇使用上**几乎是完全不同的两套体系：

| 特征 | 医学论文 | 影评 |
|------|---------|------|
| 高频 n-gram | "the patient", "randomized trial", "statistically significant" | "the film", "great performance", "highly recommend" |
| 罕见但在对立方常见的 n-gram | "blockbuster", "Oscar nomination" | "placebo-controlled", "95% confidence interval" |

这些 n-gram 模式已经足以区分两类文档。DSIR 论文的测度指标——**KL Reduction**——验证了这一点：

```
KL Reduction：衡量"经过 DSIR 选择的数据"与"目标数据"在特征空间中的接近程度
  - KL Reduction 越高 → 选出的数据越接近目标分布
  - 实验证明 KL Reduction 与下游任务准确率高度相关
```

> 🌐 **补充（Web Search / DSIR 论文, NeurIPS 2023）**：实验显示，DSIR 在 8 个目标分布上（包括 PubMed、法律文件、GitHub 代码等）的表现**与专家手工筛选相当**。在 The Pile（800GB）上选择 100M 文档仅需 **4.5 小时**，全程在 CPU 上完成。

### 4.5 三种方法的选择决策树

```
你需要在哪个阶段筛选数据？
│
├── 第一轮粗筛：去掉明显噪声（HTML 标签、乱码）
│   → 启发式规则过滤（模块 1）
│
├── 第二轮精筛：去掉"看起来正常但质量差"的文本
│   → 困惑度过滤：KenLM / CCNet（模块 1）
│   → 或分类器：FastText（本模块）
│
├── 需要按类别筛选（如只要教科书级别的内容）
│   → FastText 分类器（本模块）
│     - 优点：CPU 上 2000+ 条/秒，可在线使用
│     - 需要：标注好的训练数据来训练分类器
│
└── 需要筛选出与目标领域分布接近的数据
    → DSIR 重要性重采样（本模块）
      - 优点：自动对齐分布，无需人工标注
      - 需要：有代表性的目标数据样本（几百 MB 即可）
```

> 💡 **注意**：这三种方法不是互斥的——实际的数据 pipeline（如 FineWeb）通常将它们**串行使用**：先启发式过滤 → 再 KenLM 困惑度过滤 → 再用 FastText 分类 → 最后去重（模块 3）。

---

## 🧠 本模块问题

请在下方回答以下问题后，输入 `提交作业` 提交。

**Q1**：FastText 使用 `embedding.mean(dim=1)` 将所有 n-gram 的嵌入向量求平均来得到文档表示。这个操作**丢失了所有词序信息**（"dog bites man" 和 "man bites dog" 会得到完全相同的向量）。为什么在数据筛选任务中，这个"缺陷"是可以接受的？什么情况下它**不能**被接受？

**Q2**：KenLM（5-gram 统计模型）和 GPT-2（Transformer）都可以计算文本的困惑度。为什么 CCNet 和 Dolma 等数据 pipeline 选择用 KenLM 而不是更"聪明"的 GPT-2 来做大规模数据过滤？请从**速度、内存、适用性**三个维度分析。

**Q3**：DSIR 的"重要性权重"公式是 w_i = p(x_i) / q(x_i)。如果一个文档满足以下情况，它的权重会是高还是低？请逐一分析并解释直觉含义：
- (a) 文档在医学论文（目标 p）中非常常见，在通用语料（原始 q）中也非常常见
- (b) 文档在医学论文中非常常见，但在通用语料中很罕见
- (c) 文档在医学论文中很罕见，但在通用语料中非常常见

---

<!-- 学习者作答区（请在此处填写你的答案） -->

**A1**：

FastText 是一个轻量的文档内容分类模型，用于对文档内容分类。
因此相比较内容中 词序的信息，文档中是否包含高频词以及文档中词向量的分布应该更能表征文档的类别。
而这里用分当中embedding.mean(dim=1)是一种快速评估平均词向量的方法，来粗略表征整个文档的嵌入向量分布，在速度和精度之间的平衡。
因此这个缺陷是可以接受的。

当需要评估语句的连贯性、文档语义的正确性时，词序信息时重要的，因为一个语句的语法、语义需要通过词序来表达。


**A2**：

KenLM 相比较 GPT2 来说，首先是 速度
KenLM 可以在 CPU 上快速运行，并且因为是查表计算，速度非常快。而 GPT2 是 transformer decode 架构，运行在 GPU 速度有限。
其次，KenLM 模型是一个词频表，存储容量占用有限；而 GPT2 是个 以 GB 为单位的模型，会占用更多显存空间。
从适用性来考虑，GPT2 模型是 transformer decode 架构，模型内是黑盒的，可解释性差；而 KenLM 是 根据词频来计算的，有据可依，并且，因为 KenLM 不需要复杂的 矩阵运算，因此对设备要求相比 GPT2 要低。


**A3**：

DSIR 重要性公式 w_i = p(x_i) / q(x_i)
即 目标分布 p(x)，采样分布 q(x)，希望 q 能跟 p 近似/相近

a) 权重较低, 因为 文档在医学论文 中场景，因此 p(x_i) 高，在通用预料中也常见, q(x_i) 也较高，所以 w_i 较低
b) 权重较高，文档在医学论文中非常常见 p(x_i) 高，但在通用语料中很罕见 q(x_i) 低
c) 权重较低，文档在医学论文中很罕见 p(x_i) 低，但在通用语料中非常常见 q(x_i) 也低

---

<!-- 教师批改区（提交作业后由导师填写，请勿手动修改） -->

### 📝 批改结果

**Q1 批改**：正确抓住了核心——文档级分类任务中，词汇的统计特征（高频词、词分布）比词序更重要，mean pooling 是"速度-精度"的工程权衡。两个可改进的点：① "词向量分布"这个说法不够精确——mean 只保留了"中心位置"，丢掉了方差、众数等分布信息，但文档分类中中心位置确实已经足够；② 可以补充一个具体例子来说明为什么词序丢失在数据筛选场景下不致命——比如"clinical trial patients"和"patients clinical trial"在 n-gram 层面共享同样的 1-gram 集合（{clinical, trial, patients}），两者都会被正确识别为"医学相关"。这种局部词序颠倒在大文档层面（几百个 n-gram）会被统计特征淹没。 — 得分：**7/10**

<details>
<summary>📖 Q1 参考答案</summary>

**为什么在数据筛选任务中可以接受？**

根本原因是文档级分类依赖的是**词汇统计特征**而非**句法结构**。一个文档是否属于"医学领域"，决定性信号来自 n-gram 的频率分布：

- 医学文档中 "patient"、"clinical"、"treatment" 等词及其组合高频出现
- 不管这些词以什么顺序排列，它们的统计显著性不会改变

从信息论角度：一条包含 500 个 n-gram 的文档，词序信息对应的是这 500 个 n-gram 的排列方式（500! 种可能），而分类需要的信息只是"这些 n-gram 的整体统计倾向"（一个 500 维的线性分类器权重）。后者对词序完全不敏感。

**什么情况下不能接受？**

任何依赖**局部句法关系**或**语义组合性**的任务：

| 任务 | 为什么词序重要 | 例子 |
|------|--------------|------|
| 情感分析（短文本） | "not good" vs "good" 的一词之差 | "I don't hate this" = 正面，但平均后可能判为负面 |
| 命名实体关系 | "Apple bought Tesla" vs "Tesla bought Apple" | 主动-被动关系靠词序表达 |
| 语法纠错 | 需要判断"主谓宾"是否合法 | 句子结构是判断依据本身 |
| 事实性判断 | "Paris is in France" vs "France is in Paris" | 两个句子 n-gram 集合完全一样，但一个对、一个错 |

</details>

---

**Q2 批改**：三个维度的方向都正确。速度维度点出了 CPU vs GPU 和查表 vs 矩阵运算的关键差异；内存维度抓住了"词频表"vs"GB 级模型"的数量级区别；适用性维度提到了可解释性和硬件门槛。可以加强的点：① "查表计算"可以更精确——KenLM 实际是遍历 Trie 树做概率查询和 backoff，比简单的 hash 查表复杂但有专门的压缩和优化；② 可以补充具体数量级（KenLM 数万条/秒 vs GPT-2 数百条/秒，差距在 1-2 个数量级）。 — 得分：**7/10**

<details>
<summary>📖 Q2 参考答案</summary>

**速度维度**

| | KenLM (5-gram) | GPT-2 (124M 参数) |
|---|---|---|
| 单次推理操作 | 在压缩 Trie 上查询 5-gram 条件概率 + backoff | 12 层 Transformer 的矩阵乘法和注意力计算 |
| 计算量 | ~5 次内存访问 + 取对数 | ~10^8 FLOPs |
| 吞吐量 | CPU 单核 10,000-50,000 条/秒 | GPU 单卡 100-500 条/秒 |
| 处理 1B 条文本 | ~5.5 小时（单 CPU 核） | ~23 天（单 GPU） |

处理百亿级文档时，这个速度差距直接决定"能不能做"而非"做得有多好"。

**内存维度**

- KenLM：训练好的 5-gram 模型存储为压缩的 ARPA 格式，英语 Wikipedia 语料训练出的模型约 200-500MB，加载后直接 mmap 到内存
- GPT-2（最小版 124M）：FP16 权重约 250MB，但实际推理需要额外的 KV Cache 空间，以及 PyTorch runtime 开销，内存占用 ≥ 1GB

**适用性维度**

数据过滤是一个"信号"（signal）任务而非"推理"（reasoning）任务：

- 你只需要一个粗粒度的质量评分（"这文本看起来靠谱吗？"），不需要深层的语义理解
- 5-gram 对局部流利度（fluency）的判断与神经网络模型的判断高度一致——因为不流利的文本（重复、乱序、语法错误）在 n-gram 级别就会表现出低概率
- KenLM 完全确定性的白盒：给定同一输入永远返回同一 PPL，不存在采样随机性

**核心决策逻辑**：CCNet/Dolma 面对的是"用几百台 CPU 机器在几天内过滤万亿 tokens"这个工程约束。在这个约束下，KenLM 是唯一能满足吞吐量要求且质量信号足够可靠的选择。GPT-2 更适合"我已经过滤到小规模了，需要对剩余数据做精细语义排序"的后续环节。

</details>

---

**Q3 批改**：(a) 和 (b) 的数学判断正确，但 (c) 中存在一个表述错误——题目写的是"在通用语料中非常常见"，即 q(x_i) 应该是**高**，你写成了"q(x_i) 也低"，虽然最终结论"权重较低"是对的，但中间推理与题目条件矛盾。此外，三个小问都只给出了数值大小结论，缺少题目要求的"直觉含义"分析（weight 高/低分别意味着什么、对应什么文档、DSIR 会优先选还是优先丢弃）。 — 得分：**5/10**

<details>
<summary>📖 Q3 参考答案</summary>

**核心公式**：w_i = p(x_i) / q(x_i)，其中 p 是目标分布（如医学论文），q 是原始分布（如通用语料）。

**(a) p 高, q 高 → w ≈ 1（中等权重）**

> 例：一篇包含"the"、"is"、"and"等高频功能词的普通段落

这类文档在两类数据中都大量存在，不具备区分力。DSIR 不会优先选它（权重不突出），也不会故意丢弃它——它只是"平凡"的文档，选中与否对分布对齐影响不大。从统计学角度：w ≈ 1 意味着该文档既不比目标更像目标（w > 1），也不比通用更像通用（w < 1）。

**(b) p 高, q 高 → w >> 1（高权重，DSIR 优先选择）**

> 例：一篇包含"randomized double-blind placebo-controlled trial"的文档，是医学论文的标准措辞，但在通用网页中极少出现

这类文档是医学领域的"特征文档"——它在目标分布中非常典型，在原始分布中稀有。w 越高，说明"这个文档在通用中很罕见，但在医学中很常见"→ 选中它就能把选中的数据分布往目标方向推进一大步。

**(c) p 低, q 高 → w << 1（极低权重，DSIR 会丢弃）**

> 例：一篇体育赛事报道（"The Lakers defeated the Celtics 108-102..."），在通用网页中极为常见，但在医学论文中几乎不会出现

这类文档是"通用噪音"——它在目标分布中几乎不存在。w 接近于 0 意味着 DSIR 几乎不会选它。这也正是 DSIR 区别于无差别采样的关键：不是随便从 q 中随机选，而是有选择地"捞"出那些 p-典型的样本，过滤掉 q-典型的但 p-不典型的噪音。

**数值推演**（假设具体数字让直觉更清晰）：

| 场景 | p(x_i) | q(x_i) | w_i | DSIR 行为 |
|------|--------|--------|-----|----------|
| (a) 通用功能词 | 0.005 | 0.005 | 1.0 | 随机概率选中 |
| (b) 医学术语 | 0.003 | 0.0001 | **30.0** | **高概率选中** |
| (c) 体育报道 | 0.00001 | 0.003 | **0.0033** | **几乎不选** |

</details>

---

**综合评价**：Q1 和 Q2 概念理解扎实，方向判断准确；Q3 的权重计算逻辑基本对但有一个表述性错误，且三个子问题都缺少题目要求的"直觉含义"解读。共性建议：当题目明确要求"逐一分析并解释直觉含义"时，每个小问至少需要回答**数值方向 + 这是什么类型的文档 + DSIR 会如何处理**这三层。可以继续下一模块。

**批改时间**：2026-05-27
