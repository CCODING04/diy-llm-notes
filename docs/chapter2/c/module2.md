# 第 2 章：分词器 — 模块 2：四种分词器原理与代码对比

> 📍 学习进度：第 2 章，第 2 / 3 模块
> 📅 生成时间：2026-04-13

---

## 学习目标

- 掌握字符级、字节级、词级、BPE 四种分词器的基本原理
- 能通过代码实现理解各算法核心逻辑
- 理解四种分词器的词表大小、序列长度、OOV 程度的取舍

---

## 核心内容

### 四种分词器横向概览

| 分词器类型 | 粒度 | 词表大小 | OOV 程度 | 序列长度 | 代表模型 |
|-----------|------|---------|---------|---------|---------|
| 字符级 | 极细（单字符） | 小（~5k） | **无** | 非常长 | Char-RNN |
| 字节级 | 最细（单字节） | 固定 256 | **无** | 长 | GPT-2 |
| 词级 | 粗（完整词） | 极大（>100k） | 严重 | 短 | Word2Vec、GloVe |
| **BPE** | **中（自适应）** | **适中（30k-100k）** | **极少** | **适中** | **GPT-4、LLaMA 3** |

---

### 1. 字符级分词器

**核心思想**：把文本拆成最小语义单元——单个字符。

```python
class CharacterTokenizer:
    def encode(self, text):
        # ord() 返回字符的 Unicode 码点（整数）
        return [ord(ch) for ch in text]

    def decode(self, indices):
        # chr() 将 Unicode 码点转回字符
        return ''.join([chr(i) for i in indices])
```

以 `"hi，很好的，terrific！🐋"` 为例：

```
字符：h  i  ，  很  好  的  ，  t  e  r  r  i  f  i  c  ！ 🐋
ID：  104 105 65292 24456 22909 30340 65292 116 101...
```

**词表构建**：词表大小取决于训练语料中出现的不同字符数（Unicode 范围），中文约几千字，英文只需 26+符号。

**优点**：词表极小，无 OOV（任何词都由已知字符组成）。
**缺点**：序列长度暴增（一个汉字 = 一个 token，但英文 "terrific" 有 8 个字符），上下文窗口消耗快；单个字符语义信息少。

---

### 2. 字节级分词器

**核心思想**：不依赖"字符"概念，直接操作 UTF-8 字节序列。

```python
class ByteTokenizer:
    def __init__(self):
        self.vocab_size = 256  # 固定 256 个字节

    def encode(self, text: str):
        # text.encode("utf-8") → bytes 对象
        # list(bytes) → 每个字节变成 0-255 的整数
        return list(text.encode("utf-8"))

    def decode(self, indices):
        return bytes(indices).decode("utf-8")
```

以 `"Hello, 🌍!"` 为例：

```
字符：H   e   l   l   o   ,  [空格]  🌍   !   [全角感叹号]
字节：48  69  6C  6C  6F  2C 20      F0 9F 8C 8D  EF BC 81
                                                              ↑
                                                        🌍=3字节，！=3字节
```

**压缩率恒等于 1**：token 数 = UTF-8 字节数，所以 `byte/token = N/N = 1`，**没有任何压缩能力**。

> 💡 **补充（Context7 / HuggingFace tokenizers）**
>
> 现代 LLM 实际上用的是 **BBPE（Byte-level BPE）**，而非纯字节级。BBPE = 字节级切分 + BPE 合并，是 GPT-2 之后的标准做法：
>
> ```python
> tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
> tokenizer.decoder = decoders.ByteLevel()
> ```
> 这样先用字节覆盖所有字符（无 OOV），再用 BPE 把高频字节对合并（获得压缩）。

---

### 3. 词级分词器

**核心思想**：按空格（英文）或分词算法（中文）切分，每个"词"作为一个 token。

```python
import regex

TOKENIZER_REGEX = r"\p{L}+|\p{N}+|[^\p{L}\p{N}\s]+|\s+"
# 连续字母 | 连续数字 | 非字母数字空白字符 | 空白

class WordTokenizer:
    def build_vocab(self, texts):
        vocab = set()
        for text in texts:
            # regex.findall 按正则切分
            segments = regex.findall(self.pattern, text)
            vocab.update(segments)
        self.word2id = {w: i for i, w in enumerate(sorted(vocab))}

    def encode(self, text):
        segments = regex.findall(self.pattern, text)
        return [self.word2id.get(seg, -1) for seg in segments]
        # -1 代表 UNK（词表外）
```

以 `"It's so supercalifragilisticexpialidocious!👋👋"` 为例：

```
基础正则（\w+|.）：
  ['It', "'", 's', ' ', 'so', ' ', 'supercalifragilisticexpialidocious', '!', '👋', '👋']
  → emoji 和标点各自独立成 token（. 匹配单个字符）

DeepSeek风格（\p{L}+|\p{N}+|[^\p{L}\p{N}\s]+|\s+）：
  ['It', "'", 's', ' ', 'so', ' ', 'supercalifragilisticexpialidocious', '!👋👋']
  → [^\p{L}\p{N}\s]+ 把 ! 和 👋👋 连续非字母数字空白字符合并为一个 token
```

**最大问题**：`"look"`, `"looks"`, `"looked"`, `"looking"` 被视为 4 个完全不同的 token，词表爆炸；同时无法处理未登录词（人名、专业术语等）。

---

### 4. BPE 分词器

**核心思想**：在字符级基础上，迭代合并高频相邻字符对——既避免 OOV，又获得子词压缩。

以下是一个真实的 BPE 训练过程（来自课程代码）：

```
训练语料：["这只猫🐈很可爱", "the quick brown fox jumps over the lazy 🐕‍🦺"]
合并次数：num_merges=20

BPE 按频率贪心合并的结果（按顺序）：
  第1轮：(' ', '</w>') 频次最高 → 合并    空格+词尾符
  第2轮：('t', 'h') → 合并为 'th'       th 在 the、this、that 中高频出现
  第3轮：('th', 'e') → 合并为 'the'
  第4轮：('the', '</w>') → 合并         单独的 "the" 完整成词
  第5轮：('这', '只') → 合并为 '这只'    中文高频组合
  ...
  第12轮：('很', '可') → 合并为 '很可'
  第13轮：('很可', '爱') → 合并为 '很可爱'

最终 merge 表（前13项）：
  [(' ', '</w>'), ('t', 'h'), ('th', 'e'), ('the', '</w>'),
   ('这', '只'), ('这只', '猫'), ('🐈', '</w>'),
   ('很', '可'), ('很可', '爱'), ('很可爱', '</w>'), ...]
```

编码新文本 `"敏捷的棕色狐狸🦊"`（训练时未见过）：

```
原始字符：敏 捷 的 棕 色 狐 狸 🦊

按 merge 顺序逐一检查：
  (' ', '</w>')  → 不匹配（无空格）
  ('t', 'h')     → 不匹配（无 t/h）
  ...（前面的英文 merge 全不匹配）
  最后在每个预分词片段末尾加上 </w>

编码结果：['敏', '捷', '的', '棕', '色', '狐', '狸', '</w>', '🦊', '</w>']
          ↑ 这些字符在训练语料中没有形成高频相邻对，所以未被合并
```

**关键理解**：BPE 只合并训练语料中**实际高频出现的相邻对**。新文本中如果字符组合不在 merge 表中，就保持原样不合并——这就是为什么 BPE 不会产生 OOV（最坏情况退化为字符级）。


**BPE 训练过程（代码层面）**：

```python
def train_bpe(texts, num_merges=50):
    # 第一步：按预分词规则把每个词切成字符序列，末尾加 </w>
    vocab = Counter()
    for text in texts:
        tokens = regex.findall(DEEPSEEK_REGEX, text)  # 预分词
        for token in tokens:
            chars = tuple(token) + ('</w>',)          # 字符级 + 结束符
            vocab[chars] += 1

    merges = []
    for _ in range(num_merges):
        # 第二步：统计所有相邻字符对的出现频率
        pairs = Counter()
        for word, freq in vocab.items():
            for i in range(len(word) - 1):
                pairs[(word[i], word[i+1])] += freq

        if not pairs:
            break

        # 第三步：贪心选择频率最高的字符对，合并
        best_pair = max(pairs, key=pairs.get)
        merges.append(best_pair)

        # 第四步：更新词表，把所有该字符对合并
        new_vocab = {}
        for word, freq in vocab.items():
            merged = []
            i = 0
            while i < len(word):
                if i < len(word)-1 and (word[i], word[i+1]) == best_pair:
                    merged.append(word[i] + word[i+1])
                    i += 2
                else:
                    merged.append(word[i])
                    i += 1
            new_vocab[tuple(merged)] = freq
        vocab = new_vocab

    return merges
```

**`</w>` 结束符的作用**：告诉 BPE 哪里是单词边界，防止跨词合并。解码时去掉 `</w>` 即可还原原文。

> 🌐 **补充（Web Search / CSDN 2024）**
>
> BPE 的贪心合并策略存在一个**实践问题**：最高频 pair 未必是语义上最优的合并。Google 的 WordPiece 和 Sony 的 Unigram 通过引入**语言模型评分**（pair 的得分 = P(A,B) / P(A)×P(B)）来选择更合理的合并，而非纯频率优先。这种评分机制使 WordPiece 生成的词表在下游任务中通常表现更好，但实现也更复杂。
>
> GPT 系列沿用 BPE 而非 WordPiece 的原因：GPT 是**自回归语言模型"，生成场景下 BPE 的贪心策略足够高效；而 BERT 是**掩码语言模型"，对词表质量更敏感。

---

### 四种分词器实测对比

运行 `BPE_character_byte_level_word_segmentation_Comparison.py` 的结果：

```
输入："Hello, 🌍! 你好!"

原始字节长度：20

token 数量对比：
  字节级：20 个 token（1:1，无压缩）
  字符级：13 个 token（压缩率 1.54）
  BPE：   11 个 token（压缩率 1.82）← 最接近真实 LLM tokenizer

=== 压缩率(byte/token) ===
字节级: 1.00
字符级: 1.54  ← 中文"你好"各3字节→各1字符，压缩明显
BPE:    1.82  ← "你好"等高频词已合并为少数字节对
```

这验证了为什么 **BPE 是现代 LLM 的主流选择**——它同时具备：
1. **无 OOV**（底层仍是字符/字节）
2. **合理词表大小**（30k-100k）
3. **良好的压缩率**（token 序列短，上下文利用率高）

---

## 代码解析

`BPE_character_byte_level_word_segmentation_Comparison.py` 中的压缩率计算函数：

```python
def get_compression_ratio(text: str, token_len: int):
    input_byte_len = len(text.encode("utf-8"))
    return input_byte_len / token_len if token_len > 0 else 1
```

- `len(text.encode("utf-8"))` = 原始文本的字节数
- `token_len` = 分词后的 token 数量
- **压缩率 > 1** = token 表示比原始字节更紧凑（节省了 token 数量）

```
字节级：20字节 / 20token = 1.00  ← 无压缩
字符级：20字节 / 13token = 1.54  ← 每个 token 平均表示 1.54 字节
BPE：   20字节 / 11token = 1.82  ← 每个 token 平均表示 1.82 字节
```

---

## 🧠 本模块问题

请在下方作答区填写答案，完成后输入 `提交作业` 提交。

**Q1**：BPE 训练时，`</w>`（单词结束符）的作用是什么？如果去掉 `</w>`，会出现什么问题？请用具体例子说明。

**Q2**：字节级分词器的压缩率为什么恒等于 1？既然没有任何压缩效果，为什么现代 LLM 还用它作为 BPE 的基础（BBPE）？

**Q3**：BPE 和 WordPiece 都会生成子词，但合并策略不同。简述两者合并策略的核心差异，以及为什么 GPT 系列用 BPE 而 BERT 用 WordPiece？

---

<!-- 学习者作答区（请在此处填写你的答案） -->

**A1**：防止跨界合并。 比如 She is beautiful. 如果去掉 </w> 会当作一整个 sheisbeautiful， 整个 vocab 会当作一个巨大的字符串来进行 BPE，从而使得词级被丢失，整个BPE相当于在一个字符串序列上整合出来的，没有词性作为基础，我猜测可能会影响 embedding 的表征能力。

**A2**：因为压缩率的比较就是根据字节数量作比较，而字节级分词器其实就是简单的将每个字符转为字节，显然 compression ratio 是 恒等于 1 的。
虽然没有任何压缩效果，但是 字节级分词器有个非常大的优势：不管新增任何词汇，它都是需要由字节构成的。因此也就不存在 OOV（out of vocabulary）情况出现，所以基于 字节级分词器的 BPE，同样也保持了这个特性，并且还有很高的压缩比。

**A3**：GPT 是一个自回归模型，BPE 本质是个贪心算法，因此使用 BPE 没有太多影响，而且会更方便、便捷。而 BERT 是个 掩码语言 模型，它对 上下文、语言窗口 等有更高的要求，需要 embedding 包含更多的词性信息？可能还有别的原因，但是我暂时没法理解，需要进一步的讲解。

---

<!-- 教师批改区（提交作业后由导师填写，请勿手动修改） -->

### 📝 批改结果

**Q1 批改**：✅ 核心答对了——`</w>` 防止跨词边界合并。你举的例子很直观：没有 `</w>` 时，BPE 会把所有字符拼接后当成一个大字符串处理，词边界信息丢失。补充一个更具体的后果：如果 `"the"` 和 `"is"` 之间没有 `</w>` 隔开，BPE 可能学到 `('e', 'i')` 这个跨词 pair，合并出 `"theis"` 这种无意义 token，浪费词表空间。`</w>` 的本质是**在 token 层面保留预分词的边界信息**，让 BPE 的统计只在词内进行。 — 得分：**9/10**

**Q2 批改**：✅ 完全正确。压缩率 = UTF-8 字节数 / token 数，字节级两者相等所以恒为 1。BBPE 的核心优势你也抓住了——字节级保证无 OOV，BPE 再在上面做高频合并获得压缩。这是"先保证覆盖，再追求效率"的设计思路。 — 得分：**10/10**

**Q3 批改**：✅ GPT 用 BPE 的原因理解正确——自回归模型生成时逐 token 输出，BPE 的贪心策略足够高效。BERT 部分你的直觉方向对了，但可以更精确：BERT 用 WordPiece 不是因为"上下文要求更高"，而是因为 **BERT 是 Google 开发的，WordPiece 也是 Google 的技术**（BERT 论文直接用了 Google 内部的 WordPiece 实现）。技术层面，WordPiece 用似然比评分（P(A,B)/P(A)×P(B)）选 merge，比 BPE 的纯频率统计更"理性"，在 MLM 任务中词表质量确实更好。但这个差异在 GPT 场景下不明显，所以 OpenAI 选择了更简单的 BPE。你说的"需要进一步讲解"部分——本质上就是 WordPiece 的评分公式 vs BPE 的频率统计，**没有更深层的原因了**。 — 得分：**8/10**

**综合评价**：Q2 满分，Q1 和 Q3 都抓住了核心。Q3 的"历史原因"（Google 生态）也是一个实际因素，但更重要的是 merge 选择策略的差异。总体掌握扎实，模块 2 的四种分词器对比理解到位。

**批改时间**：2026-04-15
