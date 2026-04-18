# Assignment 1 - Part 2：BPE 训练（从语料学习 vocab + merges）

> 📍 作业进度：Assignment 1，第 2 / 3 部分
> 📅 生成时间：2026-04-17
> 📎 原作业参考：[stanford-cs336/assignment1-basics](https://github.com/stanford-cs336/assignment1-basics)

---

## 目标与要求

从原始文本语料训练 BPE tokenizer，输出 vocab 和 merges。

### 原作业要求（摘自 adapters.py）

```python
def run_train_bpe(
    input_path: str | os.PathLike,     # 训练语料路径
    vocab_size: int,                     # 目标词表大小（含 special tokens）
    special_tokens: list[str],           # 特殊 token 列表
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Returns:
        vocab: dict[int, bytes]  — token ID → 字节内容
        merges: list[tuple[bytes, bytes]]  — 合并规则（有序）
    """
```

### 核心测试用例（来自 test_train_bpe.py）

1. **训练正确性**：在 `corpus.en` 上训练 vocab_size=500，merges 必须与参考实现完全一致
2. **训练速度**：在 `corpus.en` 上训练 vocab_size=500，必须在 **1.5 秒内**完成
3. **Special token 处理**：special tokens 加入词表但不参与 BPE 合并

---

## 实现步骤

### Step 1：读取语料 + 预分词

```python
import regex

GPT2_PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

# 注意：训练时的预分词正则和编码时可能不同
# GPT-2 训练用上面的 \p{L} 版本（支持 Unicode 字母），编码用 \w 版本
# 但 tiktoken 的 GPT-2 实现两者一致，用 \p{L} 版本

with open(input_path) as f:
    text = f.read()

# 预分词，统计每个词的出现频率
word_freqs = Counter()
for word in regex.findall(GPT2_PAT, text):
    word_freqs[word] += 1
```

### Step 2：初始化基础词表

将每个词转为 UTF-8 字节序列，再转为 GPT-2 Unicode 字符：

```python
byte_encoder = bytes_to_unicode()

# 每个词转为 tuple of unicode chars，用于后续合并
vocab = {}  # word_tuple → freq
for word, freq in word_freqs.items():
    word_bytes = word.encode("utf-8")
    word_chars = tuple(byte_encoder[b] for b in word_bytes)
    vocab[word_chars] = freq
```

初始词表 = 所有 256 个字节 + special tokens。

### Step 3：迭代合并

```python
merges = []
# 初始 vocab_size 减去 256（字节）和 special tokens 数量
num_merges = vocab_size - 256 - len(special_tokens)

for _ in range(num_merges):
    # 统计所有相邻 pair 的频率（加权）
    pairs = Counter()
    for word, freq in vocab.items():
        for i in range(len(word) - 1):
            pairs[(word[i], word[i+1])] += freq
    
    if not pairs:
        break
    
    # 找最高频 pair
    best = max(pairs, key=pairs.get)
    merges.append(best)
    
    # 合并所有词中的 best pair
    new_vocab = {}
    for word, freq in vocab.items():
        new_word = merge_pair(word, best)
        new_vocab[new_word] = freq
    vocab = new_vocab
```

### Step 4：merge_pair 函数

```python
def merge_pair(word: tuple, pair: tuple) -> tuple:
    """将 word 中所有 pair 合并"""
    new_word = []
    i = 0
    while i < len(word):
        if i < len(word) - 1 and word[i] == pair[0] and word[i+1] == pair[1]:
            new_word.append(pair[0] + pair[1])
            i += 2
        else:
            new_word.append(word[i])
            i += 1
    return tuple(new_word)
```

### Step 5：构建最终 vocab

```python
# 收集所有 token（从合并后的 vocab 中）
vocab_set = set()
for word in vocab:
    for token in word:
        vocab_set.add(token)

# 构建最终 vocab dict[int, bytes]
byte_decoder = {v: k for k, v in byte_encoder.items()}
final_vocab = {}

# 先加 special tokens
for i, st in enumerate(special_tokens):
    final_vocab[i] = st.encode("utf-8")

# 再加 256 个字节
offset = len(special_tokens)
for b in range(256):
    final_vocab[offset + b] = bytes([b])

# 最后加 merge 产生的 token
for i, (s1, s2) in enumerate(merges):
    merged = s1 + s2
    token_bytes = bytes([byte_decoder[c] for c in merged])
    final_vocab[offset + 256 + i] = token_bytes
```

### Step 6：merges 输出格式

merges 中的每个 pair 需要转为 `tuple[bytes, bytes]`：

```python
final_merges = []
for s1, s2 in merges:
    b1 = bytes([byte_decoder[c] for c in s1])
    b2 = bytes([byte_decoder[c] for c in s2])
    final_merges.append((b1, b2))
```

---

## 测试方法

```bash
cd homework/assignment1
PYTHONPATH=scripts python -c "
from train_bpe import run_train_bpe
import json, time

# 测试速度
start = time.time()
vocab, merges = run_train_bpe(
    input_path='tests/fixtures/corpus.en',
    vocab_size=500,
    special_tokens=['
具体来说，8、'],
)
elapsed = time.time() - start
print(f'Training time: {elapsed:.2f}s')
assert elapsed < 1.5, f'Too slow: {elapsed:.2f}s'

# 测试 merges 正确性
print(f'Vocab size: {len(vocab)}')
print(f'Merges count: {len(merges)}')
print(f'First 5 merges: {merges[:5]}')
assert len(vocab) == 500
"
```

---

## 难点与注意事项

1. **预分词正则用 `\p{L}` 而不是 `\w`**：GPT-2 训练时用的是 Unicode 字母属性 `\p{L}`，不是 `\w`（等价于 `[a-zA-Z0-9_]`）。这个差异会影响 merge 结果

2. **频率统计要加权**：每个 pair 的频率 = 出现次数 × 该词在语料中的频率。不是简单地数 pair 出现次数

3. **性能优化**：朴素实现每轮都遍历所有词统计所有 pairs，在词表大时很慢。参考测试要求 1.5 秒内完成，需要：
   - 用 Counter 统计频率
   - 每轮只更新受影响的词的 pair 统计
   - 或使用增量更新策略

4. **special tokens 不参与合并**：special tokens 在初始化时加入词表，但它们的内部字节不应被 BPE 统计到。实际上 special tokens 的处理更简单——在预分词阶段先按 special tokens 拆分文本，special token 之间的部分才做 BPE 训练

5. **vocab 编号顺序**：special tokens 在前（0, 1, ...），然后是 256 个字节，最后是 merge 产生的 token。但参考实现只检查 vocab 的 key 和 value 集合是否匹配，不要求顺序一致

6. **空行/空字符串**：语料中可能有空行，预分词结果可能为空列表，需要跳过

---

## 代码框架

在 `homework/assignment1/scripts/train_bpe.py` 中实现：

```python
import os
import regex
from collections import Counter


def bytes_to_unicode():
    """GPT-2 字节到 Unicode 映射"""
    # ... 复用 Part 1 中的实现
    pass


def get_pairs(word):
    """获取词中所有相邻 pair"""
    pairs = set()
    for i in range(len(word) - 1):
        pairs.add((word[i], word[i + 1]))
    return pairs


def merge_pair(word, pair):
    """合并词中的指定 pair"""
    new_word = []
    i = 0
    while i < len(word):
        if i < len(word) - 1 and word[i] == pair[0] and word[i + 1] == pair[1]:
            new_word.append(pair[0] + pair[1])
            i += 2
        else:
            new_word.append(word[i])
            i += 1
    return tuple(new_word)


def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    # TODO: 实现完整的 BPE 训练流程
    raise NotImplementedError
```

完成后输入 `提交作业` 运行测试验证。
