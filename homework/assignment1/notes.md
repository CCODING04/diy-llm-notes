# Assignment 1 — 作业学习笔记

---

## Tutorial Part 1: Tokenizer 类实现 — QA 记录
> 📅 2026-04-16 ~ 2026-04-17

### 课程间隙临时 QA

**Q**：special token 应该在 regex 分词前还是分词后划分？（我感觉应该是 regex 前，因为正则有可能将 special token 切分掉）

**A**：必须在 regex **之前**划分。流程是：原始文本 → 按 special tokens 拆分为多段 → 每段分别走 regex 预分词 + BPE encode。因为 regex 会把 `


具体来说，8、` 这种 special token 拆成 `<`, `|`, `endoftext` 等片段，破坏完整性。拆分可用 `regex.split` 带捕获组，分隔符会保留在结果中。具体实现：
```python
if special_tokens:
    pattern = "(" + "|".join(regex.escape(t) for t in special_tokens) + ")"
    parts = regex.split(pattern, text)
```

---

**Q**：merge 步骤应该如何实现比较高效，逐个对比感觉非常低效

**A**：正确做法是**反过来**——从当前 token 序列中找优先级最高（rank 最小）的 pair，合并它，重复直到没有可合并的 pair。

核心思路：
```python
self.bpe_ranks = {pair: i for i, pair in enumerate(merges_str_pairs)}

while len(word) >= 2:
    pairs = {(word[i], word[i+1]) for i in range(len(word)-1)}
    ranked = [(self.bpe_ranks[p], p) for p in pairs if p in self.bpe_ranks]
    if not ranked:
        break
    _, best = min(ranked)
    # 合并所有出现的 best pair
    word = merge_all(word, best)
```

复杂度对比：逐条 merge 遍历是 O(num_merges × word_len)，找最小 rank pair 是 O(word_len²)。实际中 word_len 很少超过 20，所以快得多。

---

**Q**：merges 格式应该是什么样子的

**A**：接口定义是 `list[tuple[bytes, bytes]]`，每个 merge 是两个 bytes 对象的元组。但在 `__init__` 中需要转换成字符串来做 BPE 合并。关键是 merges 中的 bytes 不是 UTF-8 文本，而是原始字节，需要用 `bytes_to_unicode` 映射表逐字节转换：

```python
# vocab: bytes → GPT-2 unicode 字符串
for token_id, token_bytes in vocab.items():
    token_str = "".join(byte_encoder[b] for b in token_bytes)
    self.encoder[token_str] = token_id

# merges: bytes → GPT-2 unicode 字符串 pair
for i, (b1, b2) in enumerate(merges):
    s1 = "".join(byte_encoder[b] for b in b1)
    s2 = "".join(byte_encoder[b] for b in b2)
    self.bpe_ranks[(s1, s2)] = i
```

---

**Q**：`ord(b'\xe4') == 228`, `chr(228) = 'ä'`, 那么 228 如何返回成 `b'\xe4'`？这跟 encode/decode 什么关系

**A**：

```
b'\xe4'  ←→  228 (整数)  ←→  'ä' (str, Unicode 码点 U+00E4)

# 反向操作
bytes([228])           # → b'\xe4'
'ä'.encode('latin1')   # → b'\xe4'  ✅ latin1: 码点 0-255 → 字节 0-255 一一对应
'ä'.encode('utf-8')    # → b'\xc3\xa4'  ❌ UTF-8 下 'ä' 要 2 个字节
```

核心关系：
- `encode`: str → bytes（字符 → 字节）
- `decode`: bytes → str（字节 → 字符）
- latin1 的特殊性：单字节编码，码点 0-255 和字节 0-255 **一一对应**，`chr(x).encode('latin1') == bytes([x])` 对 0-255 恒成立

---

## Tutorial Part 1：作业批改记录

### 第 1 次提交（2026-04-16）

**得分：需修改后重提**

#### 问题 1（致命）：`ord(b)` 在 bytes 迭代中会崩溃

Python 3 中迭代 `bytes` 对象得到的是 `int`，不是单字节 `bytes`：

```python
>>> for b in b'\xe4\xbd':
...     print(type(b), b)
<class 'int'> 228
<class 'int'> 189
```

所以 `ord(b)` 对 int 调用会直接抛 `TypeError`。涉及位置：
- 第 34 行：`self.bytes2unicode.get(ord(b), chr(ord(b)))` in vocab 构建
- 第 41、44 行：merges 构建中同样的 `ord(b)`
- 第 82 行：encode 中的 `ord(b)`

**修复**：`bytes` 迭代得到的 `b` 已经是 `int`，直接用 `self.bytes2unicode[b]`，不需要 `ord()`。

#### 问题 2（关键）：vocab 和 merges 的 bytes_to_unicode 转换方式错误

`vocab` 是 `dict[int, bytes]`，`k` 是 int（token ID），`v` 是 bytes。`bytes2unicode` 映射的是 int(0-255) → str，不是 token ID → str。对于 bytes 值需要**逐字节**转换：

```python
# 错误写法（原代码）
self.vocab = {
    v: self.bytes2unicode.get(k, chr(k)) for k, v in vocab.items()
}

# 正确写法
self.encoder = {}
for token_id, token_bytes in vocab.items():
    token_str = "".join(self.bytes2unicode[b] for b in token_bytes)
    self.encoder[token_str] = token_id
```

同理 merges 的 k1, k2 是 bytes 对象，可能包含多个字节，也需要逐字节转换。

#### 问题 3（关键）：special token 查找方式不匹配

`self.vocab_reverse[part.encode()]` — `self.vocab` 映射 int → str（token_id → token_str），用 bytes 做 key 查不到。Special token 需要在初始化时单独建查找表。

#### 问题 4（性能）：apply_merges 逐条遍历

当前实现遍历全部 merges（可能 50000+ 条），对每个词都是 O(num_merges × word_len)。虽然结果正确，但会非常慢。建议改用 rank-based 方法。

#### 问题 5（功能）：encode_iterable 不是流式的

`encode_iterable` 接收 `Iterable[str]`（可能是文件对象），但当前实现直接对整个输入做操作，没有流式处理。

#### 正确的部分

- `bytes_to_unicode` 实现正确 ✅
- 预分词正则 GPT2_PAT 正确 ✅
- special tokens 在 regex 前拆分的思路正确 ✅
- decode 的基本逻辑正确 ✅
- `regex.split` 带捕获组保留分隔符正确 ✅

### 第 2 次提交（2026-04-17）

**得分：需修改后重提**

#### 问题 1：`ord(b)` 已修复 ✅

`self.bytes2unicode.get(b, chr(b))` 正确使用了 `b`（int 类型）。

#### 问题 2：apply_merges 中 `self.merge_ranks` 不存在

第 104 行引用了 `self.merges_ranks`（多了个 s），但 `__init__` 中定义的是 `self.merge_ranks`，导致 `AttributeError`。

#### 问题 3：apply_merges 中 `token_chars` vs `word` 变量混淆

第 98 行设了 `word = list(token_chars)`，但第 100 行用 `token_chars` 构造 pairs。第二次循环时 `token_chars` 还是原始值，`word` 的更新被忽略。应全部用 `word`。

#### 问题 4：encode_iterable 输出类型错误

```python
yield self.encode(part)  # 返回 list[int]，不是单个 int
```
应改为 `yield from self.encode(part)` 或用 `"".join(text)` 后逐个 yield。

### 第 3 次提交（2026-04-17）

**得分：通过 ✅**

仅剩 `self.merges_ranks` → `self.merge_ranks` 拼写错误，修复后全部 10 个测试通过：
- 空字符串 / 单字符 / ASCII / Unicode roundtrip ✅
- tiktoken GPT-2 编码完全一致 ✅
- special token 保留和匹配 ✅
- encode_iterable 文件流式处理 ✅
