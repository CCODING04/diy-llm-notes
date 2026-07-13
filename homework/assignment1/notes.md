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

---

## Tutorial Part 4-5: Transformer 基础算子与注意力机制 — QA 记录
> 📅 2026-07-08

### 作业批改记录

#### Part 4 提交

**得分：需修改后重提**

##### 问题 1：Embedding 参数名 `self.weight` → `self.weights`

测试用 `state_dict` 检查 key 名为 `weights`（复数），原代码写的是 `self.weight`。

**修复**：`self.weights = nn.Parameter(...)` 保持与测试一致。

##### 问题 2：RoPE 频率计算错误

原代码 `torch.arange(0, dim_half)` 只生成了 `d_k//2` 个位置，但没有正确处理 step=2 的 stride。

**修复**：`torch.arange(0, self.d_k, 2)` 直接按步长 2 生成，得到 `d_k//2` 个频率值。

##### 问题 3：softmax 多余的 eps

原代码 `e_x = torch.exp(x - max_val) + eps` 在 exp 后加 eps 会破坏概率和为 1 的性质。

**修复**：去掉 eps，`e_x / e_x.sum(...)` 本身不会除零（因为 exp 结果 > 0）。

##### 问题 4：cross_entropy 用 PyTorch 实现

原作业要求用 numpy 从零实现，不能依赖 PyTorch。

**修复**：纯 numpy 实现：`log_sum_exp = np.log(np.sum(np.exp(shifted), axis=-1))`。

**Part 4 最终：7/7 测试通过 ✅**

---

#### Part 5 提交

**得分：需修改后重提**

##### 问题 1：`float('-inft')` 拼写错误

`-inft` 不是有效的 Python，应为 `-inf`。

##### 问题 2：`self.d_modal` 拼写错误

两处引用了 `self.d_modal`，但 `__init__` 中定义的是 `self.d_model`。

##### 问题 3：Fused QKV vs Separate Q/K/V

原代码用 `self.qkv = Linear(d_model, d_model * 3)` 一次投影 Q/K/V，但测试检查 `hasattr(mha, 'w_q')`。

**修复**：改为 `self.w_q, self.w_k, self.w_v, self.w_o = Linear(...)` 分开定义。

##### 问题 4：forward 中残留 `self.qkv(x)` 引用

修改 init 后 forward 还在调用 `self.qkv(x)`，同时 B,S 也应从 `x.shape` 获取而非从 qkv。

**修复**：
```python
B, S, _ = x.shape
q, k, v = self.w_q(x), self.w_k(x), self.w_v(x)
```

**Part 5 最终：3/3 测试通过 ✅**

---

## Tutorial Part 6: 训练基础设施 — QA 记录
> 📅 2026-07-08

### 作业批改记录

#### Part 6 提交

**得分：需修改后重提**

##### 问题 1：Warmup 起点错误

原代码：`alpha_min + (alpha_max - alpha_min) * t / T_w`（从 alpha_min 线性增长到 alpha_max）

测试期望：`(t / T_w) * alpha_max`（从 0 线性增长到 alpha_max）

**修复**：改用 `(t / T_w) * alpha_max`。

##### 问题 2：gradient_clipping 空列表崩溃

`torch.stack([])` 在空列表时会报错。当所有参数都没有梯度时触发。

**修复**：
```python
params_with_grad = [p for p in parameters if p.grad is not None]
if not params_with_grad:
    return
```

**Part 6 最终：5/5 测试通过 ✅**

---

## Tutorial Part 7-8: 完整模型与训练 — QA 记录
> 📅 2026-07-08

### 作业批改记录

#### Part 7 提交

**得分：需修改后重提**

##### 问题 1：相对导入错误

`from .model_components import cross_entropy` 在直接运行测试时失败（没有 package 上下文）。

**修复**：删除相对导入，改用 `import torch.nn.functional as F`。

##### 问题 2：train 函数中 cross_entropy 未定义

删除导入后 `cross_entropy(...)` 调用报 NameError。

**修复**：改用 `F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))`。

##### 问题 3：Windows 上 int32 vs int64

`np.random.randint` 在 Windows 上默认生成 int32，PyTorch embedding 需要 int64。

**修复**：`get_batch` 中添加 `.astype(np.int64)`。

**Part 7 最终：3/3 测试通过 ✅**

---

#### Part 8 提交

**得分：通过 ✅**

Part 8（generate + evaluate + end-to-end）一次通过：
- generate：形状正确，prompt 保留，贪心确定性 ✅
- evaluate：返回 loss 和 perplexity ✅
- end-to-end：训练后 PPL 从 ~70 降到 ~35 ✅

**Part 8 最终：3/3 测试通过 ✅**
