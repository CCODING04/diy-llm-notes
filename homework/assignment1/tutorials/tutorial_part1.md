# Assignment 1 - Part 1：BPE Tokenizer 类实现

> 📍 作业进度：Assignment 1，第 1 / 3 部分
> 📅 生成时间：2026-04-16
> 📎 原作业参考：[stanford-cs336/assignment1-basics](https://github.com/stanford-cs336/assignment1-basics)

---

## 目标与要求

实现一个 BPE Tokenizer 类，使用**已有的 vocab 和 merges** 进行 encode/decode，要求输出与 GPT-2 的 tiktoken 库完全一致。

### 原作业要求（摘自 adapters.py）

```python
def get_tokenizer(
    vocab: dict[int, bytes],          # token ID → 字节内容
    merges: list[tuple[bytes, bytes]], # 合并规则（有序）
    special_tokens: list[str] | None = None,  # 特殊 token
) -> Any:
    """返回一个 BPE tokenizer，需实现以下方法"""
```

Tokenizer 类需要实现三个核心方法：

| 方法 | 签名 | 功能 |
|------|------|------|
| `encode` | `encode(text: str) -> list[int]` | 将字符串编码为 token ID 列表 |
| `decode` | `decode(ids: list[int]) -> str` | 将 token ID 列表解码为字符串 |
| `encode_iterable` | `encode_iterable(iterable: Iterable[str]) -> Iterator[int]` | 流式编码（内存友好） |

### 关键接口约定

- **vocab**：`dict[int, bytes]`，token ID → 字节内容。注意 key 是 int，value 是 bytes
- **merges**：`list[tuple[bytes, bytes]]`，合并规则按训练顺序排列。编码时按此顺序依次应用
- **special_tokens**：字符串列表，在编码时不会被拆分，作为一个完整 token 处理

---

## 实现步骤

### Step 1：预分词（Pre-tokenization）

GPT-2 使用以下正则进行预分词（将文本切分为词）：

```python
import regex

GPT2_PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\w+| ?[^\s\w]+|\s+(?!\S)|\s+"""
```

这个正则的作用：
- `'(?:[sdmt]|ll|ve|re)` — 匹配英文缩写（'s, 'd, 'm, 't, 'll, 've, 're）
- ` ?\w+` — 可选前导空格 + 连续字母数字下划线
- ` ?[^\s\w]+` — 可选前导空格 + 连续非空白非字母数字字符（标点等）
- `\s+(?!\S)` — 末尾空白（后面不跟非空白字符）
- `\s+` — 其余空白

**注意**：如果文本中包含 special tokens（如 `<|endoftext|>`），需要先按 special tokens 拆分，再对每段分别预分词。可以用 `regex.split` 以 special tokens 为分隔符。

### Step 2：字节编码

GPT-2 使用一种特殊的字节到 Unicode 映射（`gpt2_bytes_to_unicode`），将不可打印字节映射为可打印 Unicode 字符。

核心思路：
```python
def bytes_to_unicode():
    # 可直接显示的字节（33-126, 161-172, 174-255）保持不变
    # 不可显示的字节（0-32, 127-160, 173）映射到 256+ 的 Unicode 码点
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))
```

编码流程：
```
原始文本 → 预分词得到词列表 → 每个词转为 UTF-8 字节 → 用 byte_encoder 映射为 Unicode 字符序列
```

### Step 3：BPE 合并

对每个预分词得到的词（已转为 Unicode 字符序列），按 merges 顺序依次合并：

```python
def apply_merges(token_chars: list[str], merges: list[tuple]) -> list[str]:
    """按 merges 顺序贪心合并相邻字符对"""
    for merge_pair in merges:
        # 在 token_chars 中找到所有匹配 merge_pair 的相邻位置
        # 合并为一个 token
        # 重复直到没有更多匹配
    return token_chars
```

**重要**：每次合并一对后，需要重新扫描（因为合并可能产生新的相邻对）。

### Step 4：查词表得到 ID

```python
merged_tokens = apply_merges(char_sequence, merges)
ids = [vocab[token_bytes] for token in ...]
```

需要构建一个 `str → int` 的反向查找表（token 字符串 → ID）。

### Step 5：Decode 实现

```python
def decode(self, ids: list[int]) -> str:
    # 1. 查 vocab 反向表，将每个 ID 转为 bytes
    # 2. 拼接所有 bytes
    # 3. 用 UTF-8 解码为字符串
    # 注意：special tokens 的 bytes 直接是 UTF-8 编码
```

### Step 6：encode_iterable 实现

流式版本，逐字符读取输入，避免一次性加载整个文件到内存：

```python
def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
    # 思路：维护一个 buffer，不断从 iterable 读取字符
    # 当 buffer 中积累了一个完整的预分词单元时，进行编码并 yield
    # 需要用正则的 partial match 来判断何时 buffer 足够
```

---

## 测试方法

测试文件已下载到 `homework/assignment1/tests/fixtures/`，核心测试用例：

### 基础 roundtrip 测试

```python
# encode 后 decode 应还原原文
text = "Hello, how are you?"
ids = tokenizer.encode(text)
assert tokenizer.decode(ids) == text
```

### tiktoken 一致性测试

```python
import tiktoken
reference = tiktoken.get_encoding("gpt2")

for text in ["", "s", "🙃", "Hello, how are you?", "Héllò hôw are ü? 🙃"]:
    assert tokenizer.encode(text) == reference.encode(text)
    assert tokenizer.decode(tokenizer.encode(text)) == text
```

### Special token 测试

```python
tokenizer = get_tokenizer(vocab, merges, special_tokens=["<|endoftext|>"])
text = "Hello<|endoftext|>world"
ids = tokenizer.encode(text)
assert tokenizer.decode(ids) == text
# <|endoftext|> 应作为一个完整 token，不被拆分
```

### 运行测试

```bash
cd homework/assignment1
# 安装依赖
uv pip install tiktoken regex

# 运行 Part 1 对应的测试
python -m pytest tests/ -v -k "roundtrip or matches_tiktoken"
```

---

## 难点与注意事项

1. **字节到 Unicode 映射**：GPT-2 的 `bytes_to_unicode()` 映射是固定的，必须使用这个映射才能与 tiktoken 一致。空间字符 `' '` 映射为 `'Ġ'`（U+0120）

2. **merges 的应用顺序**：必须按 merges 列表的顺序依次应用，且每次应用一对后要重新扫描。不能一次性找所有匹配位置

3. **Special token 优先级**：编码时先按 special tokens 拆分文本，special token 之间的普通文本才走 BPE 流程

4. **空字符串处理**：`encode("")` 应返回 `[]`，`decode([])` 应返回 `""`

5. **encode_iterable 内存限制**：测试要求在 1MB 内存限制下处理 5MB 文件。不要一次性 `read()` 整个文件，要逐块读取并处理

6. **vocab 的 value 是 bytes 类型**：注意 `vocab` 中 value 是 `bytes`，不是 `str`。在构建查找表时需要注意类型转换

---

## 代码框架

在 `homework/assignment1/scripts/tokenizer.py` 中实现，建议结构：

```python
import regex
from collections.abc import Iterable, Iterator

class Tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]],
                 special_tokens: list[str] | None = None):
        # TODO: 初始化
        # - 存储 vocab、merges
        # - 构建 vocab 的反向查找表 (bytes/str → int)
        # - 构建 merge pair 到 rank 的映射（用于快速查找合并优先级）
        # - 处理 special tokens（构建 split pattern）
        pass

    def encode(self, text: str) -> list[int]:
        # TODO: 实现
        pass

    def decode(self, ids: list[int]) -> str:
        # TODO: 实现
        pass

    def encode_iterable(self, text: Iterable[str]) -> Iterator[int]:
        # TODO: 实现
        pass


def get_tokenizer(vocab, merges, special_tokens=None):
    return Tokenizer(vocab, merges, special_tokens)
```

完成后输入 `提交作业` 运行测试验证。
