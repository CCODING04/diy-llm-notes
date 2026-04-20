# Assignment 1 — Part 1 学习建议与知识补充

> 基于 Tutorial Part 1 期间的 QA 记录、3 次提交的错误分析生成
> 📅 2026-04-17

---

## 🔴 高优先级：Python bytes/str/编码基础

### 问题表现

3 次提交中有 2 次的核心错误都与 Python 编码类型混淆有关：
- **第 1 次**：`ord(b)` 对 int 类型调用导致 TypeError（不知道遍历 bytes 得到的是 int）
- **第 1 次**：vocab/merges 构建时把 token ID（int）当字节值传给 bytes_to_unicode
- **QA 中**：主动问了 `ord(b'\xe4')` 和 `encode/decode` 的关系，说明这个知识点不牢固

### 需要掌握的知识点

1. **Python bytes 遍历**：`bytes` 对象遍历时每个元素是 `int`（0-255），不是 `bytes`
   ```python
   for b in b'\xe4\xbd':   # b 是 int，不是 bytes
       print(type(b))       # <class 'int'>
   ```

2. **类型转换关系**：
   ```
   int ←→ bytes:  bytes([228]) ↔ b'\xe4'[0]
   int ←→ str:    chr(228) ↔ ord('ä')
   str ←→ bytes:  'ä'.encode('utf-8') ↔ b'\xc3\xa4'.decode('utf-8')
   ```

3. **latin1 vs UTF-8**：latin1 是单字节编码（码点 0-255 = 字节 0-255），UTF-8 对非 ASCII 字符用多字节

### 练习建议

- 写一个函数，实现 `bytes_to_unicode` 和 `unicode_to_bytes` 的互转，手动验证每个步骤
- 用 Python REPL 测试：`b'\xe4'` 的 `type`、遍历结果、`ord` 和 `chr` 的输入输出类型

---

## 🟡 中优先级：BPE 算法实现细节

### 问题表现

- **第 1 次**：apply_merges 用逐条遍历所有 merges，时间复杂度过高
- **第 2 次**：变量 `token_chars` 和 `word` 混淆导致合并结果未生效
- **第 3 次**：属性名 `self.merges_ranks` vs `self.merge_ranks` 拼写错误

### 需要掌握的知识点

1. **rank-based BPE 合并**：不应该遍历所有 merge 规则，而是从当前 token 序列中找 rank 最小的 pair 合并。这是 GPT-2 的标准实现方式。

2. **变量作用域意识**：循环中更新变量时，确保后续迭代使用的是更新后的值，不是原始值。这是一个常见的编程陷阱。

3. **命名一致性**：属性名在 `__init__` 中定义后，后续引用必须完全一致。建议定义后立即在 `apply_merges` 中写一个测试验证。

### 练习建议

- 手动模拟 BPE 合并过程：给定 `['h', 'e', 'l', 'l', 'o']` 和 merges `[('h','e'), ('he','l')]`，逐步跟踪 rank-based 合并过程
- 对比自己的 `apply_merges_old`（逐条遍历）和 `apply_merges`（rank-based）的输出是否一致

---

## 🟢 低优先级：工程实践习惯

### 问题表现

- special token 查找方式在第 1 次提交时未考虑 bytes_to_unicode 映射
- encode_iterable 的流式处理在第 1 次未考虑内存限制
- 代码中保留了 `apply_merges_old` 的旧实现（建议清理）

### 需要加强的习惯

1. **接口约定仔细阅读**：原作业的 `vocab: dict[int, bytes]` 和 `merges: list[tuple[bytes, bytes]]` 类型注解已经明确说明输入格式，实现前应仔细对照
2. **边界条件测试**：空字符串、单字符、多字节 Unicode 字符、special token 边界情况
3. **测试驱动开发**：先写简单测试脚本，每改一处就跑一次，而不是改完所有问题再统一测试

---

## 📚 推荐补充学习资源

| 主题 | 资源 |
|------|------|
| Python bytes/str | [Python 官方文档 Unicode HOWTO](https://docs.python.org/3/howto/unicode.html) |
| GPT-2 BPE 实现 | [OpenAI GPT-2 encoder.py 源码](https://github.com/openai/gpt-2/blob/master/src/encoder.py) |
| tiktoken 参考实现 | [tiktoken _educational.py](https://github.com/openai/tiktoken/blob/main/tiktoken/_educational.py) |

---

# Assignment 1 — Part 2 学习建议与知识补充

> 基于 Tutorial Part 2 期间的调试经历和错误分析生成
> 📅 2026-04-19

---

## 🔴 高优先级：BPE 训练中的关键细节

### 问题表现

Part 2 经历了多轮调试，主要问题集中在三个层面：

1. **正则表达式缺失数字匹配**：GPT-2 预分词正则缺少 `| ?\p{N}+`，导致数字字符被静默丢弃，训练到 merge 127 才暴露差异
2. **Tie-breaking 排序错误**：`bytes_to_unicode` 映射改变了字符排序（如空格 0x20 → Ġ U+0120），直接用 Unicode 字符串做字典序 tie-breaking 与参考实现不一致
3. **合并逻辑 bug**：`word[i:i+2] = best[0] + best[1]` 时 Python 会迭代字符串拆成两个字符，需要 `[best[0] + best[1]]` 包裹为列表

### 需要掌握的知识点

1. **GPT-2 完整预分词正则**：
   ```python
   r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
   #                                ^^^^^^^^ 数字匹配，不能遗漏
   ```
   每个分支的作用：缩写 → 字母 → **数字** → 标点 → 行末空白 → 其他空白

2. **`bytes_to_unicode` 对排序的影响**：
   - 原始字节：`0x20`（空格）< `0x74`（'t'）
   - 映射后：`Ġ`（U+0120, 288）> `'t'`（U+0074, 116）
   - **结论**：tie-breaking 必须转回原始字节再比较

3. **Python 列表切片赋值**：
   ```python
   word[i:i+2] = "ab"     # 拆成 ['a','b']，2换2，未合并
   word[i:i+2] = ["ab"]   # 包裹为列表，2换1，正确合并
   ```

4. **dict 方案 vs list 方案**：
   - List 方案：每个词实例单独存储 → 27758 个元素 → 慢（5.6s）
   - Dict 方案：`{word_tuple: freq}` 去重 → 4763 个元素 → 快（0.9s）
   - 按频率加权计数，结果等价

### 练习建议

- 手动追踪 BPE 训练的前 5 轮：给定文本 "the cat the dog"，写出每轮的 pair 频率和合并结果
- 对比 `max(stats, key=stats.get)` 和 `max(stats, key=lambda x: (stats[x], raw_bytes_key(x)))` 在有 tie 时的不同结果
- 理解为什么 `bytes_to_unicode` 存在（GPT-2 为了让 tokenizer.json 可读）

---

## 🟡 中优先级：调试方法论

### 问题表现

- 初始实现有多处 bug（语法错误、逻辑错误），一次性修改后无法定位具体问题
- 同一 bug 反复出现（合并逻辑修复后又在不同位置复发）

### 需要加强的习惯

1. **增量验证**：先确保正则匹配正确（打印 findall 结果），再确保 pair 计数正确，最后验证合并逻辑。不要一次性写完再测。

2. **对比参考实现**：遇到问题时直接读 tiktoken 的 `_educational.py` 源码对比，比自己猜测 tie-breaking 规则高效得多。

3. **性能意识**：dict 去重是 BPE 训练最基本的空间/时间优化。在实现正确后再考虑性能优化（先正确，再高效）。

---

## 📚 推荐补充学习资源

| 主题 | 资源 |
|------|------|
| BPE 算法详解 | [Sennrich et al. 2016 原始论文](https://arxiv.org/abs/1508.07909) |
| tiktoken 教育实现 | [tiktoken/_educational.py](https://github.com/openai/tiktoken/blob/main/tiktoken/_educational.py) |
| GPT-2 分词器 | [OpenAI GPT-2 encoder.py](https://github.com/openai/gpt-2/blob/master/src/encoder.py) |
