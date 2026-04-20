# Assignment 1 - Part 3：整合测试

> 📍 作业进度：Assignment 1，第 3 / 3 部分
> 📅 生成时间：2026-04-19
> 📎 原作业参考：[stanford-cs336/assignment1-basics](https://github.com/stanford-cs336/assignment1-basics)

---

## 目标

将 Part 1（Tokenizer）和 Part 2（BPE 训练）整合，完成端到端测试：
- 训练 BPE → 得到 vocab 和 merges
- 用 vocab 和 merges 创建 Tokenizer
- 验证 encode/decode roundtrip

---

## 测试清单

### 测试 1：BPE 训练正确性
- 在 `corpus.en` 上训练 vocab_size=500
- merges 必须与参考实现完全一致
- 必须在 1.5 秒内完成

### 测试 2：训练结果用于 Tokenizer
- 用训练得到的 vocab 和 merges 创建 Tokenizer
- 验证 encode → decode roundtrip

### 测试 3：Special token 处理
- 训练时传入 special_tokens
- 验证 special tokens 正确加入词表
- 创建 Tokenizer 后验证含 special token 的文本 roundtrip

---

## 关键知识点

1. **训练和编码的一致性**：训练时的预分词正则（含 `\p{N}+`）和 Tokenizer 编码时的正则（`\w` 版本）可能不同，但这不影响 roundtrip
2. **Vocab 构建**：训练输出的 `dict[int, bytes]` 直接传给 Tokenizer 的 `__init__`
3. **Merges 格式**：训练输出的 `list[tuple[bytes, bytes]]` 直接传给 Tokenizer 的 `__init__`
