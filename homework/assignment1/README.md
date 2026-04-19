# Assignment 1 — BPE Tokenizer 与训练

> Stanford CS336 Assignment 1 Basics: BPE 分词器实现与训练
> 完成日期：2026-04-19

---

## 作业概述

实现一个完整的 BPE（Byte Pair Encoding）分词器，包括：
1. 基于 GPT-2 vocab/merges 的 Tokenizer 类（编码/解码）
2. 从语料训练 BPE 的 `run_train_bpe` 函数
3. 训练结果与 Tokenizer 的端到端整合

---

## 目录结构

```
homework/assignment1/
├── tutorials/              # 分步教程
│   ├── tutorial_part1.md   # Tokenizer 实现
│   ├── tutorial_part2.md   # BPE 训练
│   └── tutorial_part3.md   # 整合测试
├── scripts/                # 实现代码
│   ├── tokenizer.py        # Tokenizer 类（Part 1）
│   ├── train_bpe.py        # BPE 训练（Part 2）
│   └── train_bpe_answer.py # 参考答案
├── tests/                  # 测试用例
│   ├── test_basic.py       # Part 1 基础测试（10 项）
│   ├── test_integration.py # Part 3 整合测试（5 项）
│   └── fixtures/           # 测试数据
├── notes.md                # QA 记录
└── suggestion.md           # 学习建议
```

---

## 完成状态

| 部分 | 内容 | 状态 | 关键成果 |
|------|------|------|----------|
| Part 1 | Tokenizer 实现 | ✅ 完成 | encode/decode/tiktoken 完全匹配 |
| Part 2 | BPE 训练 | ✅ 完成 | 243 merges 精确匹配，0.90s < 1.5s |
| Part 3 | 整合测试 | ✅ 完成 | 训练→Tokenizer roundtrip 验证通过 |

### 测试结果

- **Part 1**：10/10 通过（含 tiktoken GPT-2 编码精确匹配）
- **Part 2**：243 merges 全部匹配参考实现，训练时间 0.90s
- **Part 3**：5/5 整合测试通过

---

## 关键实现要点

### Tokenizer（Part 1）
- rank-based BPE 合并：从当前 token 序列找 rank 最小的 pair 合并，而非逐条遍历 merges
- `bytes_to_unicode` 映射：GPT-2 为让 tokenizer.json 可读，将不可显示字节映射到 Unicode 字符
- Special token 处理：按 special token 分割文本后再分词

### BPE 训练（Part 2）
- GPT-2 预分词正则需包含 `| ?\p{N}+`（数字匹配）
- Tie-breaking：频率相同时按原始字节字典序选最大（非 Unicode 字符序）
- Dict 方案（`{word_tuple: freq}`）比 list 方案快约 6 倍
- `sort_key_cache` 缓存避免重复创建 bytes 排序键

---

## 关联章节

| 章节 | 内容 | 关联 |
|------|------|------|
| 第 2 章 | 分词器 | BPE 算法原理 |
| 第 3 章 | PyTorch 与资源核算 | 编码效率分析 |
| 第 4 章 | 语言模型架构 | Tokenizer 在模型中的作用 |
