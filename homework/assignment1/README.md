# Assignment 1 — BPE Tokenizer 与 Transformer 语言模型

> Stanford CS336 Assignment 1 Basics: BPE 分词器 + 完整 Transformer LM
> 完成日期：Part 1-3: 2026-04-19, Part 4-9: 2026-07-08

---

## 作业概述

实现一个完整的语言模型训练系统，包括：
1. **Part 1-3**：BPE 分词器（编码/解码/训练）
2. **Part 4-5**：Transformer 基础算子（7 个）+ 注意力机制（3 个）
3. **Part 6**：训练基础设施（AdamW、余弦调度、梯度裁剪、检查点）
4. **Part 7-8**：完整 TransformerLM + 文本生成 + 困惑度评估
5. **Part 9**：汇总整合

---

## 目录结构

```
homework/assignment1/
├── tutorials/                # 分步教程
│   ├── tutorial_part1.md     # Tokenizer 实现
│   ├── tutorial_part2.md     # BPE 训练
│   ├── tutorial_part3.md     # 整合测试
│   ├── tutorial_part4.md     # 基础算子
│   ├── tutorial_part5.md     # 注意力机制
│   ├── tutorial_part6.md     # 训练基础设施
│   ├── tutorial_part7.md     # 完整模型
│   ├── tutorial_part8.md     # 生成与评估
│   └── tutorial_part9.md     # 汇总整合
├── scripts/                  # 实现代码
│   ├── tokenizer.py          # Tokenizer 类（Part 1）
│   ├── train_bpe.py          # BPE 训练（Part 2）
│   ├── model_components.py   # 模型组件（Part 4-5, 7-8）
│   └── training.py           # 训练组件（Part 6-8）
├── tests/                    # 测试用例
│   ├── test_basic.py         # Part 1 基础测试（10 项）
│   ├── test_integration.py   # Part 3 整合测试（5 项）
│   ├── test_part4.py         # Part 4 算子测试（7 项）
│   ├── test_part5.py         # Part 5 注意力测试（3 项）
│   ├── test_part6.py         # Part 6 训练测试（5 项）
│   ├── test_part7.py         # Part 7 模型测试（3 项）
│   └── test_part8.py         # Part 8 生成测试（3 项）
├── notes.md                  # QA 记录
└── suggestion.md             # 学习建议
```

---

## 完成状态

| 部分 | 内容 | 状态 | 测试 |
|------|------|------|------|
| Part 1 | Tokenizer 实现 | ✅ | 10/10 |
| Part 2 | BPE 训练 | ✅ | 243 merges 匹配 |
| Part 3 | 整合测试 | ✅ | 5/5 |
| Part 4 | 基础算子（Linear, Embedding, RMSNorm, SwiGLU, RoPE, softmax, cross_entropy） | ✅ | 7/7 |
| Part 5 | 注意力机制（scaled_dot_product_attention, MultiHeadAttention, TransformerBlock） | ✅ | 3/3 |
| Part 6 | 训练基础设施（AdamW, cosine_schedule, gradient_clipping, get_batch, checkpoint） | ✅ | 5/5 |
| Part 7 | TransformerLM + train | ✅ | 3/3 |
| Part 8 | generate + evaluate + end-to-end | ✅ | 3/3 |
| Part 9 | 汇总整合 | ✅ | — |

**总计：21 项模型/训练测试 + 15 项分词器测试 = 36 项全部通过**

---

## 架构概览

```
TransformerLM
├── token_embedding: Embedding(vocab_size, d_model)
├── layers: ModuleList([
│   └── TransformerBlock × N
│       ├── norm1: RMSNorm
│       ├── attention: MultiHeadAttention
│       │   ├── w_q, w_k, w_v, w_o: Linear (无 bias)
│       │   └── rope: RotaryPositionalEmbedding
│       ├── norm2: RMSNorm
│       └── ffn: SwiGLU
│           ├── w_gate: Linear(d_model, d_ff)
│           ├── w_up: Linear(d_model, d_ff)
│           └── w_down: Linear(d_ff, d_model)
├── norm: RMSNorm (最终归一化)
└── output: Linear(d_model, vocab_size) (LM head)
```

### 关键设计决策

| 决策 | 选择 | 原因 |
|------|------|------|
| 归一化 | RMSNorm (非 LayerNorm) | 计算更高效，不强制均值为 0 |
| 位置编码 | RoPE (非绝对位置) | 相对位置建模，长序列外推友好 |
| FFN | SwiGLU (非 ReLU FFN) | 门控机制更灵活，实验效果更好 |
| Norm 位置 | Pre-Norm (非 Post-Norm) | 梯度流更顺畅，训练更稳定 |
| 权重衰减 | AdamW 解耦 (非 L2) | 自适应学习率下效果更好 |
| 学习率 | 余弦退火 + warmup | 平滑衰减，训练初期稳定 |

---

## 关联章节

| 章节 | 内容 | 关联 |
|------|------|------|
| 第 2 章 | 分词器 | BPE 算法原理 |
| 第 3 章 | PyTorch 与资源核算 | 编码效率分析 |
| 第 4 章 | 语言模型架构 | Transformer 实现 |
