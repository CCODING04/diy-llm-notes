<div align="center">

```
    ____  ______  __   __    __    __  ___   _   __      __
   / __ \/  _/\ \/ /  / /   / /   /  |/  /  / | / /___  / /____  _____
  / / / // /   \  /  / /   / /   / /|_/ /  /  |/ / __ \/ __/ _ \/ ___/
 / /_/ // /    / /  / /___/ /___/ /  / /  / /|  / /_/ / /_/  __(__  )
/_____/___/   /_/  /_____/_____/_/  /_/  /_/ |_/ \____/\__/\___/____/
```

# DIY LLM 学习笔记

基于 [datawhalechina/diy-llm](https://github.com/datawhalechina/diy-llm) 教程的交互式学习记录

[![进度](https://img.shields.io/badge/进度-3%2F15%20章-blue)](https://github.com/datawhalechina/diy-llm) [![课程](https://img.shields.io/badge/课程-CS336-green)](https://stanford-cs336.github.io/spring2025/)

</div>

---

## 📈 学习进度

| # | 章节 | 状态 | 学习笔记 | 课后作业 |
|:-:|------|:----:|:--------:|:--------:|
| 1 | WandB 工具使用 | ✅ | [📖 notes.md](docs/chapter1/c/notes.md) | — |
| 2 | 分词器 | ✅ | [📖 notes.md](docs/chapter2/c/notes.md) | ✅ `assignment1-basics` |
| 3 | PyTorch 与资源核算 | ✅ | [📖 notes.md](docs/chapter3/c/notes.md) | ✅ `assignment1-basics` |
| 4 | 语言模型架构与训练细节 | ○ | — | 📂 `assignment1-basics` |
| 5 | 混合专家模型（MoE） | ○ | — | — |
| 6 | GPU 与相关优化 | ○ | — | 📂 `assignment2-systems` |
| 7 | GPU 高性能编程 | ○ | — | 📂 `assignment2-systems` |
| 8 | 分布式训练 | ○ | — | 📂 `assignment2-systems` |
| 9 | Scaling Laws | ○ | — | 📂 `assignment3-scaling` |
| 10 | 推理 | ○ | — | — |
| 11 | 数据工程 | ○ | — | 📂 `assignment4-data` |
| 12 | 评估与基准测试 | ○ | — | 📂 `assignment6-evaluation` |
| 13 | 大模型基本训练流程 | ○ | — | 📂 `assignment5-alignment` |
| 14 | 可验证奖励的强化学习 | ○ | — | 📂 `assignment5-alignment` |
| 15 | 扩展内容 | ○ | — | — |

> ✅ 已完成 &nbsp;|&nbsp; ○ 未开始 &nbsp;|&nbsp; 🔨 作业进行中 &nbsp;|&nbsp; **3 / 15 章** &nbsp;|&nbsp; 最后更新：2026-04-19

---

## 📂 笔记目录

每章学习笔记保存在 `chapter{N}/c/` 目录下：

```
docs/
├── chapter1/c/               # 第1章
│   ├── module1.md            # WandB 核心工作流
│   ├── module2.md            # WandB 进阶功能
│   └── notes.md              # 📊 学习总结 + QA 归档
├── chapter2/c/               # 第2章
│   ├── module1.md            # 分词器概述与数据准备
│   ├── module2.md            # 四种分词器原理与代码对比
│   ├── module3.md            # 迭代训练、DeepSeek 实战与思考延伸
│   └── notes.md              # 📊 学习总结 + QA 归档
├── chapter3/c/               # 第3章
│   ├── module1.md            # 资源核算思维与张量基础
│   ├── module2.md            # 内存管理与计算效率
│   ├── module3.md            # 模型构建与训练基础
│   └── notes.md              # 📊 学习总结 + QA 归档
└── ...
```

---

## 📝 各章学习概况

### 第 1 章：WandB 工具使用

| 模块 | 标题 | 得分 |
|:----:|------|:----:|
| 1 | 核心工作流 | 25/30 |
| 2 | 进阶功能 | 19/20 |

- **掌握扎实**：wandb.init/log/offline/Artifact
- **待加强**：name 唯一性认知、Sweeps 超参数搜索

### 第 2 章：分词器

| 模块 | 标题 | 得分 |
|:----:|------|:----:|
| 1 | 分词器概述与数据准备 | 25/30 |
| 2 | 四种分词器原理与代码对比 | 27/30 |
| 3 | 迭代训练、DeepSeek 实战与思考延伸 | 26/30 |

- **掌握扎实**：BPE 机制、字节级切分、四种分词器对比、latin1 编码
- **待加强**：正则表达式基础、BPE vs Unigram 概率采样差异
- **思考延申亮点**：视觉-文本特征对齐分析、少样本词频偏移分析

### 第 3 章：PyTorch 与资源核算

| 模块 | 标题 | 得分 |
|:----:|------|:----:|
| 1 | 资源核算思维与张量基础 | 23/30 |
| 2 | 内存管理与计算效率 | 17/30 |
| 3 | 模型构建与训练基础 | 19/30 |

- **掌握扎实**：张量维度操作与 einops、训练循环流程、反向传播概念、混合精度直觉
- **待加强**：数值计算精度（2 的幂次换算、多卡乘数）、资源核算公式细节（AdaGrad vs Adam 内存）、数学推导书面表达

---

## 🎯 当前建议

1. **开始第 4 章**：语言模型架构与训练细节，涉及 Transformer 核心组件
2. **数值计算基础练习**：2 的幂次心算、AdaGrad vs Adam 内存区别、反向传播矩阵推导
3. **Assignment 2 预备**：第 4 章学完后可开始 assignment2-systems（Flash Attention、DDP）

---

## 📦 课后作业

### Assignment 1 - BPE Tokenizer（✅ 已完成）

| Part | 内容 | 状态 | 笔记 |
|:----:|------|:----:|:----:|
| 1 | Tokenizer 类（encode/decode/encode_iterable） | ✅ 通过 | [📖 notes.md](homework/assignment1/notes.md) |
| 2 | BPE 训练（train_bpe） | ✅ 通过 | [📖 tutorial](homework/assignment1/tutorials/tutorial_part2.md) |
| 3 | 整合测试（训练→Tokenizer roundtrip） | ✅ 通过 | [📖 tutorial](homework/assignment1/tutorials/tutorial_part3.md) |

- **代码**：[homework/assignment1/scripts/](homework/assignment1/scripts/)
- **教程**：[homework/assignment1/tutorials/](homework/assignment1/tutorials/)
- **学习建议**：[suggestion.md](homework/assignment1/suggestion.md)

> **测试资源**：大型 fixture 文件（gpt2_vocab.json、gpt2_merges.txt、corpus.en、tinystories_sample_5M.txt）未纳入版本控制，请从 [Stanford CS336 原仓库](https://github.com/stanford-cs336/assignment1-basics/tree/main/tests/fixtures) 下载到 `homework/assignment1/tests/fixtures/`

---

## 🔗 相关链接

- **原课程仓库**：[datawhalechina/diy-llm](https://github.com/datawhalechina/diy-llm)
- **在线阅读**：[datawhalechina.github.io/diy-llm](https://datawhalechina.github.io/diy-llm/)
- **Stanford CS336**：[stanford-cs336.github.io/spring2025](https://stanford-cs336.github.io/spring2025/)
