<div align="center">

```
 ____  _____  ____       _  ____   ___  ____  
|  _ \| ____||  _ \  ___ | ||  _ \ / _ \/ ___| 
| | | |  _|  | | | |/ _ \| || |_) | | | \___ \ 
| |_| | |___ | |_| | (_) | ||  _ <| |_| |___) |
|____/|_____||____/ \___/|_||_| \_\\___/|____/ 
```

# DIY LLM 学习笔记

基于 [datawhalechina/diy-llm](https://github.com/datawhalechina/diy-llm) 教程的交互式学习记录

[![进度](https://img.shields.io/badge/进度-2%2F15%20章-blue)](https://github.com/datawhalechina/diy-llm) [![课程](https://img.shields.io/badge/课程-CS336-green)](https://stanford-cs336.github.io/spring2025/)

</div>

---

## 📈 学习进度

| # | 章节 | 状态 | 学习笔记 | 课后作业 |
|:-:|------|:----:|:--------:|:--------:|
| 1 | WandB 工具使用 | ✅ | [📖 notes.md](docs/chapter1/c/notes.md) | — |
| 2 | 分词器 | ✅ | [📖 notes.md](docs/chapter2/c/notes.md) | 📂 `assignment1-basics` |
| 3 | PyTorch 与资源核算 | ○ | — | 📂 `assignment1-basics` |
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

> ✅ 已完成 &nbsp;|&nbsp; ○ 未开始 &nbsp;|&nbsp; **2 / 15 章** &nbsp;|&nbsp; 最后更新：2026-04-15

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

---

## 🎯 当前建议

1. **优先完成 Assignment 1**：第 2 章已学完，BPE 分词器和 Transformer 实现的课后作业可以开始做了
2. **正则表达式复习**：学习过程中多次提出正则相关问题，建议系统复习 `[]` 内外行为差异
3. **下一章预告**：第 3 章 PyTorch 与资源核算，涉及 GPU 显存计算和算力估算

---

## 🔗 相关链接

- **原课程仓库**：[datawhalechina/diy-llm](https://github.com/datawhalechina/diy-llm)
- **在线阅读**：[datawhalechina.github.io/diy-llm](https://datawhalechina.github.io/diy-llm/)
- **Stanford CS336**：[stanford-cs336.github.io/spring2025](https://stanford-cs336.github.io/spring2025/)
