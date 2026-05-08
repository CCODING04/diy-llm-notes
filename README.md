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

[![进度](https://img.shields.io/badge/进度-7%2F15%20章-blue)](https://github.com/datawhalechina/diy-llm) [![课程](https://img.shields.io/badge/课程-CS336-green)](https://stanford-cs336.github.io/spring2025/)

</div>

---

## 📈 学习进度

| # | 章节 | 状态 | 学习笔记 | 课后作业 |
|:-:|------|:----:|:--------:|:--------:|
| 1 | WandB 工具使用 | ✅ | [📖 notes.md](docs/chapter1/c/notes.md) | — |
| 2 | 分词器 | ✅ | [📖 notes.md](docs/chapter2/c/notes.md) | ✅ `assignment1-basics` |
| 3 | PyTorch 与资源核算 | ✅ | [📖 notes.md](docs/chapter3/c/notes.md) | ✅ `assignment1-basics` |
| 4 | 语言模型架构与训练细节 | ✅ | [📖 notes.md](docs/chapter4/c/notes.md) | 📂 `assignment1-basics` |
| 5 | 混合专家模型（MoE） | ✅ | [📖 notes.md](docs/chapter5/c/notes.md) | — |
| 6 | GPU 与相关优化 | ✅ | [📖 notes.md](docs/chapter6/c/notes.md) | 📂 `assignment2-systems` |
| 7 | GPU 高性能编程 | ✅ | [📖 notes.md](docs/chapter7/c/notes.md) | 📂 `assignment2-systems` |
| 8 | 分布式训练 | ○ | — | 📂 `assignment2-systems` |
| 9 | Scaling Laws | ○ | — | 📂 `assignment3-scaling` |
| 10 | 推理 | ○ | — | — |
| 11 | 数据工程 | ○ | — | 📂 `assignment4-data` |
| 12 | 评估与基准测试 | ○ | — | 📂 `assignment6-evaluation` |
| 13 | 大模型基本训练流程 | ○ | — | 📂 `assignment5-alignment` |
| 14 | 可验证奖励的强化学习 | ○ | — | 📂 `assignment5-alignment` |
| 15 | 扩展内容 | ○ | — | — |

> ✅ 已完成 &nbsp;|&nbsp; ○ 未开始 &nbsp;|&nbsp; 🔨 作业进行中 &nbsp;|&nbsp; **7 / 15 章** &nbsp;|&nbsp; 最后更新：2026-05-08

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
├── chapter4/c/               # 第4章
│   ├── module1.md            # 标准 Transformer 架构回顾
│   ├── module2.md            # 现代变体（归一化与激活函数）
│   ├── module3.md            # 现代变体（位置编码与注意力机制）
│   ├── module4.md            # 超参数设计与训练稳定性
│   └── notes.md              # 📊 学习总结 + QA 归档
├── chapter5/c/               # 第5章
│   ├── module1.md            # MoE 核心概念与路由机制
│   ├── module2.md            # 容量控制与 Token 丢弃
│   ├── module3.md            # 负载均衡与辅助损失
│   ├── module4.md            # DeepSeekMoE 与共享专家
│   └── notes.md              # 📊 学习总结 + QA 归档
├── chapter6/c/               # 第6章
│   ├── module1.md            # GPU 架构与内存层次
│   ├── module2.md            # 执行模型与性能扩展
│   ├── module3.md            # 低精度计算与算子融合
│   ├── module4.md            # 内存优化（Tiling、重计算与内存合并）
│   ├── module5.md            # FlashAttention 与 PagedAttention
│   └── notes.md              # 📊 学习总结 + QA 归档
├── chapter7/c/               # 第7章
│   ├── module1.md            # Benchmark 与 Profiling
│   ├── module2.md            # Kernel Fusion 与手写 CUDA 内核
│   ├── module3.md            # Triton 与 torch.compile
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

### 第 4 章：语言模型架构与训练细节

| 模块 | 标题 | 得分 |
|:----:|------|:----:|
| 1 | 标准 Transformer 架构回顾 | 22/30 |
| 2 | 现代变体（归一化与激活函数） | 18/30 |
| 3 | 现代变体（位置编码与注意力机制） | 21/30 |
| 4 | 超参数设计与训练稳定性 | 17/30 |

- **掌握扎实**：Transformer 四大核心组件、RoPE 数学推导、MQA/GQA/MLA 对比、KV Cache 原理
- **待加强**：正弦编码 vs RoPE 精确区别、RMSNorm 去掉的具体内容、MLA 缓存细节、权重衰减与 lr 正相关关系、SwiGLU 参数量计算

### 第 5 章：混合专家模型（MoE）

| 模块 | 标题 | 得分 |
|:----:|------|:----:|
| 1 | MoE 核心概念与路由机制 | 21/30 |
| 2 | 容量控制与 Token 丢弃 | 17/30 |
| 3 | 负载均衡与辅助损失 | 20/30 |
| 4 | DeepSeekMoE 与共享专家 | 20/30 |

- **掌握扎实**：MoE 路由机制（Top-K 路由、哈希路由、LSH 分桶）、容量控制与 Token 丢弃的残差保护、DeepSeekMoE 细粒度分割的组合空间计算、共享专家并行架构
- **待加强**：概念辨析精度（Z-loss vs Auxiliary Loss、FLOPs vs 显存）、数据流追踪、数值证明习惯

### 第 6 章：GPU 与相关优化

| 模块 | 标题 | 得分 |
|:----:|------|:----:|
| 1 | GPU 架构与内存层次 | — (临时 QA) |
| 2 | 执行模型与性能扩展 | 18/30 |
| 3 | 低精度计算与算子融合 | 21/30 |
| 4 | 内存优化（Tiling、重计算与内存合并） | 20/30 |
| 5 | FlashAttention 与 PagedAttention | 23/30 |

- **掌握扎实**：混合精度训练原理（FP16 vs BF16）、PagedAttention 分页机制、FlashAttention V1 Tiling + Online Softmax、内存合并 burst mode
- **待加强**：四大提速机制的精确命名、回答时多用具体数字、数学术语精度（"多项式增长"非"指数增长"）

### 第 7 章：GPU 高性能编程

| 模块 | 标题 | 得分 |
|:----:|------|:----:|
| 1 | Benchmark 与 Profiling | 21/30 |
| 2 | Kernel Fusion 与手写 CUDA 内核 | 25/30 |
| 3 | Triton 与 torch.compile | 21/30 |

- **掌握扎实**：预热与 CUDA 同步、CUDA 坐标计算与越界检查、empty_like 优化、Triton 向量化编程模型
- **待加强**：算术强度的实际应用（matmul 是高算术强度/计算受限）、HBM 往返的数值精度、"为什么需要手写内核"的场景化展开

---

## 🎯 当前建议

1. **开始 Assignment 2（Systems）**：第 6-8 章对应 assignment2-systems，涉及 GPU 性能优化、算子实现、分布式训练
2. **第 8 章学习**：分布式训练（数据并行、模型并行、ZeRO 优化）
3. **巩固练习**：用 torch.compile 优化一个自定义算子、尝试用 Triton 重写一个逐元素操作、复习 Roofline 模型计算算术强度

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
