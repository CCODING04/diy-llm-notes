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

[![进度](https://img.shields.io/badge/进度-15%2F15%20章-brightgreen)](https://github.com/datawhalechina/diy-llm) [![课程](https://img.shields.io/badge/课程-CS336-green)](https://stanford-cs336.github.io/spring2025/)

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
| 8 | 分布式训练 | ✅ | [📖 notes.md](docs/chapter8/c/notes.md) | 📂 `assignment2-systems` |
| 9 | Scaling Laws | ✅ | [📖 notes.md](docs/chapter9/c/notes.md) | 📂 `assignment3-scaling` |
| 10 | 推理 | ✅ | [📖 notes.md](docs/chapter10/c/notes.md) | — |
| 11 | 数据工程 | ✅ | [📖 notes.md](docs/chapter11/c/notes.md) | 📂 `assignment4-data` |
| 12 | 评估与基准测试 | ✅ | [📖 notes.md](docs/chapter12/c/notes.md) | 📂 `assignment6-evaluation` |
| 13 | 大模型基本训练流程 | ✅ | [📖 notes.md](docs/chapter13/c/notes.md) | 📂 `assignment5-alignment` |
| 14 | 可验证奖励的强化学习 | ✅ | [📖 notes.md](docs/chapter14/c/notes.md) | 📂 `assignment5-alignment` |
| 15 | 扩展内容 | ✅ | [📖 notes.md](docs/chapter15/c/notes.md) | — |
| ✨ | GLM-5.2 全景分析（附加） | 📖 | [📖 module5_extra.md](docs/chapter14/c/module5_extra.md) | — |

> ✅ 已完成 &nbsp;|&nbsp; ○ 未开始 &nbsp;|&nbsp; 🔨 作业进行中 &nbsp;|&nbsp; **15 / 15 章** &nbsp;|&nbsp; 最后更新：2026-06-25

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
├── chapter8/c/               # 第8章
│   ├── module1.md            # 数据并行（DP/DDP/FSDP）
│   ├── module2.md            # 模型并行（Pipeline/1F1B/GPipe）
│   ├── module3.md            # ZeRO 优化与 3D 并行
│   ├── module4.md            # 通信基础与 NCCL
│   ├── module5.md            # 混合精度训练
│   └── notes.md              # 📊 学习总结 + QA 归档
├── chapter9/c/               # 第9章
│   ├── module1.md            # Scaling Laws 核心发现
│   ├── module2.md            # Chinchilla 最优计算分配
│   ├── module3.md            # 涌现能力与超越 Chinchilla
│   ├── module4.md            # 数据约束与重复
│   └── notes.md              # 📊 学习总结 + QA 归档
├── chapter10/c/              # 第10章
│   ├── module1.md            # LLM 推理基础
│   ├── module2.md            # 投机解码
│   ├── module3.md            # KV Cache 与量化推理
│   └── notes.md              # 📊 学习总结 + QA 归档
├── chapter11/c/              # 第11章
│   ├── module1.md            # 数据质量与过滤
│   ├── module2.md            # 数据去重（MinHash/LSH）
│   ├── module3.md            # 数据配比与课程学习
│   ├── module4.md            # 合成数据生成
│   └── notes.md              # 📊 学习总结 + QA 归档
├── chapter12/c/              # 第12章
│   ├── module1.md            # 评估方法论
│   ├── module2.md            # 标准 Benchmark
│   ├── module3.md            # LLM-as-Judge
│   ├── module4.md            # 污染与泄漏
│   ├── module5.md            # 红队测试
│   └── notes.md              # 📊 学习总结 + QA 归档
├── chapter13/c/              # 第13章
│   ├── module1.md            # 预训练数据与基础设施
│   ├── module2.md            # 数据选择与课程学习
│   ├── module3.md            # SFT 与指令微调
│   ├── module4.md            # RLHF 与 PPO
│   ├── module5.md            # DPO 与对齐优化
│   └── notes.md              # 📊 学习总结 + QA 归档
└── chapter14/c/              # 第14章
    ├── module1.md            # RLVR 动机与 GRPO 算法演化
    ├── module2.md            # GRPO 训练循环、长度偏差与 Dr.GRPO
    ├── module3.md            # DeepSeek R1 案例研究
    ├── module4.md            # Kimi k1.5 与 Qwen 3 思考模式融合
    ├── module5_extra.md      # ✨ GLM-5.2 全景分析（附加选学）
    └── notes.md              # 📊 学习总结 + QA 归档
└── chapter15/c/              # 第15章
    ├── module1.md            # 预训练阶段的推理能力
    ├── module2.md            # 后训练与思维链
    ├── module3.md            # Prompt工程、工具增强与总结
    └── notes.md              # 📊 学习总结 + QA 归档
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

### 第 8 章：分布式训练

| 模块 | 标题 | 得分 |
|:----:|------|:----:|
| 1 | 数据并行（DP/DDP/FSDP） | 20/30 |
| 2 | 模型并行（Pipeline/1F1B/GPipe） | 20/30 |
| 3 | ZeRO 优化与 3D 并行 | 21/30 |
| 4 | 通信基础与 NCCL | 19/30 |
| 5 | 混合精度训练 | 19/30 |

- **掌握扎实**：Ring All-Reduce 通信模式、GPipe 气泡率公式 `(P-1)/(M+P-1)`、ZeRO-1/2/3 的分片层级（O/P/G）、3D 并行组合方式、NCCL 通信原语
- **待加强**：1F1B vs Interleaved 1F1B（VPP）的精确区分、Reduce-Scatter vs All-Gather 的使用场景区分、公式推导时单位/数量级的检查习惯

### 第 9 章：Scaling Laws

| 模块 | 标题 | 得分 |
|:----:|------|:----:|
| 1 | Scaling Laws 核心发现 | 19/30 |
| 2 | Chinchilla 最优计算分配 | 20/30 |
| 3 | 涌现能力与超越 Chinchilla | 20/30 |
| 4 | 数据约束与重复 | 20/30 |

- **掌握扎实**：幂律关系与 log-log 线性、Chinchilla 公式 L(N,D)=E+AN^(-α)+BD^(-β)、WSD 调度器"一次训练多次 Decay"、μP 的工程价值
- **待加强**：μP 的适用范围（架构宽度迁移 vs 数据语言无关）、单位换算习惯（B=10⁹）、"为什么能用小模型外推"的因果链条描述、MoE 专用 96:1 数据:参数比例

### 第 10 章：推理

| 模块 | 标题 | 得分 |
|:----:|------|:----:|
| 1 | LLM 推理基础 | 20/30 |
| 2 | 投机解码 | 20/30 |
| 3 | KV Cache 与量化推理 | 16/30 |

- **掌握扎实**：Prefill/Decode 两阶段流程与 KV Cache 工作原理、算术强度计算与 memory-bound vs compute-bound 判断、推测解码的 draft-verify 流程
- **待加强**：线性注意力的因果性保证（前缀累加的精确数学表达）、级联解码 vs 推测级联的区分（token 级 vs block 级）、KV Cache 量化的具体实现细节

### 第 11 章：数据工程

| 模块 | 标题 | 得分 |
|:----:|------|:----:|
| 1 | 数据质量与过滤 | 20/30 |
| 2 | 数据去重（MinHash/LSH） | 19/30 |
| 3 | 数据配比与课程学习 | 22/30 |
| 4 | 合成数据生成 | 22/30 |

- **掌握扎实**：PPL 过滤的"恶性循环"因果链、MinHash 的 Jaccard 相似度计算（100% 精确）、FastText 哈希 Embedding 机制、合成数据的"真实数据奠基、合成数据精调"范式
- **待加强**：启发式过滤的"误杀"vs"漏杀"对称分析、定量推理习惯（具体数字而非定性描述）、"为什么不能纯合成数据"的机制级论证

### 第 12 章：评估与基准测试

| 模块 | 标题 | 得分 |
|:----:|------|:----:|
| 1 | 评估方法论 | 19/30 |
| 2 | 标准 Benchmark | 19/30 |
| 3 | LLM-as-Judge | 20/30 |
| 4 | 污染与泄漏 | 20/30 |
| 5 | 红队测试 | 21/30 |

- **掌握扎实**："不对称错误"的分层测试设计、跨模块知识串联（MLEBench 中调用"元决策"）、IFEval vs AlpacaEval 的适用范围区分、GCG 攻击的"概率分布对抗性重定向"直觉
- **待加强**：方法论级批判性思维（"这个结论在什么条件下不成立？"）、机制拆解到 token 级/梯度级、抽象结论配具体例子（如 Goodhart's Law 需"冷知识难题"等失败模式）

### 第 13 章：大模型基本训练流程

| 模块 | 标题 | 得分 |
|:----:|------|:----:|
| 1 | 预训练数据与基础设施 | 21/30 |
| 2 | 数据选择与课程学习 | 22/30 |
| 3 | SFT 与指令微调 | 20/30 |
| 4 | RLHF 与 PPO | 11/30 |
| 5 | DPO 与对齐优化 | 21/30 |

- **掌握扎实**：预训练 next-token 预测的输入/标签构造、ChatML 的 chat template 转换机制、Bradley-Terry → DPO Loss 的四步推导、PPO 中 clip 机制的保守选择原理、GAE 的偏差-方差权衡
- **待加强**：RM vs V 的概念区分（Mod 4 核心失分项）、clip vs KL 两种约束的对比（局部速率限制 vs 全局锚点）、"为什么"层面的攻击路径展开、"极限测试"思维（r_t→0 时的 PPO 行为分析）

### 第 14 章：可验证奖励的强化学习

| 模块 | 标题 | 得分 |
|:----:|------|:----:|
| 1 | RLVR 动机与 GRPO 算法演化 | 18/30 |
| 2 | GRPO 训练循环、长度偏差与 Dr.GRPO | 20.5/30 |
| 3 | DeepSeek R1 案例研究 | 18/30 |
| 4 | Kimi k1.5 与 Qwen 3 思考模式融合 | 17/30 |

- **掌握扎实**：GRPO 的 z-score 归一化与 group-relative advantage、三种长度控制方案的分层理解（梯度层/奖励层/推理层）、"越错越长"恶性循环的因果链路、PRM 与 RLVR 的结构性矛盾
- **待加强**：`1/std`（难度偏差）vs `1/|o_i|`（长度偏差）的精确区分、跨层分析（gradient/reward/generation 三层控制面）、边界条件分析（"std≈0 时 GRPO 如何退化"）、定性判断到因果链展开（至少 3 步推理）

### 第 15 章：扩展内容（LLM 推理）

| 模块 | 标题 | 得分 |
|:----:|------|:----:|
| 1 | 预训练阶段的推理能力 | 22/30 |
| 2 | 后训练与思维链 | 25/30 |
| 3 | Prompt 工程、工具增强与总结 | 23/30 |

- **掌握扎实**：Pass@k 分析与概率重分配、DTR 机制（JS 散度追踪逐层预测分布）、FFN 键值记忆的深层/浅层分工、CoT Decoding vs CoT Prompting 区分、"语言是思维的载体不是思维本身"的深层理解
- **待加强**：机制描述精度（从"现象级"到"机制级"——如 attention softmax 摊薄的具体过程）、CoT 倒 U 型的三条机理（注意力稀释 + 关键信息遗忘 + 恶性修正循环）、自我改进闭环的具体循环描述

---

## 🎯 当前建议

1. **🎉 全部 15 章课程已完成！** 可开始系统复习薄弱环节（见各章"待加强"部分）
2. **补做作业**：Assignment 2 (Systems)、3 (Scaling)、4 (Data)、5 (Alignment)、6 (Evaluation) 均未开始
3. **优先推荐**：Assignment 5（Alignment，对应第 13-14 章 RLHF/DPO/GRPO）——已完成课程理论学习，实现层面收益最大
4. **第 14 章复习**：重点关注 `1/std` vs `1/|o_i|` 概念辨析，以及 R1/Kimi/Qwen 3 的多方法对比分析

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
