# Assignment 2 — 分布式训练系统

> Stanford CS336 Assignment 2 Systems: 分布式数据并行训练与优化
> 开始日期：2026-07-08

---

## 作业概述

实现分布式训练系统的核心组件，包括：
1. **Part 1**：分布式通信基准测试（all-reduce 性能测量）
2. **Part 2**：朴素 DDP + 梯度展平（逐参数通信 vs 批量通信）
3. **Part 3**：DDP 计算通信重叠（梯度分桶 + 异步通信）
4. **Part 4**：优化器状态分片（简化版 ZeRO-1）

---

## 目录结构

```
homework/assignment2/
├── tutorials/                # 分步教程
│   ├── tutorial_part1.md     # 分布式通信基准测试
│   ├── tutorial_part2.md     # 朴素 DDP + 梯度展平
│   ├── tutorial_part3.md     # DDP 计算通信重叠
│   └── tutorial_part4.md     # 优化器状态分片
├── scripts/                  # 实现代码
│   ├── distributed_benchmark.py
│   ├── naive_ddp.py
│   ├── naive_ddp_flat.py
│   ├── ddp_bucketed.py
│   ├── sharded_optimizer.py
│   └── adapters.py
├── tests/                    # 测试用例
│   ├── common.py             # 测试辅助函数和玩具模型
│   ├── adapters.py           # 适配器接口（需实现）
│   ├── conftest.py           # Pytest 配置
│   ├── fixtures/             # 测试数据
│   │   ├── ddp_test_data.pt
│   │   └── ddp_test_labels.pt
│   ├── test_distributed_benchmark.py
│   ├── test_ddp_individual_parameters.py
│   ├── test_ddp_bucketed.py
│   └── test_sharded_optimizer.py
├── notes.md                  # QA 记录
└── suggestion.md             # 学习建议
```

---

## 完成状态

| 部分 | 内容 | 分数 | 状态 |
|------|------|------|------|
| Part 1 | 分布式通信基准测试 | 5 | ⬜ |
| Part 2a | 朴素 DDP | 5 | ⬜ |
| Part 2b | 朴素 DDP 基准测试 | 3 | ⬜ |
| Part 2c | 梯度展平 | 2 | ⬜ |
| Part 3a | DDP 计算通信重叠 | 5 | ⬜ |
| Part 3b | 分桶 DDP 基准测试 | 3 | ⬜ |
| Part 4 | 优化器状态分片 | 15 | ⬜ |

**总计：38 分**

---

## 关键技术概念

### 分布式通信原语

| 操作 | 说明 | DDP 用途 |
|------|------|---------|
| `all_reduce` | 所有进程的张量求和，结果写回每个进程 | 梯度同步 |
| `broadcast` | 一个进程的张量发送到所有进程 | 参数初始化 |
| `all_gather` | 收集所有进程的张量 | 结果汇总 |
| `barrier` | 所有进程同步等待 | 调试/基准测试 |

### DDP 优化层次

| 层次 | 技术 | 优势 |
|------|------|------|
| 朴素 DDP | 逐参数 all-reduce | 简单正确 |
| 梯度展平 | 单次 all-reduce | 减少通信开销 |
| 梯度分桶 | 异步通信 + 计算重叠 | 隐藏通信延迟 |
| ZeRO-1 | 优化器状态分片 | 降低内存占用 |

---

## 关联章节

| 章节 | 内容 | 关联 |
|------|------|------|
| 第 6 章 | GPU 与相关优化 | 性能分析、基准测试 |
| 第 7 章 | GPU 高性能编程 | Triton kernel（可选） |
| 第 8 章 | 分布式训练 | DDP、ZeRO、通信原语 |
