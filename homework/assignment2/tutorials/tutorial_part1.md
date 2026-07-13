# Assignment 2 - Part 1：分布式通信基准测试

> 📍 作业进度：Assignment 2，第 1 / 4 部分
> 📅 生成时间：2026-07-08
> 📎 原作业：`coursework/assignment2-systems/cs336_systems/作业1.ipynb`

---

## 目标与要求

编写一个脚本，用于在**单节点多进程**设置下，**基准测试 all-reduce 操作的运行时间**。

**分数**：5 分

### 实验设置

| 维度 | 选项 |
|------|------|
| 后端 + 设备 | Gloo + CPU, NCCL + GPU |
| 张量大小 | 1MB, 10MB, 100MB, 1GB (float32) |
| 进程数量 | 2, 4 |

### 交付内容

- 图表和/或表格，比较不同设置下的性能
- 2-3 句话的分析说明

---

## 实现步骤

### 脚本框架

编辑 `scripts/distributed_benchmark.py`，文件已包含完整的函数签名和 TODO 标记。

### Step 1：分布式环境初始化

文件中已有 `init_distributed()` 和 `destroy_distributed()` 函数，只需填写 TODO 部分：

```python
def init_distributed(rank: int, world_size: int, backend: str):
    """初始化分布式进程组"""
    # TODO: 设置 MASTER_ADDR 和 MASTER_PORT
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"

    # TODO: 初始化进程组
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
```

### Step 2：All-Reduce 基准测试函数

`benchmark_all_reduce()` 函数的关键实现点：

```python
def benchmark_all_reduce(rank, world_size, tensor_size_mb, backend, device):
    # TODO: 初始化分布式环境
    init_distributed(rank, world_size, backend)

    # TODO: 如果是 GPU，绑定设备（rank i → GPU i）
    if device == "cuda":
        torch.cuda.set_device(rank)

    # TODO: 构造测试张量
    # tensor_size_mb 是 MB，float32 占 4 字节
    num_elements = (tensor_size_mb * 1024 * 1024) // 4
    tensor = torch.randn(num_elements, device=device)

    # TODO: Warm-up（5 次迭代）
    for _ in range(5):
        dist.all_reduce(tensor)
        if device == "cuda":
            torch.cuda.synchronize()

    # TODO: 同步所有进程，确保同时开始
    dist.barrier()

    # TODO: 正式测试（20 次迭代）
    start_time = time.time()
    for _ in range(20):
        dist.all_reduce(tensor)
    if device == "cuda":
        torch.cuda.synchronize()
    end_time = time.time()

    # TODO: 计算性能指标
    avg_latency = (end_time - start_time) / 20
    bandwidth_gbps = (tensor_size_mb * 1024 * 1024) / avg_latency / 1e9

    # TODO: 只让 rank 0 打印结果
    if rank == 0:
        print(f"Backend={backend:<5} Device={device:<4} World={world_size} "
              f"Size={tensor_size_mb:>4}MB: "
              f"Latency={avg_latency*1000:.3f}ms  Bandwidth={bandwidth_gbps:.2f}GB/s")

    # TODO: 清理环境
    destroy_distributed()
```

### Step 3：遍历所有配置

`main()` 函数使用 `spawn` 启动多进程：

```python
def main():
    # 测试 Gloo + CPU
    for size in [1, 10, 100, 1000]:
        for ws in [2, 4]:
            spawn(benchmark_all_reduce,
                  args=(ws, size, "gloo", "cpu"),
                  nprocs=ws, join=True)

    # 测试 NCCL + GPU（如果有）
    if torch.cuda.is_available():
        for size in [1, 10, 100, 1000]:
            for ws in [2, 4]:
                spawn(benchmark_all_reduce,
                      args=(ws, size, "nccl", "cuda"),
                      nprocs=ws, join=True)
```

---

## 测试方法

```bash
cd homework/assignment2
python scripts/distributed_benchmark.py
```

预期输出：表格或图表，展示不同配置下的延迟和带宽。

---

## 难点与注意事项

| # | 难点 | 解决方案 |
|---|------|---------|
| 1 | CUDA 异步执行导致计时不准确 | 每次 all-reduce 后调用 `torch.cuda.synchronize()` |
| 2 | NCCL 不支持 CPU 张量 | Gloo 用于 CPU 测试，NCCL 用于 GPU 测试 |
| 3 | 多进程打印顺序不确定 | 只让 rank 0 打印结果 |
| 4 | GPU 数量不足 | 只测试可用的 GPU 数量，在报告中说明 |

---

## 关键概念

### All-Reduce 操作

将所有进程的张量求和，并将结果写回到每个进程：

```
Rank 0: [1, 2, 3]    Rank 1: [4, 5, 6]
         ↓ all_reduce(SUM) ↓
Rank 0: [5, 7, 9]    Rank 1: [5, 7, 9]  ← 相同结果
```

### Gloo vs NCCL

| 特性 | Gloo | NCCL |
|------|------|------|
| 设备 | CPU / GPU（有限） | GPU（NVIDIA 专用） |
| 性能 | 中等 | 很高 |
| 用途 | 调试 / CPU 训练 | 生产级 GPU 训练 |

### 带宽计算

$$\text{Bandwidth} = \frac{\text{Data Size (bytes)}}{\text{Latency (seconds)}}$$

对于 all-reduce，理论通信量是 $2 \times \text{Data Size}$（reduce-scatter + all-gather），但实际测量的是端到端时间。
