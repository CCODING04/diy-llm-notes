# Assignment 2 - Part 2：朴素 DDP 与梯度展平

> 📍 作业进度：Assignment 2，第 2 / 4 部分
> 📅 生成时间：2026-07-08
> 📎 原作业：`coursework/assignment2-systems/cs336_systems/作业1.ipynb`

---

## 目标与要求

### 问题 2a：朴素 DDP（5 分）

实现分布式数据并行训练：反向传播后对各参数梯度单独进行 all-reduce。验证多进程训练的权重与单进程匹配。

### 问题 2b：朴素 DDP 基准测试（3 分）

对 XL 模型进行基准测试，测量每步训练时间和通信时间比例。

### 问题 2c：梯度展平（2 分）

修改 DDP 实现，将所有梯度展平为一个张量后单次 all-reduce，与逐参数通信对比性能。

---

## 实现步骤

### 脚本框架

编辑以下两个文件：
- `scripts/naive_ddp.py` — Part 2a: 朴素 DDP 实现
- `scripts/naive_ddp_flat.py` — Part 2c: 梯度展平实现

### Step 1：朴素 DDP 训练流程（naive_ddp.py）

```
┌─────────────────────────────────────────────────────────┐
│                    Naive DDP 流程                        │
├─────────────────────────────────────────────────────────┤
│  1. broadcast: rank 0 的参数 → 所有 rank                 │
│  2. forward: 每个 rank 用自己的数据子集前向传播             │
│  3. backward: 每个 rank 计算本地梯度                      │
│  4. all_reduce: 逐参数同步梯度（SUM / world_size）        │
│  5. optimizer.step(): 所有 rank 用相同梯度更新参数         │
└─────────────────────────────────────────────────────────┘
```

文件中已有 `SimpleModel` 和 `init_distributed()`，只需填写 `naive_ddp_train()` 的 TODO：

```python
def naive_ddp_train(rank: int, world_size: int, backend: str):
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    init_distributed(rank, world_size, backend)
    # ... 设备设置 ...

    model = SimpleModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # TODO: 步骤 1 - 广播参数
    for param in model.parameters():
        dist.broadcast(param.data, src=0)

    model.train()
    for step in range(100):
        # TODO: 生成随机训练数据
        x = torch.randn(32, 784, device=device)
        y = torch.randint(0, 10, (32,), device=device)

        # TODO: 前向传播
        optimizer.zero_grad()
        logits = model(x)
        loss = nn.functional.cross_entropy(logits, y)

        # TODO: 反向传播
        loss.backward()

        # TODO: 逐参数 all-reduce 梯度
        for param in model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                param.grad.data /= world_size

        # TODO: 优化器更新
        optimizer.step()
```

### Step 2：梯度展平实现（naive_ddp_flat.py）

导入 `SimpleModel`，然后实现展平逻辑：

```python
from naive_ddp import SimpleModel, init_distributed, destroy_distributed

def flat_ddp_train(rank: int, world_size: int, backend: str):
    # ... 初始化代码 ...

    model = SimpleModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # 广播参数
    for param in model.parameters():
        dist.broadcast(param.data, src=0)

    # TODO: 记录每个参数的形状和大小
    grad_shapes = [p.shape for p in model.parameters()]
    grad_numels = [p.numel() for p in model.parameters()]
    total_numel = sum(grad_numels)

    for step in range(100):
        # ... 前向 + 反向 ...

        # TODO: 步骤 1 - 展平所有梯度
        flat_grad = torch.zeros(total_numel, device=device)
        offset = 0
        for p in model.parameters():
            if p.grad is not None:
                flat_grad[offset:offset + p.numel()] = p.grad.view(-1)
            offset += p.numel()

        # TODO: 步骤 2 - 单次 all-reduce
        dist.all_reduce(flat_grad, op=dist.ReduceOp.SUM)
        flat_grad /= world_size

        # TODO: 步骤 3 - 写回各参数梯度
        offset = 0
        for p, shape, numel in zip(model.parameters(), grad_shapes, grad_numels):
            if p.grad is not None:
                p.grad.copy_(flat_grad[offset:offset + numel].view(shape))
            offset += numel

        optimizer.step()
```

### Step 3：适配器接口

在 `tests/adapters.py` 中实现测试接口：

```python
# Part 2a 接口
def get_ddp_individual_parameters(module: torch.nn.Module) -> torch.nn.Module:
    # TODO: 返回你的 DDP 容器
    # 提示: 创建一个包装 module 的类，在 forward 中处理梯度同步
    raise NotImplementedError

def ddp_individual_parameters_on_after_backward(ddp_model, optimizer):
    # TODO: 在 backward 后、optimizer.step() 前执行
    # 提示: 等待所有梯度同步完成
    raise NotImplementedError
```

---

## 测试方法

```bash
cd homework/assignment2

# 运行 Part 2a 测试
pytest tests/test_ddp_individual_parameters.py -v

# 直接运行脚本
python scripts/naive_ddp.py
python scripts/naive_ddp_flat.py
```

---

## 难点与注意事项

| # | 难点 | 解决方案 |
|---|------|---------|
| 1 | 随机种子导致所有 rank 数据相同 | 每个 rank 使用不同的数据子集 |
| 2 | 逐参数 all-reduce 开销大 | 展平为单次通信（问题 2c） |
| 3 | 通信时间测量不准确 | 使用 `torch.cuda.synchronize()` |

---

## 关键概念

### 数据并行 vs 模型并行

| 并行方式 | 数据 | 模型 | 通信 |
|---------|------|------|------|
| 数据并行 | 分片 | 完整副本 | 梯度同步 |
| 模型并行 | 完整 | 分片 | 激活同步 |

### 为什么需要广播参数？

确保所有 rank 从**完全相同**的初始参数开始训练。如果不广播，每个 rank 的随机初始化不同，训练结果会发散。

### 展平梯度的优势

| 方式 | 通信次数 | 每次数据量 | 总开销 |
|------|---------|-----------|-------|
| 逐参数 | N（参数个数） | 小 | N × latency |
| 展平 | 1 | 大 | 1 × latency |

对于 XL 模型（~1000 个参数张量），展平可以将通信开销降低 1000 倍的 latency 部分。
