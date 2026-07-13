# Assignment 2 - Part 3：DDP 计算通信重叠

> 📍 作业进度：Assignment 2，第 3 / 4 部分
> 📅 生成时间：2026-07-08
> 📎 原作业：`coursework/assignment2-systems/cs336_systems/作业2.ipynb`

---

## 目标与要求

### 问题 3a：DDP 计算通信重叠（5 分）

实现一个 DDP 容器类，通过**梯度分桶**和**异步通信**实现计算与通信的重叠。

### 问题 3b：分桶 DDP 基准测试（3 分）

变化桶大小（1, 10, 100, 1000 MB），基准测试并分析结果。

### 问题 3c：DDP 开销建模（包含在 3b 中）

推导 DDP 通信开销和最优桶大小的方程。

---

## 实现步骤

### 脚本框架

编辑 `scripts/ddp_bucketed.py`，文件已包含完整的 `DDPBucketed` 类框架和 TODO 标记。

### Step 1：理解计算通信重叠

```
┌─────────────────────────────────────────────────────────────┐
│                    朴素 DDP（无重叠）                         │
├─────────────────────────────────────────────────────────────┤
│  Forward │ Backward │ ← 等待 → │ All-Reduce │ Optimizer     │
│  ────────│──────────│──────────│────────────│────────────   │
│  计算     │ 计算      │ 空闲     │ 通信        │ 更新          │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                    分桶 DDP（有重叠）                         │
├─────────────────────────────────────────────────────────────┤
│  Forward │ Backward(layer N) │ All-Reduce(bucket 1) │       │
│          │ Backward(layer N-1)│ All-Reduce(bucket 2) │       │
│          │ Backward(layer N-2)│ All-Reduce(bucket 3) │       │
│          │        ...         │        ...            │       │
└─────────────────────────────────────────────────────────────┘
  计算和通信同时进行！
```

### Step 2：梯度分桶（_create_buckets）

```python
def _create_buckets(self):
    """按大小创建梯度桶（倒序遍历，因为反向传播从后往前）"""
    current_bucket = []
    current_size = 0

    # TODO: 倒序遍历参数，按 bucket_size_bytes 分组
    for p in reversed(list(self.module.parameters())):
        if not p.requires_grad:
            continue

        p_size = p.numel() * p.element_size()

        # 当前桶满了，保存并创建新桶
        if current_bucket and (current_size + p_size > self.bucket_size_bytes):
            self._finalize_bucket(current_bucket)
            current_bucket = []
            current_size = 0

        current_bucket.append(p)
        current_size += p_size

    # 保存最后一个桶
    if current_bucket:
        self._finalize_bucket(current_bucket)
```

### Step 3：梯度钩子（_register_hooks + _on_gradient_ready）

```python
def _register_hooks(self):
    """为每个参数注册梯度钩子"""
    # TODO: 为每个参数注册钩子，梯度准备好时调用 _on_gradient_ready
    for bucket_idx, bucket in enumerate(self.buckets):
        for param in bucket["params"]:
            param.register_hook(
                lambda grad, b_idx=bucket_idx: self._on_gradient_ready(b_idx)
            )

def _on_gradient_ready(self, bucket_idx: int):
    """梯度就绪回调"""
    bucket = self.buckets[bucket_idx]
    bucket["ready_count"] += 1

    # TODO: 检查是否所有梯度都准备好
    if (bucket["ready_count"] == bucket["total_params"] and
            not bucket["triggered"]):
        bucket["triggered"] = True

        def launch_all_reduce():
            # TODO: 拷贝梯度到缓冲区
            offset = 0
            for p in bucket["params"]:
                numel = p.numel()
                if p.grad is not None:
                    bucket["buffer"][offset:offset + numel].copy_(p.grad.view(-1))
                else:
                    bucket["buffer"][offset:offset + numel].zero_()
                offset += numel

            # TODO: 启动异步 all-reduce
            handle = dist.all_reduce(bucket["buffer"], async_op=True)
            self.handles.append((handle, bucket_idx))

        # 延迟到反向传播完成后执行
        torch.autograd.Variable._execution_engine.queue_callback(launch_all_reduce)
```

### Step 4：等待通信完成（finish_gradient_synchronization）

```python
def finish_gradient_synchronization(self):
    """在 optimizer.step() 之前调用"""
    # TODO: 等待所有 all-reduce 完成
    for handle, bucket_idx in self.handles:
        handle.wait()

        bucket = self.buckets[bucket_idx]
        bucket["buffer"].div_(self.world_size)

        # TODO: 写回梯度到各参数
        offset = 0
        for p in bucket["params"]:
            numel = p.numel()
            if p.grad is not None:
                p.grad.view(-1).copy_(bucket["buffer"][offset:offset + numel])
            offset += numel

    self.handles.clear()
```

### Step 5：适配器接口

在 `tests/adapters.py` 中实现测试接口：

```python
def get_ddp_bucketed(module: torch.nn.Module, bucket_size_mb: float) -> torch.nn.Module:
    # TODO: 返回你的 DDPBucketed 实例
    from scripts.ddp_bucketed import DDPBucketed
    return DDPBucketed(module, bucket_size_mb)

def ddp_bucketed_on_after_backward(ddp_model, optimizer):
    # TODO: 等待梯度同步完成
    ddp_model.finish_gradient_synchronization()

def ddp_bucketed_on_train_batch_start(ddp_model, optimizer):
    # TODO: 每个训练步骤开始时的操作（如果需要）
    pass
```

### Step 6：DDP 开销建模

设：
- $s$ = 模型参数总大小（字节）
- $w$ = all-reduce 带宽（字节/秒）
- $o$ = 每次通信调用的开销（秒）
- $n_b$ = 桶的数量

**通信开销方程**：

$$\text{Overhead} = n_b \times \left(\frac{s}{n_b \times w} + o\right) = \frac{s}{w} + n_b \times o$$

**最优桶大小**：

$$B^* = \sqrt{s \times o \times w}$$

---

## 测试方法

```bash
cd homework/assignment2

# 运行测试（需要 2+ GPU 或 CPU gloo）
pytest tests/test_ddp_bucketed.py -v

# 直接运行脚本
python scripts/ddp_bucketed.py
```

---

## 难点与注意事项

| # | 难点 | 解决方案 |
|---|------|---------|
| 1 | 梯度钩子的闭包捕获 | 使用默认参数 `lambda grad, b_idx=i: ...` |
| 2 | `queue_callback` 时机 | 确保在反向传播完成后才启动通信 |
| 3 | 桶的创建顺序 | 倒序遍历参数，因为反向传播从后往前 |
| 4 | 测试不稳定 | 多次运行确保可靠通过 |

---

## 关键概念

### 为什么倒序遍历参数？

反向传播的顺序是从模型的最后一层到第一层。梯度计算的顺序与 `model.parameters()` 的顺序相反。倒序遍历可以确保桶内的参数在同一时间点准备好梯度。

### 异步通信的优势

```
同步：  Compute → Wait → Compute → Wait → ...
异步：  Compute → Compute → Compute → ...
              ↘ Comm ↗   ↘ Comm ↗
```

异步通信允许计算和通信同时进行，充分利用 GPU 的计算和通信硬件。

### 桶大小的权衡

| 桶大小 | 通信次数 | 重叠机会 | 内存开销 |
|--------|---------|---------|---------|
| 小 | 多 | 高 | 低 |
| 大 | 少 | 低 | 高 |
