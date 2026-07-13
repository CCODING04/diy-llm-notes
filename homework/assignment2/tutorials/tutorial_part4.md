# Assignment 2 - Part 4：优化器状态分片

> 📍 作业进度：Assignment 2，第 4 / 4 部分
> 📅 生成时间：2026-07-08
> 📎 原作业：`coursework/assignment2-systems/cs336_systems/作业2.ipynb`

---

## 目标与要求

### 问题 4：优化器状态分片（15 分）

实现一个 Python 类，用于处理优化器状态的分片（简化版 ZeRO-1）。

**核心思想**：
- 所有 rank 共享完整的模型参数
- 每个 rank 只负责 1/world_size 的优化器状态
- step 后通过 broadcast 同步参数

---

## 实现步骤

### 脚本框架

编辑 `scripts/sharded_optimizer.py`，文件已包含完整的 `ShardedOptimizer` 类框架和 TODO 标记。

### Step 1：理解 ZeRO-1

```
┌─────────────────────────────────────────────────────────────┐
│                    标准 DDP                                  │
├─────────────────────────────────────────────────────────────┤
│  Rank 0: [参数] [梯度] [优化器状态 m,v]                      │
│  Rank 1: [参数] [梯度] [优化器状态 m,v]                      │
│  Rank 2: [参数] [梯度] [优化器状态 m,v]                      │
│  Rank 3: [参数] [梯度] [优化器状态 m,v]                      │
│                                                             │
│  内存：4 × (参数 + 梯度 + 优化器状态)                         │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                    ZeRO-1（优化器状态分片）                    │
├─────────────────────────────────────────────────────────────┤
│  Rank 0: [参数] [梯度] [m,v for params 0,4,8,...]           │
│  Rank 1: [参数] [梯度] [m,v for params 1,5,9,...]           │
│  Rank 2: [参数] [梯度] [m,v for params 2,6,10,...]          │
│  Rank 3: [参数] [梯度] [m,v for params 3,7,11,...]          │
│                                                             │
│  内存：4 × (参数 + 梯度) + 1 × 优化器状态                    │
│  节省：优化器状态内存降低到 1/world_size                      │
└─────────────────────────────────────────────────────────────┘
```

### Step 2：构造函数

```python
class ShardedOptimizer(Optimizer):
    def __init__(self, params: Iterable, optimizer_cls: Type[Optimizer], **kwargs):
        # TODO: 检查分布式环境已初始化
        if not dist.is_initialized():
            raise RuntimeError("torch.distributed must be initialized")

        # TODO: 获取当前 rank 和 world_size
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()

        # 保存优化器类型和参数
        self.optimizer_cls = optimizer_cls
        self.optimizer_kwargs = kwargs

        # TODO: 调用父类构造函数（会触发 add_param_group）
        super().__init__(params, defaults={})

        # TODO: 构建本地优化器
        self._build_local_optimizer()
```

### Step 3：构建本地优化器

```python
def _build_local_optimizer(self):
    """构建只包含本 rank 负责参数的优化器"""
    local_param_groups: List[Dict[str, Any]] = []

    for group in self.param_groups:
        # TODO: 筛选属于当前 rank 的参数
        local_params = [
            p for p in group["params"]
            if self._param_owner(p) == self.rank
        ]

        if len(local_params) == 0:
            continue

        # 复制参数组配置
        local_group = dict(group)
        local_group["params"] = local_params
        local_param_groups.append(local_group)

    # TODO: 创建本地优化器
    self.local_optimizer = self.optimizer_cls(
        local_param_groups,
        **self.optimizer_kwargs
    )
```

### Step 4：参数分片策略

```python
def _param_owner(self, param: torch.nn.Parameter) -> int:
    """确定参数属于哪个 rank"""
    return self._global_param_index(param) % self.world_size

def _global_param_index(self, param: torch.nn.Parameter) -> int:
    """为每个参数分配全局索引（保证所有 rank 上索引一致）"""
    # TODO: 建立参数到索引的映射（懒初始化）
    if not hasattr(self, "_param_to_index"):
        self._param_to_index = {}
        idx = 0
        for group in self.param_groups:
            for p in group["params"]:
                self._param_to_index[p] = idx
                idx += 1
    return self._param_to_index[param]
```

### Step 5：Step 和参数同步

```python
@torch.no_grad()
def step(self, closure=None, **kwargs):
    loss = None
    if closure is not None:
        with torch.enable_grad():
            loss = closure()

    # TODO: 本地优化器更新（只更新本 rank 拥有的参数）
    self.local_optimizer.step(**kwargs)

    # TODO: 同步参数到所有 rank
    self._sync_parameters()

    return loss

def _sync_parameters(self):
    """通过 broadcast 同步参数（owner rank 广播到其他 rank）"""
    # TODO: 对每个参数执行 broadcast
    for group in self.param_groups:
        for p in group["params"]:
            owner = self._param_owner(p)
            dist.broadcast(p.data, src=owner)
```

### Step 6：适配器接口

在 `tests/adapters.py` 中实现测试接口：

```python
def get_sharded_optimizer(params, optimizer_cls: Type[torch.optim.Optimizer], **kwargs):
    # TODO: 返回你的 ShardedOptimizer 实例
    from scripts.sharded_optimizer import ShardedOptimizer
    return ShardedOptimizer(params, optimizer_cls, **kwargs)
```

---

## 测试方法

```bash
cd homework/assignment2

# 运行测试（需要 2+ GPU 或 CPU gloo）
pytest tests/test_sharded_optimizer.py -v

# 直接运行脚本
python scripts/sharded_optimizer.py
```

---

## 难点与注意事项

| # | 难点 | 解决方案 |
|---|------|---------|
| 1 | 参数遍历顺序必须一致 | 所有 rank 使用相同的模型结构和遍历方式 |
| 2 | `add_param_group` 被多次调用 | 在 `_build_local_optimizer` 中重建本地优化器 |
| 3 | 优化器状态不在所有 rank 上 | 只有 owner rank 有 m/v，其他 rank 的优化器状态为空 |
| 4 | 测试需要多 GPU | 使用 CPU gloo 后端测试 |

---

## 关键概念

### AdamW 的内存开销

对于每个参数，AdamW 维护：
- `m`（一阶矩）：与参数同大小
- `v`（二阶矩）：与参数同大小

总内存 = 参数 + 梯度 + m + v = 4 × 参数大小

ZeRO-1 将 m + v 分片到不同 rank，每个 rank 只保存 1/world_size 的优化器状态。

### 分片 vs 复制

| 方式 | 参数 | 梯度 | 优化器状态 | 通信 |
|------|------|------|-----------|------|
| DDP | 完整副本 | 完整副本 | 完整副本 | 梯度 all-reduce |
| ZeRO-1 | 完整副本 | 完整副本 | 分片 | 参数 broadcast |
| ZeRO-2 | 完整副本 | 分片 | 分片 | 梯度 reduce-scatter + all-gather |
| ZeRO-3 | 分片 | 分片 | 分片 | 参数 all-gather |

### 为什么用 broadcast 而不是 all-reduce？

- DDP：所有 rank 都计算梯度，需要 all-reduce 求平均
- ZeRO-1：只有 owner rank 更新参数，只需要 broadcast 给其他 rank

### 内存节省计算

假设模型有 1B 参数，使用 AdamW：
- 参数：1B × 4 bytes = 4 GB
- 梯度：4 GB
- 优化器状态：8 GB (m + v)
- **总计**：16 GB / rank

使用 ZeRO-1 (world_size=4)：
- 参数：4 GB
- 梯度：4 GB
- 优化器状态：2 GB (8/4)
- **总计**：10 GB / rank → 节省 37.5%
