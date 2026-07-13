"""Assignment 2 Part 4: 优化器状态分片

简化版 ZeRO-1：每个 rank 只负责部分参数的优化器状态。
"""
from typing import Any, Type, Iterable, Dict, List
import torch
import torch.distributed as dist
from torch.optim import Optimizer


class ShardedOptimizer(Optimizer):
    """
    分片优化器（简化版 ZeRO-1）

    核心思想：
    - 所有 rank 共享完整的模型参数
    - 每个 rank 只负责 1/world_size 的优化器状态
    - step 后通过 broadcast 同步参数

    内存节省：
    - AdamW: 每个参数需要 m, v 两个状态
    - 分片后: 每个 rank 只保存 1/world_size 的 m, v
    """

    def __init__(
        self,
        params: Iterable,
        optimizer_cls: Type[Optimizer],
        **kwargs: Any,
    ):
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

    def _build_local_optimizer(self):
        """
        构建当前 rank 的本地优化器

        只包含本 rank 负责的参数
        """
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

    def _param_owner(self, param: torch.nn.Parameter) -> int:
        """
        确定参数属于哪个 rank

        策略：param_index % world_size
        """
        return self._global_param_index(param) % self.world_size

    def _global_param_index(self, param: torch.nn.Parameter) -> int:
        """
        为每个参数分配全局索引

        保证所有 rank 上索引一致
        """
        # TODO: 建立参数到索引的映射（懒初始化）
        if not hasattr(self, "_param_to_index"):
            self._param_to_index = {}
            idx = 0
            for group in self.param_groups:
                for p in group["params"]:
                    self._param_to_index[p] = idx
                    idx += 1
        return self._param_to_index[param]

    @torch.no_grad()
    def step(self, closure=None, **kwargs):
        """
        执行一次优化步骤

        流程：
        1. （可选）执行 closure
        2. 本地优化器更新参数
        3. 广播更新后的参数
        """
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
        """
        通过 broadcast 同步参数

        owner rank 广播参数到其他 rank
        """
        # TODO: 对每个参数执行 broadcast
        for group in self.param_groups:
            for p in group["params"]:
                owner = self._param_owner(p)
                dist.broadcast(p.data, src=owner)

    def add_param_group(self, param_group: Dict[str, Any]):
        """
        动态添加参数组

        注意：需要重建本地优化器
        """
        super().add_param_group(param_group)
        # TODO: 重建本地优化器
        self._build_local_optimizer()


# ============================================================
# 测试代码
# ============================================================
if __name__ == "__main__":
    import torch.nn as nn

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    device = torch.device(f"cuda:{rank}")

    model = nn.Sequential(
        nn.Linear(1024, 1024),
        nn.ReLU(),
        nn.Linear(1024, 10)
    ).to(device)

    # 广播初始参数
    for p in model.parameters():
        dist.broadcast(p.data, src=0)

    # 使用分片优化器
    optimizer = ShardedOptimizer(
        model.parameters(),
        torch.optim.AdamW,
        lr=1e-3
    )

    loss_fn = nn.CrossEntropyLoss()

    for step in range(5):
        x = torch.randn(32, 1024, device=device)
        y = torch.randint(0, 10, (32,), device=device)

        optimizer.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()

        if rank == 0:
            print(f"Step {step}: loss={loss.item():.4f}")

    dist.destroy_process_group()
