"""Assignment 2 Part 3: DDP 计算通信重叠

通过梯度分桶和异步通信实现计算与通信的重叠。
"""
import os
import torch
import torch.nn as nn
import torch.distributed as dist


class DDPBucketed(nn.Module):
    """
    分桶 DDP 容器类

    核心特性：
    - 梯度分桶：按大小将梯度分组
    - 异步通信：梯度准备好后立即启动 all-reduce
    - 计算通信重叠：反向传播和通信同时进行
    """

    def __init__(self, module: nn.Module, bucket_size_mb: float):
        super().__init__()
        self.module = module
        self.bucket_size_bytes = int(bucket_size_mb * 1024 * 1024)
        self.world_size = dist.get_world_size()
        self.buckets = []
        self.handles = []

        # 初始化步骤
        self._broadcast_parameters()
        self._create_buckets()
        self._register_hooks()

    def _broadcast_parameters(self):
        """将 rank 0 的参数广播到所有进程"""
        # TODO: 广播所有参数
        if self.world_size > 1:
            for param in self.module.parameters():
                dist.broadcast(param.data, src=0)

    def _create_buckets(self):
        """
        按大小创建梯度桶

        注意：倒序遍历参数，因为反向传播从最后一层开始
        """
        # TODO: 倒序遍历参数，按 bucket_size_bytes 分组
        current_bucket = []
        current_size = 0

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

    def _finalize_bucket(self, params):
        """为桶创建缓冲区和状态"""
        if not params:
            return

        buffer_size = sum(p.numel() for p in params)
        buffer = torch.zeros(
            buffer_size,
            device=params[0].device,
            dtype=params[0].dtype
        )

        self.buckets.append({
            "params": params,
            "buffer": buffer,
            "ready_count": 0,
            "triggered": False,
            "total_params": len(params)
        })

    def _register_hooks(self):
        """为每个参数注册梯度钩子"""
        # TODO: 为每个参数注册钩子，梯度准备好时调用 _on_gradient_ready
        for bucket_idx, bucket in enumerate(self.buckets):
            for param in bucket["params"]:
                param.register_hook(
                    lambda grad, b_idx=bucket_idx: self._on_gradient_ready(b_idx)
                )

    def _on_gradient_ready(self, bucket_idx: int):
        """
        梯度就绪回调

        当桶内所有参数的梯度都准备好时，启动异步 all-reduce
        """
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

    def forward(self, *args, **kwargs):
        """前向传播，重置桶状态"""
        # TODO: 重置所有桶的状态
        for bucket in self.buckets:
            bucket["triggered"] = False
            bucket["ready_count"] = 0

        self.handles.clear()
        return self.module(*args, **kwargs)

    def finish_gradient_synchronization(self):
        """
        等待所有异步通信完成，写回梯度

        必须在 optimizer.step() 之前调用
        """
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


# ============================================================
# 测试代码
# ============================================================
if __name__ == "__main__":
    import torch.optim as optim

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    model = nn.Linear(1024, 10).to(device)
    ddp_model = DDPBucketed(model, bucket_size_mb=1)
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.01)

    for step in range(5):
        x = torch.randn(32, 1024, device=device)
        loss = ddp_model(x).sum()
        loss.backward()
        ddp_model.finish_gradient_synchronization()
        optimizer.step()
        optimizer.zero_grad()

        if rank == 0:
            print(f"Step {step}: loss={loss.item():.4f}")

    dist.destroy_process_group()
