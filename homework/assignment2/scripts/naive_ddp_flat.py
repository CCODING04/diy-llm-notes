"""Assignment 2 Part 2c: 梯度展平 DDP

将所有梯度展平为一个张量后单次 all-reduce，与逐参数通信对比性能。
"""
import os
import time
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.multiprocessing import spawn

from naive_ddp import SimpleModel, init_distributed, destroy_distributed


def flat_ddp_train(rank: int, world_size: int, backend: str):
    """
    梯度展平 DDP 训练：
    1. 广播参数
    2. 前向 + 反向传播
    3. 展平所有梯度为一个张量
    4. 单次 all-reduce
    5. 写回各参数梯度
    6. 优化器更新
    """
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    init_distributed(rank, world_size, backend)

    if torch.cuda.is_available():
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")
    else:
        device = torch.device("cpu")

    model = SimpleModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # 广播参数
    for param in model.parameters():
        dist.broadcast(param.data, src=0)

    # TODO: 记录每个参数的形状和大小（用于展平和还原）
    grad_shapes = [p.shape for p in model.parameters()]
    grad_numels = [p.numel() for p in model.parameters()]
    total_numel = sum(grad_numels)

    model.train()
    batch_size = 32
    num_steps = 100

    for step in range(num_steps):
        x = torch.randn(batch_size, 784, device=device)
        y = torch.randint(0, 10, (batch_size,), device=device)

        optimizer.zero_grad()
        logits = model(x)
        loss = nn.functional.cross_entropy(logits, y)
        loss.backward()

        # TODO: 步骤 1 - 展平所有梯度
        flat_grad = torch.zeros(total_numel, device=device)
        offset = 0
        for p in model.parameters():
            if p.grad is not None:
                numel = p.numel()
                flat_grad[offset:offset + numel] = p.grad.view(-1)
                offset += numel

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

        if step % 20 == 0 and rank == 0:
            print(f"Step {step}: loss={loss.item():.4f}")

    if rank == 0:
        print("梯度展平 DDP 训练完成")

    destroy_distributed()


def main():
    world_size = 2
    backend = "nccl" if torch.cuda.is_available() else "gloo"

    print(f"启动梯度展平 DDP 训练，world_size={world_size}")
    spawn(flat_ddp_train, args=(world_size, backend), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()
