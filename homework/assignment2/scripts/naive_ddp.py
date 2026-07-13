"""Assignment 2 Part 2: 朴素 DDP 实现

通过在反向传播后对各参数梯度单独进行 all-reduce，实现分布式数据并行训练。
"""
import os
import time
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.multiprocessing import spawn


# ============================================================
# 1. 简单模型定义（用于验证 DDP 正确性）
# ============================================================
class SimpleModel(nn.Module):
    """简单的三层全连接网络"""
    def __init__(self, input_dim=784, hidden_dim=256, output_dim=10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


# ============================================================
# 2. 分布式环境
# ============================================================
def init_distributed(rank: int, world_size: int, backend: str):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)


def destroy_distributed():
    dist.destroy_process_group()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ============================================================
# 3. 朴素 DDP 训练
# ============================================================
def naive_ddp_train(rank: int, world_size: int, backend: str):
    """
    朴素 DDP 训练流程：
    1. 广播参数（确保初始状态一致）
    2. 前向传播
    3. 反向传播
    4. 逐参数 all-reduce 梯度
    5. 优化器更新
    """
    # 固定随机种子
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    # 初始化分布式环境
    init_distributed(rank, world_size, backend)

    # 设置设备
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")
    else:
        device = torch.device("cpu")

    # 创建模型和优化器
    model = SimpleModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # TODO: 步骤 1 - 广播参数
    # 确保所有 rank 从相同的初始参数开始
    for param in model.parameters():
        dist.broadcast(param.data, src=0)

    model.train()

    # 模拟训练数据
    batch_size = 32
    num_steps = 100

    for step in range(num_steps):
        # TODO: 生成随机训练数据
        x = torch.randn(batch_size, 784, device=device)
        y = torch.randint(0, 10, (batch_size,), device=device)

        # TODO: 步骤 2 - 前向传播
        optimizer.zero_grad()
        logits = model(x)
        loss = nn.functional.cross_entropy(logits, y)

        # TODO: 步骤 3 - 反向传播
        loss.backward()

        # TODO: 步骤 4 - 逐参数 all-reduce 梯度
        # 将所有 rank 的梯度求和，然后除以 world_size 取平均
        for param in model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                param.grad.data /= world_size

        # TODO: 步骤 5 - 优化器更新
        optimizer.step()

        if step % 20 == 0 and rank == 0:
            print(f"Step {step}: loss={loss.item():.4f}")

    # 保存模型（仅 rank 0）
    if rank == 0:
        torch.save(model.state_dict(), "naive_ddp_model.pt")
        print("模型已保存")

    destroy_distributed()


def main():
    world_size = 2  # 根据 GPU 数量调整
    backend = "nccl" if torch.cuda.is_available() else "gloo"

    print(f"启动朴素 DDP 训练，world_size={world_size}")
    spawn(naive_ddp_train, args=(world_size, backend), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()
