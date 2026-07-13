"""Assignment 2 Part 1: 分布式通信基准测试

测量 all-reduce 操作在不同后端、数据规模、进程数下的性能。
"""
import os
import time
import argparse
import torch
import torch.distributed as dist
from torch.multiprocessing import spawn


def init_distributed(rank: int, world_size: int, backend: str):
    """初始化分布式进程组"""
    # TODO: 设置 MASTER_ADDR 和 MASTER_PORT
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"

    # TODO: 初始化进程组
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)


def destroy_distributed():
    """清理分布式环境"""
    # TODO: 销毁进程组
    dist.destroy_process_group()

    # TODO: 清空 CUDA 缓存（如果可用）
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def benchmark_all_reduce(
    rank: int,
    world_size: int,
    tensor_size_mb: int,
    backend: str,
    device: str,
):
    """
    对 all-reduce 进行基准测试

    流程：
    1. 初始化分布式环境
    2. 构造测试张量
    3. Warm-up（5 次）
    4. 正式测试（20 次）
    5. 计算延迟和带宽
    6. 清理环境
    """
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
    num_iterations = 20
    start_time = time.time()

    for _ in range(num_iterations):
        dist.all_reduce(tensor)

    # TODO: GPU 场景下等待所有 kernel 完成
    if device == "cuda":
        torch.cuda.synchronize()

    end_time = time.time()

    # TODO: 计算性能指标
    total_time = end_time - start_time
    avg_latency = total_time / num_iterations
    bandwidth_gbps = (tensor_size_mb * 1024 * 1024) / avg_latency / 1e9

    # TODO: 只让 rank 0 打印结果
    if rank == 0:
        print(
            f"Backend={backend:<5} Device={device:<4} World={world_size} "
            f"Size={tensor_size_mb:>4}MB: "
            f"Latency={avg_latency*1000:.3f}ms  Bandwidth={bandwidth_gbps:.2f}GB/s"
        )

    # TODO: 清理环境
    destroy_distributed()


def main():
    """遍历所有配置进行测试"""
    parser = argparse.ArgumentParser(description="All-Reduce Benchmark")
    parser.add_argument("--sizes", type=int, nargs="+", default=[1, 10, 100, 1000],
                        help="张量大小 (MB)")
    parser.add_argument("--world_sizes", type=int, nargs="+", default=[2, 4],
                        help="进程数量")
    args = parser.parse_args()

    # 检查可用 GPU 数量
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    print(f"可用 GPU 数量: {num_gpus}")

    # 测试 NCCL + GPU
    if num_gpus > 0:
        print("\n=== NCCL + GPU ===")
        for size in args.sizes:
            for ws in args.world_sizes:
                if ws <= num_gpus:
                    spawn(
                        benchmark_all_reduce,
                        args=(ws, size, "nccl", "cuda"),
                        nprocs=ws, join=True
                    )

    # 测试 Gloo + CPU
    print("\n=== Gloo + CPU ===")
    for size in args.sizes:
        for ws in args.world_sizes:
            spawn(
                benchmark_all_reduce,
                args=(ws, size, "gloo", "cpu"),
                nprocs=ws, join=True
            )


if __name__ == "__main__":
    main()
