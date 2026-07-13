"""Assignment 2 Part 1 测试: 分布式通信基准测试

测试基准测试脚本是否能正常运行。
注意：这个测试主要验证代码能跑通，不验证具体性能数值。
"""
import pytest
import torch
import torch.multiprocessing as mp

from .common import _cleanup_process_group, _setup_process_group


def test_benchmark_all_reduce_gloo():
    """测试 Gloo 后端的 all-reduce 基准测试"""
    world_size = 2
    mp.spawn(
        _test_benchmark_all_reduce,
        args=(world_size, "gloo", "cpu"),
        nprocs=world_size,
        join=True,
    )


def _test_benchmark_all_reduce(rank: int, world_size: int, backend: str, device: str):
    """测试 all-reduce 基准测试的核心逻辑"""
    import time
    import torch.distributed as dist

    # 初始化分布式环境
    if device == "cuda":
        _setup_process_group(rank=rank, world_size=world_size, backend=backend)
    else:
        _setup_process_group(rank=rank, world_size=world_size, backend="gloo")

    # 构造测试张量 (1MB)
    tensor_size_mb = 1
    num_elements = (tensor_size_mb * 1024 * 1024) // 4
    tensor = torch.randn(num_elements, device=device if device == "cuda" else "cpu")

    # Warm-up (5 次迭代)
    for _ in range(5):
        dist.all_reduce(tensor)
        if device == "cuda":
            torch.cuda.synchronize()

    # 同步所有进程
    dist.barrier()

    # 正式测试 (20 次迭代)
    num_iterations = 20
    start_time = time.time()

    for _ in range(num_iterations):
        dist.all_reduce(tensor)

    if device == "cuda":
        torch.cuda.synchronize()

    end_time = time.time()

    # 计算性能指标
    total_time = end_time - start_time
    avg_latency = total_time / num_iterations

    # 验证延迟在合理范围内 (应该大于 0)
    assert avg_latency > 0

    if rank == 0:
        print(f"Backend={backend} Device={device} World={world_size} Size={tensor_size_mb}MB: Latency={avg_latency*1000:.3f}ms")

    _cleanup_process_group()
