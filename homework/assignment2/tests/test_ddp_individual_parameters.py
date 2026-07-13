"""Assignment 2 Part 2a 测试: 朴素 DDP（逐参数通信）

测试 DDP 实现是否正确：
1. 参数广播：所有 rank 的模型参数应该一致
2. 梯度同步：all-reduce 后梯度应该取平均
3. 训练等价性：DDP 模型和非并行模型应该产生相同的参数更新
"""
import logging
from copy import deepcopy
from typing import Type

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim

from .adapters import (
    ddp_individual_parameters_on_after_backward,
    get_ddp_individual_parameters,
)
from .common import (
    FIXTURES_PATH,
    ToyModel,
    ToyModelWithTiedWeights,
    _cleanup_process_group,
    _setup_process_group,
    validate_ddp_net_equivalence,
)

logger = logging.getLogger(__name__)


@pytest.mark.parametrize("model_class", [ToyModel, ToyModelWithTiedWeights])
def test_DistributedDataParallelIndividualParameters(model_class):
    world_size = 2
    mp.spawn(
        _test_DistributedDataParallelIndividualParameters,
        args=(world_size, model_class),
        nprocs=world_size,
        join=True,
    )


def _test_DistributedDataParallelIndividualParameters(rank: int, world_size: int, model_class: Type[torch.nn.Module]):
    # 使用 CPU 的 gloo 后端
    device = _setup_process_group(rank=rank, world_size=world_size, backend="gloo")
    dist.barrier()

    # 设置种子以确保 rank 使用不同的初始模型进行初始化
    torch.manual_seed(rank)

    # 创建一个玩具模型并将其移动到适当的设备
    non_parallel_model = model_class().to(device)

    # 创建一个 DDP 模型
    ddp_base = deepcopy(non_parallel_model)
    ddp_model = get_ddp_individual_parameters(ddp_base)

    # 检查参数是否正确广播
    for (non_parallel_param_name, non_parallel_model_parameter), (
        ddp_model_param_name,
        ddp_model_parameter,
    ) in zip(non_parallel_model.named_parameters(), ddp_model.named_parameters()):
        is_no_grad_fixed_param = (
            "no_grad_fixed_param" in ddp_model_param_name or "no_grad_fixed_param" in non_parallel_param_name
        )
        if rank == 0 or is_no_grad_fixed_param:
            assert torch.allclose(non_parallel_model_parameter, ddp_model_parameter)
        else:
            assert not torch.allclose(non_parallel_model_parameter, ddp_model_parameter)

    # 确保所有 rank 具有相同的模型状态
    validate_ddp_net_equivalence(ddp_model)

    # 加载测试数据
    all_x = torch.load(FIXTURES_PATH / "ddp_test_data.pt")
    all_y = torch.load(FIXTURES_PATH / "ddp_test_labels.pt")

    assert all_x.size(0) % world_size == 0
    local_bs = int(all_y.size(0) / world_size)

    loss_fn = nn.MSELoss()
    ddp_optimizer = optim.SGD(ddp_model.parameters(), lr=0.1)
    non_parallel_optimizer = optim.SGD(non_parallel_model.parameters(), lr=0.1)

    for i in range(5):
        ddp_optimizer.zero_grad()
        non_parallel_optimizer.zero_grad()

        # 非并行模型在所有数据上训练
        non_parallel_data = all_x.to(device)
        non_parallel_labels = all_y.to(device)
        non_parallel_outputs = non_parallel_model(non_parallel_data)
        non_parallel_loss = loss_fn(non_parallel_outputs, non_parallel_labels)
        non_parallel_loss.backward()
        non_parallel_optimizer.step()

        # DDP 模型只在部分数据上训练
        offset = rank * local_bs
        ddp_data = all_x[offset : offset + local_bs, :].to(device)
        ddp_labels = all_y[offset : offset + local_bs, :].to(device)
        ddp_outputs = ddp_model(ddp_data)
        ddp_loss = loss_fn(ddp_outputs, ddp_labels)
        ddp_loss.backward()

        # 运行梯度同步
        ddp_individual_parameters_on_after_backward(ddp_model, ddp_optimizer)
        ddp_optimizer.step()

        # 检查参数是否一致
        if rank == 0:
            for non_parallel_model_parameter, ddp_model_parameter in zip(
                non_parallel_model.parameters(), ddp_model.parameters()
            ):
                assert torch.allclose(non_parallel_model_parameter, ddp_model_parameter)

        # 打乱数据
        torch.manual_seed(42 + i)
        shuffle_idxs = torch.randperm(all_x.size(0))
        all_x = all_x[shuffle_idxs]
        all_y = all_y[shuffle_idxs]

    # 训练结束后检查最终参数是否一致
    if rank == 0:
        for non_parallel_model_parameter, ddp_model_parameter in zip(
            non_parallel_model.parameters(), ddp_model.parameters()
        ):
            assert torch.allclose(non_parallel_model_parameter, ddp_model_parameter)
    _cleanup_process_group()
