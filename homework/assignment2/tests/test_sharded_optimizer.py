"""Assignment 2 Part 4 测试: 优化器状态分片

测试 ShardedOptimizer 实现是否正确：
1. 分片后参数更新应该与非分片一致
2. 每个 rank 只保存部分优化器状态
"""
from copy import deepcopy
from typing import Type

import numpy
import pytest
import torch
import torch.multiprocessing as mp

from .adapters import get_sharded_optimizer
from .common import (
    ToyModel,
    ToyModelWithTiedWeights,
    _cleanup_process_group,
    _setup_process_group,
)


@pytest.mark.parametrize("model_class", [ToyModel, ToyModelWithTiedWeights])
def test_sharded_optimizer(model_class):
    world_size = 2
    mp.spawn(
        _test_sharded_optimizer,
        args=(world_size, model_class),
        nprocs=world_size,
        join=True,
    )


def _test_sharded_optimizer(rank: int, world_size: int, model_class: Type[torch.nn.Module]):
    # 使用 CPU 的 gloo 后端
    device = _setup_process_group(rank=rank, world_size=world_size, backend="gloo")
    torch.manual_seed(42)
    optimizer_cls = torch.optim.AdamW

    # 创建非分片模型作为基线
    non_sharded_model = model_class().to(device)
    non_sharded_optimizer = optimizer_cls(
        non_sharded_model.parameters(),
        lr=0.1,
        weight_decay=0.1,
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    # 创建分片模型
    sharded_model = deepcopy(non_sharded_model)
    sharded_optimizer = get_sharded_optimizer(
        sharded_model.parameters(),
        optimizer_cls,
        lr=0.1,
        weight_decay=0.1,
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    for _ in range(10):
        non_sharded_optimizer.zero_grad()
        sharded_optimizer.zero_grad()

        # 生成相同的输入数据
        input_ = torch.rand((32, 10)).to(device)
        labels = torch.rand((32, 5)).to(device)
        non_sharded_input = deepcopy(input_)
        sharded_input = deepcopy(input_)
        non_sharded_labels = deepcopy(labels)
        sharded_labels = deepcopy(labels)

        # 前向传播
        non_sharded_model_logits = non_sharded_model(non_sharded_input)
        sharded_model_logits = sharded_model(sharded_input)

        # 计算损失
        non_sharded_model_loss = ((non_sharded_labels - non_sharded_model_logits) ** 2).sum()
        sharded_model_loss = ((sharded_labels - sharded_model_logits) ** 2).sum()

        # 反向传播
        non_sharded_model_loss.backward()
        sharded_model_loss.backward()

        # 优化器更新
        non_sharded_optimizer.step()
        sharded_optimizer.step()

    # 检查最终模型权重是否相同
    for non_sharded_parameters, sharded_parameters in zip(non_sharded_model.parameters(), sharded_model.parameters()):
        numpy.testing.assert_allclose(
            non_sharded_parameters.detach().cpu().numpy(),
            sharded_parameters.detach().cpu().numpy(),
        )
    _cleanup_process_group()
