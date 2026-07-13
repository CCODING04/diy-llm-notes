"""Assignment 2 适配器接口

这些函数是测试的入口点，需要调用你在 scripts/ 中实现的代码。
每个函数对应一个作业部分：

- Part 2a: get_ddp_individual_parameters, ddp_individual_parameters_on_after_backward
- Part 3:  get_ddp_bucketed, ddp_bucketed_on_after_backward, ddp_bucketed_on_train_batch_start
- Part 4:  get_sharded_optimizer
"""
from __future__ import annotations

from typing import Type

import torch


def get_ddp_individual_parameters(module: torch.nn.Module) -> torch.nn.Module:
    """
    返回一个处理分布式数据并行训练中参数广播和梯度同步的 torch.nn.Module 容器。

    该容器应该通过在反向传播中异步传递就绪的梯度来重叠通信与反向传播计算。
    每个参数张量的梯度单独通信。

    Args:
        module: torch.nn.Module - 要用 DDP 包装的底层模型
    Returns:
        DDP 类的实例
    """
    # TODO: 实现 Part 2a
    # 提示: 从 scripts/naive_ddp.py 导入你的实现
    # from scripts.naive_ddp import NaiveDDP
    # return NaiveDDP(module)
    raise NotImplementedError


def ddp_individual_parameters_on_after_backward(ddp_model: torch.nn.Module, optimizer: torch.optim.Optimizer):
    """
    在完成反向传播之后，但在执行优化器步骤之前运行的代码。

    Args:
        ddp_model: torch.nn.Module - DDP 包装的模型
        optimizer: torch.optim.Optimizer - 与 DDP 包装模型一起使用的优化器
    """
    # TODO: 实现 Part 2a
    # 提示: 等待所有梯度同步完成
    # ddp_model.finish_gradient_synchronization()
    raise NotImplementedError


def get_ddp_bucketed(module: torch.nn.Module, bucket_size_mb: float) -> torch.nn.Module:
    """
    返回一个处理分布式数据并行训练中参数广播和梯度同步的 torch.nn.Module 容器。

    该容器应该通过在反向传播中异步传递就绪的梯度桶来重叠通信与反向传播计算。

    Args:
        module: torch.nn.Module - 要用 DDP 包装的底层模型
        bucket_size_mb: 桶大小，以兆字节为单位
    Returns:
        DDP 类的实例
    """
    # TODO: 实现 Part 3
    # 提示: 从 scripts/ddp_bucketed.py 导入你的实现
    # from scripts.ddp_bucketed import DDPBucketed
    # return DDPBucketed(module, bucket_size_mb)
    raise NotImplementedError


def ddp_bucketed_on_after_backward(ddp_model: torch.nn.Module, optimizer: torch.optim.Optimizer):
    """
    在完成反向传播之后，但在执行优化器步骤之前运行的代码。

    Args:
        ddp_model: torch.nn.Module - DDP 包装的模型
        optimizer: torch.optim.Optimizer - 与 DDP 包装模型一起使用的优化器
    """
    # TODO: 实现 Part 3
    # 提示: 等待所有梯度同步完成
    # ddp_model.finish_gradient_synchronization()
    raise NotImplementedError


def ddp_bucketed_on_train_batch_start(ddp_model: torch.nn.Module, optimizer: torch.optim.Optimizer):
    """
    在训练步骤最开始时运行的代码。

    Args:
        ddp_model: torch.nn.Module - DDP 包装的模型
        optimizer: torch.optim.Optimizer - 与 DDP 包装模型一起使用的优化器
    """
    # TODO: 实现 Part 3
    # 提示: 如果需要，在每个训练步骤开始时执行某些操作
    pass


def get_sharded_optimizer(params, optimizer_cls: Type[torch.optim.Optimizer], **kwargs) -> torch.optim.Optimizer:
    """
    返回一个处理给定优化器类在提供的参数上优化器状态分片的 torch.optim.Optimizer。

    Arguments:
        params (``Iterable``): 一个包含所有参数的 ``Iterable``
        optimizer_class (:class:`torch.nn.Optimizer`): 本地优化器的类
    Keyword arguments:
        kwargs: 要转发给优化器构造器的关键字参数
    Returns:
        分片优化器的实例
    """
    # TODO: 实现 Part 4
    # 提示: 从 scripts/sharded_optimizer.py 导入你的实现
    # from scripts.sharded_optimizer import ShardedOptimizer
    # return ShardedOptimizer(params, optimizer_cls, **kwargs)
    raise NotImplementedError
