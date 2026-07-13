"""Part 6 训练基础设施测试"""
import sys, os, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

import torch
import numpy as np
from training import (
    AdamW, get_lr_cosine_schedule, gradient_clipping,
    get_batch, save_checkpoint, load_checkpoint
)


def test_adamw():
    """AdamW: 参数更新 + 权重衰减解耦"""
    model = torch.nn.Linear(3, 2, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.ones(2, 3))

    opt = AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    x = torch.randn(4, 3)
    loss = model(x).sum()
    loss.backward()
    opt.step()

    # 参数应该被更新
    assert not torch.allclose(model.weight, torch.ones(2, 3))
    # 梯度应该被清除
    opt.zero_grad()
    assert model.weight.grad is None

    # 多步测试
    for _ in range(10):
        loss = model(torch.randn(4, 3)).sum()
        loss.backward()
        opt.step()
        opt.zero_grad()

    print("  [PASS] AdamW")


def test_lr_schedule():
    """余弦调度: warmup + 退火 + 边界值"""
    alpha_max, alpha_min = 1e-3, 1e-5
    T_w, T_c = 100, 1000

    # warmup 起点
    assert get_lr_cosine_schedule(0, alpha_max, alpha_min, T_w, T_c) == 0.0
    # warmup 中点
    lr_mid_warmup = get_lr_cosine_schedule(50, alpha_max, alpha_min, T_w, T_c)
    assert abs(lr_mid_warmup - alpha_max / 2) < 1e-10
    # warmup 结束 = 最大值
    lr_warmup_end = get_lr_cosine_schedule(T_w, alpha_max, alpha_min, T_w, T_c)
    assert abs(lr_warmup_end - alpha_max) < 1e-10
    # 退火中点 ≈ (max+min)/2
    lr_mid = get_lr_cosine_schedule((T_w + T_c) // 2, alpha_max, alpha_min, T_w, T_c)
    assert abs(lr_mid - (alpha_max + alpha_min) / 2) < 1e-6
    # 退火结束 = 最小值
    lr_end = get_lr_cosine_schedule(T_c, alpha_max, alpha_min, T_w, T_c)
    assert abs(lr_end - alpha_min) < 1e-10
    # 退火后保持最小值
    lr_after = get_lr_cosine_schedule(T_c + 100, alpha_max, alpha_min, T_w, T_c)
    assert abs(lr_after - alpha_min) < 1e-10

    print("  [PASS] get_lr_cosine_schedule")


def test_gradient_clipping():
    """梯度裁剪: 触发裁剪 + 不触发"""
    p1 = torch.tensor([1.0, 2.0], requires_grad=True)
    p2 = torch.tensor([2.0, 2.0], requires_grad=True)
    p1.grad = torch.tensor([3.0, 0.0])
    p2.grad = torch.tensor([0.0, 4.0])

    # 触发裁剪 (total_norm=5, max_norm=1)
    gradient_clipping([p1, p2], max_norm=1.0)
    new_norm = torch.norm(torch.stack([torch.norm(p1.grad, 2), torch.norm(p2.grad, 2)]), 2)
    assert abs(new_norm.item() - 1.0) < 1e-5

    # 不触发裁剪
    p3 = torch.tensor([0.1, 0.1], requires_grad=True)
    p3.grad = torch.tensor([0.1, 0.2])
    orig = p3.grad.clone()
    gradient_clipping([p3], max_norm=10.0)
    assert torch.equal(p3.grad, orig)

    # 无梯度参数
    p4 = torch.tensor([1.0], requires_grad=True)
    gradient_clipping([p4], max_norm=1.0)  # 不应报错

    print("  [PASS] gradient_clipping")


def test_get_batch():
    """数据加载: 形状 + 偏移关系"""
    data = np.arange(100, dtype=np.int64)
    x, y = get_batch(data, batch_size=4, context_length=10, device='cpu')
    assert x.shape == (4, 10)
    assert y.shape == (4, 10)
    assert x.dtype == torch.long
    assert y.dtype == torch.long
    # y 是 x 右移 1 位
    for i in range(4):
        assert torch.equal(x[i, 1:], y[i, :-1])

    print("  [PASS] get_batch")


def test_checkpoint():
    """检查点: 保存 + 加载 + 状态恢复"""
    model = torch.nn.Linear(3, 2, bias=False)
    opt = AdamW(model.parameters(), lr=1e-3)

    # 触发优化器状态
    loss = model(torch.randn(2, 3)).sum()
    loss.backward()
    opt.step()
    opt.zero_grad()

    path = os.path.join(tempfile.gettempdir(), 'test_ckpt.pt')
    save_checkpoint(model, opt, iteration=42, out=path)

    # 加载到新模型
    model2 = torch.nn.Linear(3, 2, bias=False)
    opt2 = AdamW(model2.parameters(), lr=1e-3)
    it = load_checkpoint(path, model2, opt2)

    assert it == 42
    assert torch.allclose(model.weight, model2.weight)
    os.remove(path)

    print("  [PASS] save/load_checkpoint")


if __name__ == "__main__":
    print("Part 6 训练基础设施测试")
    print("=" * 40)
    test_adamw()
    test_lr_schedule()
    test_gradient_clipping()
    test_get_batch()
    test_checkpoint()
    print("=" * 40)
    print("全部测试通过!")
