"""Part 4 基础算子测试"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

import torch
import numpy as np
from model_components import (
    Linear, Embedding, RMSNorm, SwiGLU,
    RotaryPositionalEmbedding, softmax, cross_entropy
)


def test_linear():
    """Linear: 形状正确 + 手动 matmul 一致 + 无 bias"""
    model = Linear(6, 3)
    assert model.weight.shape == (3, 6), f"权重形状错误: {model.weight.shape}"

    test_w = torch.randn(3, 6)
    model.load_state_dict({'weight': test_w})
    x = torch.randn(4, 6)
    out = model(x)

    assert out.shape == (4, 3)
    assert torch.allclose(out, x @ test_w.t(), atol=1e-5)

    # 检查没有 bias
    assert not hasattr(model, 'bias') or model.bias is None
    print("  [PASS] Linear")


def test_embedding():
    """Embedding: 正确的行索引"""
    w = torch.randn(10, 3)
    model = Embedding(10, 3)
    model.load_state_dict({'weights': w})

    ids = torch.tensor([[2, 9, 5], [3, 2, 6]])
    out = model(ids)
    assert out.shape == (2, 3, 3)
    assert torch.equal(out[0, 0], w[2])
    assert torch.equal(out[1, 2], w[6])
    print("  [PASS] Embedding")


def test_rmsnorm():
    """RMSNorm: 归一化后方差≈1 + 可学习 weight 生效"""
    d_model = 16
    norm = RMSNorm(d_model, eps=1e-5)
    x = torch.randn(4, 8, d_model)
    out = norm(x)

    assert out.shape == x.shape
    rms_per_row = out.float().pow(2).mean(-1)
    assert torch.allclose(rms_per_row, torch.ones_like(rms_per_row), atol=1e-4)

    # 验证 weight 生效
    norm2 = RMSNorm(d_model, eps=1e-5)
    norm2.weight.data.fill_(2.0)
    out2 = norm2(x)
    rms_per_row2 = out2.float().pow(2).mean(-1)
    assert torch.allclose(rms_per_row2, 4.0 * torch.ones_like(rms_per_row2), atol=1e-3)
    print("  [PASS] RMSNorm")


def test_swiglu():
    """SwiGLU: 输出形状正确 + d_ff 对齐"""
    d_model = 64
    swiglu = SwiGLU(d_model)
    x = torch.randn(2, 10, d_model)
    out = swiglu(x)

    assert out.shape == x.shape

    expected_d_ff = int(8/3 * d_model)
    expected_d_ff = (expected_d_ff + 63) // 64 * 64
    assert swiglu.w_gate.out_features == expected_d_ff
    assert swiglu.w_gate.out_features % 64 == 0
    print("  [PASS] SwiGLU")


def test_rope():
    """RoPE: 输出形状不变 + 位置 0 不变 + 缓存机制"""
    d_k = 8
    rope = RotaryPositionalEmbedding(theta=10000.0, d_k=d_k)
    x = torch.randn(1, 1, 5, d_k)
    out = rope(x)

    assert out.shape == x.shape
    assert torch.allclose(out[:, :, 0, :], x[:, :, 0, :], atol=1e-5)

    # 缓存测试：重复调用不应崩溃
    out2 = rope(x)
    print("  [PASS] RoPE")


def test_softmax():
    """softmax: 概率分布 + 数值稳定"""
    x = torch.tensor([[1.0, 2.0, 3.0], [1.0, 2.0, 5.0]])
    out = softmax(x, dim=-1)
    assert torch.allclose(out.sum(-1), torch.ones(2), atol=1e-5)
    assert (out >= 0).all()

    # 大数值输入——验证数值稳定
    x_big = torch.tensor([[1.0, 2.0, 1000.0]])
    out_big = softmax(x_big, dim=-1)
    assert not torch.isnan(out_big).any()
    assert not torch.isinf(out_big).any()
    assert torch.allclose(out_big.sum(-1), torch.tensor([1.0]), atol=1e-5)
    assert out_big[0, 2] > 0.999

    print("  [PASS] softmax")


def test_cross_entropy():
    """cross_entropy: 与 torch 实现一致"""
    logits = np.array([[2.0, 1.0, 0.1], [0.5, 2.5, 0.1], [0.1, 0.1, 5.0]])
    targets = np.array([0, 1, 2])
    loss = cross_entropy(logits, targets)

    t_loss = torch.nn.functional.cross_entropy(
        torch.tensor(logits), torch.tensor(targets, dtype=torch.long)
    )
    assert np.allclose(loss, t_loss.item(), atol=1e-5), f"{loss} vs {t_loss.item()}"
    assert loss < 1.0
    print(f"  [PASS] cross_entropy (loss={loss:.4f})")


if __name__ == "__main__":
    print("Part 4 基础算子测试")
    print("=" * 40)
    test_linear()
    test_embedding()
    test_rmsnorm()
    test_swiglu()
    test_rope()
    test_softmax()
    test_cross_entropy()
    print("=" * 40)
    print("全部测试通过!")
