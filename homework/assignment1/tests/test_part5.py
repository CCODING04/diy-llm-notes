"""Part 5 注意力机制与 Transformer Block 测试"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

import torch
import math
from model_components import (
    scaled_dot_product_attention, MultiHeadAttention, TransformerBlock
)


def test_scaled_dot_product_attention():
    """SDPA: 形状正确 + 概率和为 1 + mask 生效"""
    q = torch.randn(2, 3, 8)
    k = torch.randn(2, 3, 8)
    v = torch.randn(2, 3, 8)

    out, weights = scaled_dot_product_attention(q, k, v)
    assert out.shape == (2, 3, 8), f"输出形状错误: {out.shape}"
    assert weights.shape == (2, 3, 3), f"权重形状错误: {weights.shape}"
    assert torch.allclose(weights.sum(-1), torch.ones(2, 3), atol=1e-5)

    # 因果 mask
    mask = torch.tril(torch.ones(3, 3)).bool()
    out_m, weights_m = scaled_dot_product_attention(q, k, v, mask=mask)
    assert out_m.shape == (2, 3, 8)
    # 第一个位置只能关注自己
    assert torch.allclose(weights_m[:, 0, 0], torch.ones(2), atol=1e-5)
    assert (weights_m[:, 0, 1:] == 0).all()

    # 4D 输入（含 head 维度）
    q4 = torch.randn(2, 4, 3, 8)
    k4 = torch.randn(2, 4, 3, 8)
    v4 = torch.randn(2, 4, 3, 8)
    out4, _ = scaled_dot_product_attention(q4, k4, v4)
    assert out4.shape == (2, 4, 3, 8)

    # 数值稳定：大输入
    q_big = torch.randn(1, 3, 64) * 100
    k_big = torch.randn(1, 3, 64) * 100
    v_big = torch.randn(1, 3, 64)
    out_big, w_big = scaled_dot_product_attention(q_big, k_big, v_big)
    assert not torch.isnan(out_big).any()
    assert not torch.isinf(out_big).any()

    print("  [PASS] scaled_dot_product_attention")


def test_multihead_attention():
    """MHA: 形状正确 + mask 生效 + RoPE 作用于 Q/K"""
    d_model = 64
    num_heads = 8
    mha = MultiHeadAttention(d_model, num_heads)

    x = torch.randn(2, 10, d_model)
    out = mha(x)
    assert out.shape == (2, 10, d_model), f"输出形状错误: {out.shape}"

    # 带 mask
    mask = torch.tril(torch.ones(10, 10)).bool()
    out_m = mha(x, mask=mask)
    assert out_m.shape == (2, 10, d_model)

    # 检查内部结构
    assert mha.d_k == d_model // num_heads
    assert hasattr(mha, 'rope')
    assert hasattr(mha, 'w_q')
    assert hasattr(mha, 'w_k')
    assert hasattr(mha, 'w_v')
    assert hasattr(mha, 'w_o')

    # 不同输入产生不同输出（非退化）
    x2 = torch.randn(2, 10, d_model)
    out2 = mha(x2)
    assert not torch.allclose(out, out2)

    print("  [PASS] MultiHeadAttention")


def test_transformer_block():
    """TransformerBlock: 形状正确 + 残差连接 + Pre-Norm"""
    d_model = 64
    num_heads = 8
    block = TransformerBlock(d_model, num_heads)

    x = torch.randn(2, 10, d_model)
    out = block(x)
    assert out.shape == (2, 10, d_model), f"输出形状错误: {out.shape}"

    # 带 mask
    mask = torch.tril(torch.ones(10, 10)).bool()
    out_m = block(x, mask=mask)
    assert out_m.shape == (2, 10, d_model)

    # 检查内部结构
    assert hasattr(block, 'attention')
    assert hasattr(block, 'ffn')
    assert hasattr(block, 'norm1')
    assert hasattr(block, 'norm2')

    # 残差连接验证：全零输入时输出应接近 0
    x_zero = torch.zeros(2, 10, d_model)
    out_zero = block(x_zero)
    assert out_zero.abs().max() < 1.0, "残差连接可能有问题"

    print("  [PASS] TransformerBlock")


if __name__ == "__main__":
    print("Part 5 注意力机制与 Transformer Block 测试")
    print("=" * 45)
    test_scaled_dot_product_attention()
    test_multihead_attention()
    test_transformer_block()
    print("=" * 45)
    print("全部测试通过!")
