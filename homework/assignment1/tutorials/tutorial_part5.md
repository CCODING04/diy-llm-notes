# Assignment 1 - Part 5：注意力机制与 Transformer Block

> 📍 作业进度：Assignment 1，第 5 / 9 部分
> 📅 生成时间：2026-07-08
> 📎 原作业参考：`coursework/assignment1-basics/CS336_Assignment1_Transformer.ipynb`

---

## 目标与要求

在 Part 4 的 7 个基础算子之上，实现 Transformer 的 3 个核心组件：缩放点积注意力、因果多头注意力、Transformer Block。

**实现文件**：`homework/assignment1/scripts/model_components.py`（追加到文件末尾）

### 3 个组件清单

| # | 组件 | 类型 | 依赖 |
|---|------|------|------|
| 1 | `scaled_dot_product_attention` | 函数 | `softmax`（Part 4） |
| 2 | `MultiHeadAttention` | `nn.Module` | `Linear`、`RotaryPositionalEmbedding`、`scaled_dot_product_attention` |
| 3 | `TransformerBlock` | `nn.Module` | `MultiHeadAttention`、`SwiGLU`、`RMSNorm` |

---

## Step 1：scaled_dot_product_attention（缩放点积注意力）

### 规格

```python
def scaled_dot_product_attention(q, k, v, mask=None):
    """
    q: (..., seq_len, d_k)    # 查询
    k: (..., seq_len, d_k)    # 键
    v: (..., seq_len, d_v)    # 值
    mask: (seq_len, seq_len)  # bool 张量，True=保留，False=屏蔽
    返回: (output, attn_weights)
    """
    # 1. scores = q @ k^T / sqrt(d_k)
    # 2. if mask: scores.masked_fill(mask==False, -inf)
    # 3. attn_weights = softmax(scores, dim=-1)
    # 4. output = attn_weights @ v
    # 5. return output, attn_weights
```

### 关键点

- **为什么要除以 `sqrt(d_k)`？** 当 d_k 较大时，`q @ k^T` 的方差会随 d_k 线性增长，导致 softmax 的输入值过大，梯度趋近于 0（饱和区）。除以 `sqrt(d_k)` 将方差稳定在 1 附近。
- **mask 的语义**：`True` 表示**保留**该位置的注意力权重，`False` 表示**屏蔽**（设为 -inf，softmax 后变为 0）。因果 mask 是下三角矩阵：位置 i 只能看到 ≤ i 的位置。
- **返回两个值**：`output` 是加权后的 V，`attn_weights` 是注意力权重矩阵（可用于可视化或调试）。
- **支持任意前缀维度**：`q` 可以是 3D `(B, S, d_k)` 或 4D `(B, H, S, d_k)`，matmul 会自动在前缀维度上广播。

### 公式

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

### 数值推演（d_k=64 时）

```
q @ k^T 的每个元素：64 个乘积之和
  → 每个乘积的方差 ≈ 1（假设 q, k 各分量独立、均值 0、方差 1）
  → 和的方差 = 64
  → 标准差 = 8

除以 sqrt(64)=8 后：
  → 标准差回到 1
  → softmax 输入值在 [-3, 3] 范围内，梯度健康
```

### 提示

```python
import math

d_k = q.size(-1)
scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

if mask is not None:
    scores = scores.masked_fill(mask == False, float('-inf'))

attn_weights = softmax(scores, dim=-1)
output = torch.matmul(attn_weights, v)
return output, attn_weights
```

### 验证

```python
q = torch.randn(2, 3, 8)
k = torch.randn(2, 3, 8)
v = torch.randn(2, 3, 8)
out, weights = scaled_dot_product_attention(q, k, v)
assert out.shape == (2, 3, 8)
assert weights.shape == (2, 3, 3)
assert torch.allclose(weights.sum(-1), torch.ones(2, 3), atol=1e-5)

# 因果 mask
mask = torch.tril(torch.ones(3, 3)).bool()
out_m, weights_m = scaled_dot_product_attention(q, k, v, mask=mask)
# 第一行只有 weights_m[0,0,0]=1，其余为 0
assert weights_m[0, 0, 1] == 0.0
```

---

## Step 2：MultiHeadAttention（因果多头注意力）

### 规格

```
MultiHeadAttention(d_model, num_heads)
  - d_model % num_heads == 0
  - d_k = d_model // num_heads
  - w_q, w_k, w_v, w_o: Linear(d_model, d_model)
  - rope: RotaryPositionalEmbedding(theta=10000.0, d_k=d_k)

  forward(x, mask=None):
    # x: (B, S, d_model)
    1. Q = w_q(x) → view → transpose → (B, H, S, d_k)
    2. K = w_k(x) → view → transpose → (B, H, S, d_k)
    3. V = w_v(x) → view → transpose → (B, H, S, d_k)
    4. Q = rope(Q), K = rope(K)    # V 不做 RoPE！
    5. out, _ = scaled_dot_product_attention(Q, K, V, mask)
    6. out = transpose → view → (B, S, d_model)
    7. return w_o(out)
```

### 关键点

- **为什么 V 不做 RoPE？** RoPE 的设计目标是让 Q·K 的点积包含相对位置信息。V 承载的是"内容"而非"位置查询"，旋转 V 没有数学意义，反而会破坏内容表示。
- **4 个 Linear 的作用**：
  - `w_q`：将输入投影为"查询"（我想找什么）
  - `w_k`：将输入投影为"键"（我能提供什么）
  - `w_v`：将输入投影为"值"（我的具体内容）
  - `w_o`：将多头输出合并投影回 d_model 维度
- **多头拆分**：`(B, S, d_model)` → `(B, S, H, d_k)` → `(B, H, S, d_k)`。先 view 拆维度，再 transpose 把 head 维提前。
- **合并多头**：`(B, H, S, d_k)` → `(B, S, H, d_k)` → `(B, S, d_model)`。transpose + contiguous + view。

### 数据流图

```
x (B, S, d_model)
    │
    ├──→ w_q ──→ view(B,S,H,d_k) ──→ transpose(1,2) ──→ rope ──→ Q
    ├──→ w_k ──→ view(B,S,H,d_k) ──→ transpose(1,2) ──→ rope ──→ K
    └──→ w_v ──→ view(B,S,H,d_k) ──→ transpose(1,2) ──────────→ V
                                                                      │
                    ┌─────────────────────────────────────────────────┘
                    │
                    ▼
          scaled_dot_product_attention(Q, K, V, mask)
                    │
                    ▼ out (B, H, S, d_k)
          transpose(1,2) ──→ contiguous() ──→ view(B, S, d_model)
                    │
                    ▼
                   w_o ──→ output (B, S, d_model)
```

### 提示

```python
# 多头拆分
q = self.w_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

# RoPE 只作用于 Q 和 K
q = self.rope(q)
k = self.rope(k)

# 合并多头
out = scores.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
```

### 验证

```python
mha = MultiHeadAttention(d_model=64, num_heads=8)
x = torch.randn(2, 10, 64)
out = mha(x)
assert out.shape == (2, 10, 64)

# 带 mask
mask = torch.tril(torch.ones(10, 10)).bool()
out_m = mha(x, mask=mask)
assert out_m.shape == (2, 10, 64)
```

---

## Step 3：TransformerBlock（Pre-Norm 残差块）

### 规格

```
TransformerBlock(d_model, num_heads)
  - attention: MultiHeadAttention(d_model, num_heads)
  - ffn: SwiGLU(d_model)
  - norm1: RMSNorm(d_model)    # 注意力前的归一化
  - norm2: RMSNorm(d_model)    # FFN 前的归一化

  forward(x, mask=None):
    # Pre-Norm 架构：Norm → 子层 → 残差
    residual = x
    x = norm1(x)
    x = attention(x, mask=mask)
    x = x + residual

    residual = x
    x = norm2(x)
    x = ffn(x)
    x = x + residual
    return x
```

### 关键点

- **Pre-Norm vs Post-Norm**：
  - **Post-Norm**（原始 Transformer）：`x + SubLayer(x)` → 然后 Norm。训练不稳定，需要 warmup。
  - **Pre-Norm**（GPT-2+, LLaMA）：`x + SubLayer(Norm(x))`。梯度流更顺畅，训练更稳定，不需要 warmup。
  - 现代 LLM 几乎全部使用 Pre-Norm。
- **为什么用两个独立的 RMSNorm？** 每个子层（Attention 和 FFN）的输入分布不同，需要独立的归一化参数。共享 Norm 会限制表达能力。
- **d_ff 参数**：TransformerBlock 不需要接收 `d_ff`，因为 `SwiGLU(d_model)` 内部自动计算 `d_ff = ceil64(int(8/3 * d_model))`。

### Pre-Norm 的梯度直觉

```
Post-Norm:  loss → Norm → SubLayer → x
  问题：Norm 层会压缩梯度，深层网络中梯度逐渐衰减

Pre-Norm:   loss → SubLayer(Norm(x)) + x
  优势：残差连接提供"梯度高速公路"，梯度可以直接跳过子层传回
  → Norm 只影响子层的输入，不阻碍梯度回传
```

### 提示

```python
def forward(self, x, mask=None):
    # Attention 子层
    residual = x
    x = self.norm1(x)
    x = self.attention(x, mask=mask)
    x = x + residual

    # FFN 子层
    residual = x
    x = self.norm2(x)
    x = self.ffn(x)
    x = x + residual
    return x
```

### 验证

```python
block = TransformerBlock(d_model=64, num_heads=8)
x = torch.randn(2, 10, 64)
out = block(x)
assert out.shape == (2, 10, 64)

# 残差连接：输入输出形状一定相同
mask = torch.tril(torch.ones(10, 10)).bool()
out_m = block(x, mask=mask)
assert out_m.shape == x.shape
```

---

## 完整测试

将以上 3 个组件追加到 `model_components.py`，然后运行：

```bash
cd homework/assignment1
python tests/test_part5.py
```

### 测试脚本 `tests/test_part5.py`

```python
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

    # 残差连接验证：初始化时输出应接近输入（weight 接近 0）
    # 用全零输入验证残差传播
    x_zero = torch.zeros(2, 10, d_model)
    out_zero = block(x_zero)
    # 由于 Linear 初始化接近 0 + 残差，输出应接近 0
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
```

---

## 常见陷阱

| # | 陷阱 | 正确做法 |
|---|------|---------|
| 1 | softmax 的 dim 用错（如 dim=0） | 应该是 `dim=-1`，沿最后一维做 softmax |
| 2 | mask 语义反了（True=屏蔽） | `masked_fill(mask == False, -inf)`，True=保留 |
| 3 | V 也做了 RoPE | RoPE 只作用于 Q 和 K，V 不做 |
| 4 | 多头拆分后忘记 transpose | `view(B,S,H,d_k)` 后需要 `transpose(1,2)` 变成 `(B,H,S,d_k)` |
| 5 | 合并多头时忘记 contiguous | `transpose` 后 tensor 不连续，需要 `contiguous()` 才能 `view` |
| 6 | TransformerBlock 用 Post-Norm | 应该用 Pre-Norm：先 Norm 再子层再残差 |
| 7 | SwiGLU 接收 d_ff 参数 | `SwiGLU(d_model)` 内部自动算 d_ff，不需要外部传入 |

---

## 完成标志

运行 `python tests/test_part5.py` 全部 3 项通过后，输入 `提交作业`。
