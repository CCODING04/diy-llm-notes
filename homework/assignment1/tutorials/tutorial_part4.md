# Assignment 1 - Part 4：基础算子实现

> 📍 作业进度：Assignment 1，第 4 / 9 部分
> 📅 生成时间：2026-06-26
> 📎 原作业参考：`coursework/assignment1-basics/CS336_Assignment1_Transformer.ipynb`

---

## 目标与要求

从零实现 Transformer 的 7 个基础算子。这些算子会被后续的注意力机制、Transformer Block、训练脚本复用。

**实现文件**：`homework/assignment1/scripts/model_components.py`

### 7 个算子清单

| # | 算子 | 类型 | 关键点 |
|---|------|------|--------|
| 1 | `Linear` | `nn.Module` | 截断正态初始化，无 bias |
| 2 | `Embedding` | `nn.Module` | token ID → 行下标索引 |
| 3 | `RMSNorm` | `nn.Module` | rsqrt(mean(x²))，可学习 weight |
| 4 | `SwiGLU` | `nn.Module` | SiLU gate + GLU，d_ff 向上取整到 64 的倍数 |
| 5 | `RotaryPositionalEmbedding` | `nn.Module` | 频率预计算 + cos/sin 缓存 + 旋转 |
| 6 | `softmax` | 函数 | 数值稳定（max 减法） |
| 7 | `cross_entropy` | 函数 | 从零实现（用 numpy，不调 torch） |

---

## Step 1：Linear（自定义线性层，无 bias）

### 规格

```
Linear(in_features, out_features)
  - self.weight: Parameter(out_features, in_features)    # 注意形状！
  - 初始化：trunc_normal(mean=0, std=sigma, a=-3σ, b=3σ)
    其中 σ² = 2 / (in_features + out_features)
  - forward(x): return x @ self.weight.t()
```

### 关键点

- **为什么是 `(out_features, in_features)` 而非反过来？** 因为 forward 做 `x @ weight.t()`，x 形状为 `(B, in_features)`，需要 `weight.t()` 形状为 `(in_features, out_features)` 才能做矩阵乘法。这与 `nn.Linear` 的权重布局一致。
- **没有 bias**：整个 Transformer 中所有 Linear 层都不使用 bias 项。

### 提示

```python
import math
import torch.nn.init as init

sigma = math.sqrt(2.0 / (in_features + out_features))
init.trunc_normal_(self.weight, mean=0.0, std=sigma, a=-3.0 * sigma, b=3.0 * sigma)
```

### 验证

```python
model = Linear(6, 3)
test_w = torch.randn(3, 6)
model.load_state_dict({'weight': test_w})
x = torch.randn(1, 6)
assert torch.allclose(model(x), x @ test_w.t())
```

---

## Step 2：Embedding

### 规格

```
Embedding(num_embeddings, embedding_dim)
  - self.weights: Parameter(num_embeddings, embedding_dim)
  - 初始化：trunc_normal(mean=0, std=1.0, a=-3.0, b=3.0)
  - forward(token_ids): return self.weights[token_ids]
```

### 关键点

- **不是 `nn.Embedding` 的封装**——需要自己管理 weight 参数。
- `forward` 接收 `(B, L)` 的 token ID 张量，返回 `(B, L, D)` 的向量。
- 直接用 **高级索引**：`self.weights[token_ids]`，PyTorch 会自动处理多维索引。

### 验证

```python
w = torch.randn(10, 3)
model = Embedding(10, 3)
model.load_state_dict({'weights': w})
ids = torch.tensor([[2, 9, 5], [3, 2, 6]])
out = model(ids)          # (2, 3, 3)
assert torch.equal(out[0, 0], w[2])
assert torch.equal(out[1, 2], w[6])
```

---

## Step 3：RMSNorm

### 规格

```
RMSNorm(d_model, eps=1e-5)
  - self.weight: Parameter(d_model)    # 可学习的缩放参数，初始化为全 1
  - self.eps: float
  - forward(x):
      x_fp32 = x.float()
      rms = rsqrt(mean(x_fp32², dim=-1) + eps)    # 注意是对最后一维求均值
      return (x_fp32 * rms * weight).to(x.dtype)
```

### 关键点

- **对哪个维度归一化？** `dim=-1`——对每个 token 的特征向量内部做归一化。`x` 形状为 `(B, L, d_model)`，对最后一维 `d_model` 求均值。
- **为什么转 float32？** 混合精度训练中 `x` 可能是 float16/bfloat16，直接计算 rsqrt 可能数值下溢。
- **为什么转回原始 dtype？** 匹配后续层的输入类型，避免类型不匹配错误。
- RMSNorm vs LayerNorm：RMSNorm **不做均值减法**（re-centering），只做缩放（re-scaling）。计算量更少，且不强制分布中心为 0，对深层残差信息更温和。

### 公式

$$\text{RMSNorm}(x) = \frac{x}{\sqrt{\text{mean}(x^2) + \epsilon}} \cdot w$$

### 提示

```python
# torch.rsqrt(x) = 1/sqrt(x)，计算更高效
rms = torch.rsqrt(x_fp32.pow(2).mean(-1, keepdim=True) + self.eps)
```

### 验证

```python
norm = RMSNorm(3)
x = torch.randn(2, 3)
out = norm(x)    # (2, 3)，每行归一化后平方和的均值约等于 1
```

---

## Step 4：SwiGLU

### 规格

```
SwiGLU(d_model)
  内部计算 d_ff：
    d_ff = int(8/3 * d_model)
    d_ff = ceil_to_multiple_of_64(d_ff)    # (d_ff + 63) // 64 * 64

  - self.w_gate: Linear(d_model, d_ff)    # Gate 投影
  - self.w_up:   Linear(d_model, d_ff)    # Up 投影（Value）
  - self.w_down: Linear(d_ff, d_model)    # Down 投影（输出）

  forward(x):
    gate = w_gate(x)
    swish = gate * sigmoid(gate)        # SiLU = x·σ(x)
    return w_down(swish * w_up(x))
```

### 关键点

- **为什么有 3 个 Linear 而非 2 个？** 传统 FFN 用 `w1(x) → activation → w2(...)`（2 个矩阵）。SwiGLU 用 3 个：Gate 投影（决定激活的"开关"强度）、Up 投影（Value，用来乘门控）、Down 投影（回到 d_model）。三个矩阵的参数总量与传统 FFN 扩维 4× 大致相当。
- **d_ff 取整到 64 的倍数**：为了 GPU 计算效率——Tensor Core 偏爱 64 对齐的维度。
- **SiLU 也叫 Swish**：`SiLU(x) = x · σ(x)`，其中 σ 是 sigmoid。它与 GELU 形态相似但在尾部分布不同，且 SwiGLU 中 gate 的可学习参数使得激活模式更灵活。

### SwiGLU 计算流程

```
x (B, L, d_model)
    │
    ├──→ w_gate ──→ gate ──→ SiLU(gate) ──┐
    │                                       ├──→ × ──→ w_down ──→ out (B, L, d_model)
    └──→ w_up ──────────────────────────────┘
```

### 提示

```python
# SiLU 实现
torch.sigmoid(gate) * gate    # gate * sigmoid(gate)
```

### 验证

```python
swiglu = SwiGLU(256)
x = torch.randn(2, 10, 256)
out = swiglu(x)
assert out.shape == (2, 10, 256)

# d_ff 检查：256 * 8/3 ≈ 682.67 → ceil64 → 704
# 可以检查内部 d_ff 是否为 64 的倍数
```

---

## Step 5：RotaryPositionalEmbedding（RoPE）

### 规格

```
RotaryPositionalEmbedding(theta=10000.0, d_k)

  __init__:
    - self.theta, self.d_k
    - register_buffer("cos", None, persistent=False)   # 缓存，惰性计算
    - register_buffer("sin", None, persistent=False)

  _build_cache(seq_len, device, dtype):
    - 仅在 cos/sin 为 None 或 seq_len 超过缓存长度时才重新计算
    - powers = [0, 2, 4, ..., d_k-2]    # 步长 2，在 device 上创建
    - inv_freq = theta^(-powers / d_k)   # 注意用 ** 而非 ^
    - t = [0, 1, 2, ..., seq_len-1]
    - freqs = outer(t, inv_freq)        # (seq_len, d_k/2)
    - 用 register_buffer 持久化：
        self.register_buffer("cos", cos(freqs).to(dtype), persistent=False)
        self.register_buffer("sin", sin(freqs).to(dtype), persistent=False)

  forward(x):    # x: (B, H, S, d_k)  注意是 4 维含 head 维度
    - 确保 d_k 是偶数
    - 调用 _build_cache(S, x.device, x.dtype)
    - 取 cos/sin 的前 S 行 → reshape 为 (1, 1, S, d_k/2) 或对等变体
    - x_even = x[..., 0::2]    # 偶数位置：0, 2, 4, ...
    - x_odd  = x[..., 1::2]    # 奇数位置：1, 3, 5, ...
    - x_rot_even = x_even * cos - x_odd * sin
    - x_rot_odd  = x_even * sin + x_odd * cos
    - 交错还原为 (B, H, S, d_k)
```

### 关键点

- **RoPE 作用在哪个维度？** d_k 维度（head_dim），不是 seq_len 维度。所以 `forward` 输入是 4 维：`(B, num_heads, seq_len, d_k)`。
- **为什么拆分偶数/奇数位置？** RoPE 将 d_k 维的每两个相邻元素配对，视为复数的实部和虚部，用一个 2D 旋转矩阵对应一个频率。配对方式是 `(0,1), (2,3), (4,5), ...` 而非 `(0, d_k/2), (1, d_k/2+1), ...`。
- **缓存机制**：`_build_cache` 只在 seq_len 增长时重新计算，避免重复计算。使用 `register_buffer` 确保 cos/sin 跟随模型移动到正确的 device。
- **theta=10000** 是业界默认值，控制旋转频率的范围。theta 越大 → 频率越低 → 适合长距离依赖。

### RoPE 的数学直觉

```
位置 i 的旋转角度：θ_i = i / (theta^(2k/d_k))   对 k = 0, 1, ..., d_k/2-1

对位置 i 的 x 向量，将其视为 d_k/2 个 2D 向量对，每个对旋转 θ_i 弧度。

查询 q_i 和键 k_j 经过旋转后，点积中自动出现 (i-j) 的项
→ 注意力分数显式依赖于相对位置差 (i-j) 而非绝对位置 i
```

### 提示

```python
# 频率计算（在 _build_cache 中）
powers = torch.arange(0, d_k, 2, device=device).float()     # [0, 2, 4, ..., d_k-2]
inv_freq = theta ** (-powers / d_k)

# 位置序列
t = torch.arange(seq_len, device=device).float()            # [0, 1, ..., seq_len-1]
freqs = torch.outer(t, inv_freq)                            # (seq_len, d_k/2)

# 持久化缓存（关键：用 register_buffer 而非普通属性，确保跟随 model.to(device)）
self.register_buffer("cos", torch.cos(freqs).to(dtype), persistent=False)
self.register_buffer("sin", torch.sin(freqs).to(dtype), persistent=False)
```

### 提示：cos/sin 的维度对齐

`cos` 是 `(seq_len, d_k/2)`，需要 broadcast 到 `(1, 1, seq_len, d_k/2)` 才能与 `(B, H, seq_len, d_k/2)` 的 x_even/x_odd 相乘。可以用 `view` 或 `unsqueeze`。也可以用 `view(1, seq_len, -1)` 省略 head 维度（Pytorch 会自动广播）。

### 提示：交错还原

```python
# 方法 1：用 stack + flatten
x_rot = torch.stack([x_rot_even, x_rot_odd], dim=-1)   # (B,H,S,D/2,2)
x_rot = x_rot.flatten(-2)                               # (B,H,S,D)

# 方法 2：预分配 + 分片赋值
x_rot = torch.empty_like(x)
x_rot[..., 0::2] = x_rot_even
x_rot[..., 1::2] = x_rot_odd
```

### 验证

```python
rope = RotaryPositionalEmbedding(theta=10000.0, d_k=8)
x = torch.randn(1, 2, 5, 8)   # (B=1, H=2, S=5, d_k=8)
out = rope(x)
assert out.shape == x.shape
# 位置 0 应该只包含 cos(0)=1, sin(0)=0 → 不变？
# 不是的，频率不同导致旋转角度不同，每个 (位置, 频率对) 的旋转角都不同。
# 只有每个频率对的 cos(0)=1, sin(0)=0 成立，所以位置 0 理论上不变。
```

---

## Step 6：softmax（函数）

### 规格

```python
def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    # 1. 沿 dim 减去最大值（数值稳定）
    # 2. exp
    # 3. 除以沿 dim 的和
```

### 关键点

- **为什么减最大值？** 防止 exp 溢出。如果 x 中有 +100，`exp(100)` 会溢出到 inf。减去最大值后 max=0 → `exp(0)=1`，其余值 ≤ 1。
- **函数不是 Module**：这是一个纯函数（与 `F.softmax` 类似），不需要可学习参数。

### 公式

$$\text{softmax}(x_i) = \frac{e^{x_i - \max(x)}}{\sum_j e^{x_j - \max(x)}}$$

### 验证

```python
x = torch.tensor([[1.0, 2.0, 3.0], [1.0, 2.0, 100.0]])
out = softmax(x, dim=-1)
assert torch.allclose(out.sum(-1), torch.ones(2))
assert out[0, 2] > out[0, 1] > out[0, 0]  # 单调性
assert not torch.isnan(out).any()            # 数值稳定
```

---

## Step 7：cross_entropy（函数）

### 规格

```python
def cross_entropy(logits: np.ndarray, targets: np.ndarray) -> float:
    # logits: (batch_size, vocab_size)  模型输出
    # targets: (batch_size,)           正确 token ID
    # 返回标量 loss（取 batch 平均）

    # 步骤：
    # 1. 每行减最大值（数值稳定）
    # 2. log_sum_exp = log(sum(exp(shifted_logits), axis=-1))
    # 3. target_logits = shifted_logits[每行的 target 位置]
    # 4. loss_i = log_sum_exp - target_logits      # 每个样本的 CE loss
    # 5. return mean(loss_i)
```

### 关键点

- **用 numpy 实现，不调 torch**——理解 CE 的底层计算。
- `log_sum_exp - target_logits` 为什么等于交叉熵？
  - 交叉熵 = `-log(softmax(logits)[target])`
  - `softmax(logits)[target] = exp(logits[target]) / sum(exp(logits))`
  - `-log(...) = log(sum(exp)) - logits[target]` ← 就是 `log_sum_exp - target_logits`
- **数值稳定**：`shifted_logits = logits - max` 后，exp 不会溢出，且不影响 CE 结果（分子分母消掉）。

### 公式推导

$$\text{CE} = -\log\left(\frac{e^{z_y}}{\sum_j e^{z_j}}\right) = \log\left(\sum_j e^{z_j}\right) - z_y$$

$$\text{（带稳定性）} = \log\left(\sum_j e^{z_j - z_{\max}}\right) - (z_y - z_{\max})$$

### 提示

```python
import numpy as np

# 取 target 位置的 logits
target_logits = np.take_along_axis(shifted_logits, targets[:, None], axis=-1).squeeze(-1)
# 或者用高级索引
target_logits = shifted_logits[np.arange(len(targets)), targets]
```

### 验证

```python
logits = np.array([[2.0, 1.0, 0.1], [0.5, 2.5, 0.1]])
targets = np.array([0, 1])
loss = cross_entropy(logits, targets)   # 第一个样本预测正确（低 loss），第二个也正确

# 参考 torch 结果
import torch
t_loss = torch.nn.functional.cross_entropy(torch.tensor(logits), torch.tensor(targets))
assert np.allclose(loss, t_loss.item(), atol=1e-5)
```

---

## 完整测试

将以上 7 个组件全部实现到 `homework/assignment1/scripts/model_components.py`，然后运行以下测试脚本验证。

```bash
cd homework/assignment1
python tests/test_part4.py
```

### 测试脚本 `tests/test_part4.py`

```python
"""Part 4 基础算子测试"""
import sys
sys.path.insert(0, 'scripts')

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
    # 验证归一化后每行的平方和均值 ≈ 1（因为 weight 初始为 1）
    rms_per_row = out.float().pow(2).mean(-1)
    assert torch.allclose(rms_per_row, torch.ones_like(rms_per_row), atol=1e-4)
    
    # 验证 weight 生效：手动设置 weight=2
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
    
    # d_ff 应该是 64 的倍数
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
    
    # 位置 0: cos(0)=1, sin(0)=0 对所有频率成立 → 旋转不改变值
    # 注意：这只对单一频率对的第一个位置成立。但因为有 d_k/2 个不同频率，
    # 每个频率的 cos(0)=1 都成立，所以位置 0 全部不变。
    assert torch.allclose(out[:, :, 0, :], x[:, :, 0, :], atol=1e-5)
    
    # 缓存测试：重复调用不应崩溃
    out2 = rope(x)
    print("  [PASS] RoPE")

def test_softmax():
    """softmax: 概率分布 + 数值稳定"""
    # 常规输入
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
    # 最大值的 prob 应该接近 1
    assert out_big[0, 2] > 0.999
    
    print("  [PASS] softmax")

def test_cross_entropy():
    """cross_entropy: 与 torch 实现一致"""
    logits = np.array([[2.0, 1.0, 0.1], [0.5, 2.5, 0.1], [0.1, 0.1, 5.0]])
    targets = np.array([0, 1, 2])
    loss = cross_entropy(logits, targets)
    
    # 对比 torch
    t_loss = torch.nn.functional.cross_entropy(
        torch.tensor(logits), torch.tensor(targets, dtype=torch.long)
    )
    assert np.allclose(loss, t_loss.item(), atol=1e-5), f"{loss} vs {t_loss.item()}"
    
    # per-sample 验证：第一个样本应该 loss 最低（最高 logit 匹配 target）
    # 第三个样本也应该 loss 低
    assert loss < 1.0  # 整体 loss 应该较小
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
```

---

## 常见陷阱

| # | 陷阱 | 正确做法 |
|---|------|---------|
| 1 | Linear 的 weight 形状写成 `(in, out)` | 应该是 `(out, in)`，forward 中 `x @ weight.t()` |
| 2 | RMSNorm 沿 `dim=0`（batch 维）归一化 | 应沿 `dim=-1`（特征维），逐 token 归一化 |
| 3 | RMSNorm 忘记 `to(x.dtype)` 转回原类型 | 必须转回，否则后续层可能报 dtype mismatch |
| 4 | SwiGLU 用 GeLU/GELU 而非 SiLU | SiLU 是 `x * sigmoid(x)`，不是 `x * Φ(x)` |
| 5 | RoPE `_build_cache` 每次 forward 都重建 | 只在 seq_len 增长时重建；用 `register_buffer` 持久化（非普通属性），确保 `model.to(device)` 时自动跟随 |
| 6 | RoPE `_build_cache` 忘记传 device/dtype | 从 `forward` 传入 `x.device, x.dtype`，在对应 device 上创建 tensor |
| 7 | RoPE `_build_cache` 算好 cos/sin 后没有 register_buffer | 必须用 `self.register_buffer("cos", ..., persistent=False)` 持久化，否则下次 forward 丢失 |
| 8 | softmax 不减去最大值直接 exp | 大输入下 exp 溢出到 inf |
| 9 | cross_entropy 用 torch 而非 numpy | 作业要求从零实现，手写 log-sum-exp |

---

## 完成标志

运行 `python tests/test_part4.py` 全部 7 项通过后，输入 `提交作业`。
