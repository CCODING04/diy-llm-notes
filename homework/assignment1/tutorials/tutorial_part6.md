# Assignment 1 - Part 6：训练基础设施

> 📍 作业进度：Assignment 1，第 6 / 9 部分
> 📅 生成时间：2026-07-08
> 📎 原作业参考：`coursework/assignment1-basics/CS336_Assignment1_Transformer.ipynb`

---

## 目标与要求

实现 Transformer 训练所需的 5 个基础设施组件：优化器、学习率调度、梯度裁剪、数据加载、检查点保存/加载。

**实现文件**：`homework/assignment1/scripts/training.py`（新文件）

### 5 个组件清单

| # | 组件 | 类型 | 说明 |
|---|------|------|------|
| 1 | `AdamW` | `Optimizer` | 解耦权重衰减的 Adam 优化器 |
| 2 | `get_lr_cosine_schedule` | 函数 | 余弦退火学习率调度 |
| 3 | `gradient_clipping` | 函数 | 全局梯度范数裁剪 |
| 4 | `get_batch` | 函数 | 从 token 序列中采样训练 batch |
| 5 | `save_checkpoint` / `load_checkpoint` | 函数 | 模型/优化器状态持久化 |

---

## Step 1：AdamW 优化器

### 规格

```
AdamW(params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
  继承 torch.optim.Optimizer

  step():
    对每个参数 p（有梯度时）：
      1. m_t = β1 * m_{t-1} + (1 - β1) * g_t       # 一阶矩（动量）
      2. v_t = β2 * v_{t-1} + (1 - β2) * g_t²       # 二阶矩（梯度平方）
      3. m̂_t = m_t / (1 - β1^t)                      # 偏置修正
      4. v̂_t = v_t / (1 - β2^t)                      # 偏置修正
      5. θ = θ - lr * m̂_t / (√v̂_t + ε)              # 参数更新
      6. θ = θ - lr * λ * θ                           # 解耦权重衰减
```

### 关键点

- **Adam vs AdamW**：Adam 的 L2 正则化和权重衰减混在一起（`grad + λ*θ`），在自适应学习率下效果不佳。AdamW 将权重衰减从梯度更新中解耦出来（先 Adam 更新，再单独衰减），这是论文 "Decoupled Weight Decay Regularization" (Loshchilov & Hutter, 2019) 的核心贡献。
- **偏置修正为什么需要？** m 和 v 初始化为 0，前几步它们会偏向 0。除以 `(1 - β^t)` 修正这个偏差。当 t 较大时 `β^t ≈ 0`，修正项趋近于 1。
- **状态管理**：每个参数需要存储 `step`（计数器）、`exp_avg`（一阶矩 m）、`exp_avg_sq`（二阶矩 v）。用 `self.state[param]` 管理。

### 公式

$$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$$
$$v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$$
$$\hat{m}_t = \frac{m_t}{1-\beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1-\beta_2^t}$$
$$\theta_t = \theta_{t-1} - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} - \eta \lambda \theta_{t-1}$$

### 数值推演（β1=0.9, β2=0.999, t=1）

```
初始: m=0, v=0, g=2.0

第 1 步:
  m = 0.9*0 + 0.1*2.0 = 0.2
  v = 0.999*0 + 0.001*4.0 = 0.004
  m̂ = 0.2 / (1-0.9) = 2.0        ← 修正后等于原始梯度！
  v̂ = 0.004 / (1-0.999) = 4.0    ← 修正后等于梯度平方！
  update = 2.0 / (2.0 + 1e-8) ≈ 1.0

第 100 步（假设梯度恒为 2.0）:
  m ≈ 2.0（收敛）
  v ≈ 4.0（收敛）
  m̂ ≈ 2.0 / (1-0.9^100) ≈ 2.0 / 0.99997 ≈ 2.0
  v̂ ≈ 4.0 / (1-0.999^100) ≈ 4.0 / 0.0952 ≈ 42.0
  update = 2.0 / (6.48 + 1e-8) ≈ 0.308
  → 自适应学习率使步长逐渐减小
```

### 提示

```python
import torch
from torch.optim import Optimizer

class AdamW(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)

                state['step'] += 1
                t = state['step']
                m, v = state['exp_avg'], state['exp_avg_sq']
                β1, β2 = group['betas']

                # 更新矩估计
                m.mul_(β1).add_(p.grad, alpha=1-β1)
                v.mul_(β2).addcmul_(p.grad, p.grad, value=1-β2)

                # 偏置修正
                bias_corr1 = 1 - β1**t
                bias_corr2 = 1 - β2**t

                # 参数更新
                step_size = group['lr'] * (bias_corr2**0.5) / bias_corr1
                p.addcdiv_(m, v.sqrt().add_(group['eps']), value=-step_size)

                # 解耦权重衰减
                if group['weight_decay'] != 0:
                    p.add_(p, alpha=-group['lr'] * group['weight_decay'])
```

### 验证

```python
model = torch.nn.Linear(3, 2)
opt = AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
x = torch.randn(4, 3)
loss = model(x).sum()
loss.backward()
opt.step()
opt.zero_grad()
# 验证参数已更新
```

---

## Step 2：get_lr_cosine_schedule（余弦退火学习率调度）

### 规格

```
get_lr_cosine_schedule(t, alpha_max, alpha_min, T_w, T_c):
  t: 当前步数
  alpha_max: 最大学习率
  alpha_min: 最小学习率
  T_w: warmup 步数
  T_c: 总训练步数（warmup + 退火）

  if t < T_w:        → linear warmup: (t/T_w) * alpha_max
  elif t <= T_c:     → cosine decay: alpha_min + 0.5*(1+cos(π*(t-T_w)/(T_c-T_w))) * (alpha_max - alpha_min)
  else:              → alpha_min
```

### 关键点

- **Warmup 阶段**：训练初期梯度方差大，大学习率会导致发散。线性 warmup 让学习率从 0 逐步增大到 `alpha_max`。
- **Cosine 退火阶段**：warmup 结束后，学习率按余弦曲线从 `alpha_max` 平滑降到 `alpha_min`。比阶梯式衰减更平滑，训练更稳定。
- **T_c 之后**：学习率保持 `alpha_min` 不变（如果还有多余步数的话）。

### 学习率曲线

```
lr
  ^
  |      ╱‾‾‾‾‾‾╲
  |     ╱        ╲
  |    ╱          ╲
  |   ╱            ╲________
  |  ╱
  | ╱
  +--+----+----------------→ t
  0  T_w  T_c
```

### 提示

```python
import math

def get_lr_cosine_schedule(t, alpha_max, alpha_min, T_w, T_c):
    if t < T_w:
        return (t / T_w) * alpha_max
    elif t <= T_c:
        progress = (t - T_w) / (T_c - T_w)
        return alpha_min + 0.5 * (1 + math.cos(math.pi * progress)) * (alpha_max - alpha_min)
    else:
        return alpha_min
```

### 验证

```python
lr = get_lr_cosine_schedule(t=0, alpha_max=1e-3, alpha_min=1e-5, T_w=100, T_c=1000)
assert lr == 0.0  # t=0 时 warmup 阶段
lr = get_lr_cosine_schedule(t=100, alpha_max=1e-3, alpha_min=1e-5, T_w=100, T_c=1000)
assert abs(lr - 1e-3) < 1e-10  # warmup 结束时达到最大值
lr = get_lr_cosine_schedule(t=1000, alpha_max=1e-3, alpha_min=1e-5, T_w=100, T_c=1000)
assert abs(lr - 1e-5) < 1e-10  # 退火结束时达到最小值
```

---

## Step 3：gradient_clipping（梯度裁剪）

### 规格

```
gradient_clipping(parameters, max_norm, eps=1e-6):
  parameters: 模型参数（可迭代）
  max_norm: 允许的最大全局 L2 范数

  1. 收集所有有梯度的参数
  2. 计算全局梯度范数: total_norm = ‖[‖g1‖₂, ‖g2‖₂, ...]‖₂
  3. clip_coeff = max_norm / (total_norm + eps)
  4. 如果 clip_coeff < 1: 所有梯度 *= clip_coeff
```

### 关键点

- **全局范数 vs 逐参数范数**：不是每个参数独立裁剪，而是计算所有参数梯度的**全局 L2 范数**，然后统一缩放。这保证了梯度方向不变，只缩小幅度。
- **为什么不直接 clamp？** `torch.clamp` 会逐元素裁剪，改变梯度方向。`gradient_clipping` 保持方向一致，只缩放长度。
- **eps 的作用**：防止 `total_norm=0` 时除零。

### 公式

$$\text{total\_norm} = \sqrt{\sum_i \|g_i\|_2^2}$$
$$\text{clip\_coeff} = \min\left(1, \frac{\text{max\_norm}}{\text{total\_norm} + \epsilon}\right)$$
$$g_i \leftarrow g_i \cdot \text{clip\_coeff}$$

### 数值推演

```
参数 1 梯度: [3, 0]  → ‖g1‖₂ = 3
参数 2 梯度: [0, 4]  → ‖g2‖₂ = 4
total_norm = √(9 + 16) = 5

max_norm = 1.0:
  clip_coeff = 1.0 / 5.0 = 0.2 < 1 → 裁剪！
  g1 = [0.6, 0], g2 = [0, 0.8]
  新 total_norm = √(0.36 + 0.64) = 1.0 ✓

max_norm = 10.0:
  clip_coeff = 10.0 / 5.0 = 2.0 > 1 → 不裁剪
```

### 提示

```python
def gradient_clipping(parameters, max_norm, eps=1e-6):
    params_with_grad = [p for p in parameters if p.grad is not None]
    if not params_with_grad:
        return
    total_norm = torch.norm(
        torch.stack([torch.norm(p.grad.detach(), 2) for p in params_with_grad]), 2
    )
    clip_coeff = max_norm / (total_norm + eps)
    if clip_coeff < 1.0:
        for p in params_with_grad:
            p.grad.detach().mul_(clip_coeff)
```

### 验证

```python
p1 = torch.tensor([1.0, 2.0], requires_grad=True)
p2 = torch.tensor([2.0, 2.0], requires_grad=True)
p1.grad = torch.tensor([3.0, 0.0])
p2.grad = torch.tensor([0.0, 4.0])
gradient_clipping([p1, p2], max_norm=1.0)
new_norm = torch.norm(torch.stack([torch.norm(p1.grad, 2), torch.norm(p2.grad, 2)]), 2)
assert abs(new_norm.item() - 1.0) < 1e-5
```

---

## Step 4：get_batch（数据加载）

### 规格

```
get_batch(data, batch_size, context_length, device):
  data: numpy array of token IDs, shape (N,)
  返回: (x_batch, y_batch)
    x_batch: (batch_size, context_length)  — 输入 token
    y_batch: (batch_size, context_length)  — 目标 token（右移 1 位）

  1. 随机采样 batch_size 个起始位置 i ∈ [0, N - context_length)
  2. x_batch[j] = data[i : i + context_length]
  3. y_batch[j] = data[i+1 : i + context_length + 1]
```

### 关键点

- **自回归训练**：给定前 n 个 token，预测第 n+1 个。所以 x 和 y 是同一个序列的偏移版本。
- **为什么 `len(data) - context_length`？** 因为 y 需要多一个 token 作为最后一个目标，所以起始位置最大为 `N - context_length - 1`（含）。
- **返回 torch.Tensor**：需要将 numpy 数据转为 `torch.long`（int64），因为 Embedding 层需要 long 类型索引。

### 提示

```python
import numpy as np
import torch

def get_batch(data: np.ndarray, batch_size: int, context_length: int, device: str):
    max_idx = len(data) - context_length
    ix = torch.randint(0, max_idx, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i+context_length].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i+1:i+context_length+1].astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)
```

### 验证

```python
data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
x, y = get_batch(data, batch_size=2, context_length=3, device='cpu')
assert x.shape == (2, 3)
assert y.shape == (2, 3)
# y 应该是 x 右移 1 位
for i in range(2):
    for j in range(2):
        assert y[i, j] == x[i, j+1] + 1  # 因为数据是连续整数
```

---

## Step 5：save_checkpoint / load_checkpoint（检查点）

### 规格

```
save_checkpoint(model, optimizer, iteration, out):
  checkpoint = {
      'model_state_dict': model.state_dict(),
      'optimizer_state_dict': optimizer.state_dict(),
      'iteration': iteration
  }
  torch.save(checkpoint, out)

load_checkpoint(src, model, optimizer) -> int:
  checkpoint = torch.load(src, map_location='cpu')
  model.load_state_dict(checkpoint['model_state_dict'])
  optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
  return checkpoint['iteration']
```

### 关键点

- **为什么要保存 optimizer 状态？** AdamW 的 m 和 v 矩估计是训练状态的一部分。如果不保存，加载后训练会从 t=0 重新计算矩估计，导致学习率调度混乱。
- **`map_location='cpu'`**：加载时统一映射到 CPU，避免 GPU 内存不足的问题。之后可以用 `model.to(device)` 移到目标设备。
- **iteration 的作用**：恢复训练时需要知道当前步数，以正确计算学习率调度和偏置修正。

### 提示

```python
import torch

def save_checkpoint(model, optimizer, iteration, out):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'iteration': iteration
    }
    torch.save(checkpoint, out)

def load_checkpoint(src, model, optimizer):
    checkpoint = torch.load(src, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['iteration']
```

### 验证

```python
import tempfile, os

model = torch.nn.Linear(3, 2)
opt = AdamW(model.parameters(), lr=1e-3)

# 保存
path = os.path.join(tempfile.gettempdir(), 'test_ckpt.pt')
save_checkpoint(model, opt, iteration=42, out=path)

# 加载到新模型
model2 = torch.nn.Linear(3, 2)
opt2 = AdamW(model2.parameters(), lr=1e-3)
iter_num = load_checkpoint(path, model2, opt2)
assert iter_num == 42
assert torch.allclose(model.weight, model2.weight)
os.remove(path)
```

---

## 完整测试

将以上 5 个组件实现到 `homework/assignment1/scripts/training.py`，然后运行：

```bash
cd homework/assignment1
python tests/test_part6.py
```

### 测试脚本 `tests/test_part6.py`

```python
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
```

---

## 常见陷阱

| # | 陷阱 | 正确做法 |
|---|------|---------|
| 1 | AdamW 权重衰减和梯度更新混在一起 | 先 Adam 更新，再单独 `p.add_(p, alpha=-lr*wd)` |
| 2 | 忘记偏置修正 | `step_size = lr * sqrt(1-β2^t) / (1-β1^t)` |
| 3 | `@torch.no_grad()` 缺失 | step 方法必须加，否则优化器自身也计算梯度 |
| 4 | cosine schedule 的 `t < T_w` 写成 `t <= T_w` | t=T_w 时应该刚好达到 alpha_max（cosine 起点） |
| 5 | gradient_clipping 用逐元素裁剪 | 应该用全局范数裁剪，保持梯度方向 |
| 6 | get_batch 返回 float 类型 | 应该返回 `torch.long`（int64），Embedding 需要 |
| 7 | load_checkpoint 不加 `map_location='cpu'` | GPU 保存的 checkpoint 在 CPU 机器上加载会报错 |

---

## 完成标志

运行 `python tests/test_part6.py` 全部 5 项通过后，输入 `提交作业`。
