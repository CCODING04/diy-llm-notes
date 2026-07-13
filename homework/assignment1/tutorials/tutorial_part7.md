# Assignment 1 - Part 7：完整 Transformer 语言模型

> 📍 作业进度：Assignment 1，第 7 / 9 部分
> 📅 生成时间：2026-07-08
> 📎 原作业参考：`coursework/assignment1-basics/CS336_Assignment1_Transformer.ipynb`

---

## 目标与要求

将 Part 4-6 的所有组件组装为一个完整的 Transformer 语言模型，并实现训练循环。

**实现文件**：
- `homework/assignment1/scripts/model_components.py` — 追加 `TransformerLM` 类
- `homework/assignment1/scripts/training.py` — 追加 `train` 函数

### 2 个组件清单

| # | 组件 | 类型 | 说明 |
|---|------|------|------|
| 1 | `TransformerLM` | `nn.Module` | 完整语言模型：Embedding + N×TransformerBlock + RMSNorm + 输出 Linear |
| 2 | `train` | 函数 | 训练循环：get_batch → forward → loss → backward → clip → step |

---

## Step 1：TransformerLM（完整语言模型）

### 规格

```
TransformerLM(vocab_size, context_length, d_model, num_layers, num_heads)
  - token_embedding: Embedding(vocab_size, d_model)
  - layers: ModuleList([TransformerBlock(d_model, num_heads) for _ in range(num_layers)])
  - norm: RMSNorm(d_model)           # 最终归一化
  - output: Linear(d_model, vocab_size)  # 语言模型头（LM head）

  forward(token_ids):
    # token_ids: (B, S) — token ID 序列
    1. x = token_embedding(token_ids)          # (B, S, d_model)
    2. for layer in layers: x = layer(x)       # (B, S, d_model)
    3. x = norm(x)                             # (B, S, d_model)
    4. logits = output(x)                      # (B, S, vocab_size)
    5. return logits
```

### 关键点

- **因果 mask 在哪生成？** 不在 TransformerLM 内部生成！mask 应该在训练循环中生成并传入。但为了简化，可以让 TransformerBlock 内部自动生成因果 mask（下三角矩阵）。具体取决于你的设计选择。
- **输出 Linear 没有 bias**：与所有其他 Linear 层一致。
- **为什么最后还有一个 RMSNorm？** 最后一层 TransformerBlock 的输出可能数值不稳定（尤其是深层网络），最终 RMSNorm 确保送入 LM head 的向量是归一化的。这是 Pre-Norm 架构的标准做法。
- **参数量估算**：
  ```
  Embedding: vocab_size × d_model
  每层 TransformerBlock: 12 × d_model² (Attention 4d² + FFN 8d²)
  最终 RMSNorm: d_model
  LM head: d_model × vocab_size
  总计 ≈ num_layers × 12d² + 2 × vocab_size × d
  ```

### 数据流图

```
token_ids (B, S)
    │
    ▼
Embedding ──→ x (B, S, d_model)
    │
    ├──→ TransformerBlock_1 ──→ x
    ├──→ TransformerBlock_2 ──→ x
    │    ...
    └──→ TransformerBlock_N ──→ x
    │
    ▼
RMSNorm ──→ x (B, S, d_model)
    │
    ▼
Linear(d_model, vocab_size) ──→ logits (B, S, vocab_size)
```

### 提示

```python
class TransformerLM(nn.Module):
    def __init__(self, vocab_size, context_length, d_model, num_layers, num_heads):
        super().__init__()
        self.token_embedding = Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads) for _ in range(num_layers)
        ])
        self.norm = RMSNorm(d_model)
        self.output = Linear(d_model, vocab_size)

    def forward(self, token_ids):
        x = self.token_embedding(token_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return self.output(x)
```

### 验证

```python
model = TransformerLM(vocab_size=1000, context_length=128, d_model=64, num_layers=2, num_heads=8)
token_ids = torch.randint(0, 1000, (2, 128))
logits = model(token_ids)
assert logits.shape == (2, 128, 1000)
```

---

## Step 2：train（训练循环）

### 规格

```
train(model, data, optimizer, batch_size, context_length, device, max_iters, log_interval):
  model: TransformerLM
  data: numpy array of token IDs
  optimizer: AdamW
  max_iters: 总训练步数
  log_interval: 每 N 步打印一次 loss

  for step in range(max_iters):
    1. x, y = get_batch(data, batch_size, context_length, device)
    2. logits = model(x)                        # (B, S, V)
    3. loss = F.cross_entropy(logits.view(-1, V), y.view(-1))
    4. loss.backward()
    5. gradient_clipping(model.parameters(), max_norm=1.0)
    6. optimizer.step()
    7. optimizer.zero_grad()
    8. if step % log_interval == 0: print loss
```

### 关键点

- **loss 计算**：`logits.view(-1, V)` 将 `(B, S, V)` 展平为 `(B*S, V)`，`y.view(-1)` 展平为 `(B*S,)`。这样 `cross_entropy` 对每个 token 位置独立计算 loss。
- **梯度裁剪在 optimizer.step() 之前**：先裁剪梯度，再更新参数。
- **optimizer.zero_grad() 在 step 之后**：PyTorch 默认累加梯度，每步需要清零。
- **学习率调度**：完整训练中应该在每步调用 `get_lr_cosine_schedule` 更新学习率。简化版可以省略。

### 训练循环流程

```
for step in range(max_iters):
    ┌─ get_batch ──→ x(B,S), y(B,S)
    │
    ├─ model(x) ──→ logits(B,S,V)
    │
    ├─ cross_entropy(logits, y) ──→ loss
    │
    ├─ loss.backward() ──→ 计算梯度
    │
    ├─ gradient_clipping ──→ 裁剪梯度
    │
    ├─ optimizer.step() ──→ 更新参数
    │
    └─ optimizer.zero_grad() ──→ 清零梯度
```

### 提示

```python
import torch.nn.functional as F

def train(model, data, optimizer, batch_size, context_length, device, max_iters, log_interval=100, max_norm=1.0):
    model.train()
    for step in range(max_iters):
        x, y = get_batch(data, batch_size, context_length, device)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        loss.backward()
        gradient_clipping(model.parameters(), max_norm)
        optimizer.step()
        optimizer.zero_grad()
        if step % log_interval == 0:
            print(f"step {step}: loss={loss.item():.4f}")
```

### 验证

```python
import numpy as np

model = TransformerLM(vocab_size=256, context_length=32, d_model=32, num_layers=2, num_heads=4)
optimizer = AdamW(model.parameters(), lr=1e-3)
data = np.random.randint(0, 256, 1000)
train(model, data, optimizer, batch_size=4, context_length=32, device='cpu', max_iters=10, log_interval=5)
# 应该看到 loss 从 ~5.5 逐渐下降
```

---

## 完整测试

将 `TransformerLM` 追加到 `model_components.py`，将 `train` 追加到 `training.py`，然后运行：

```bash
cd homework/assignment1
python tests/test_part7.py
```

### 测试脚本 `tests/test_part7.py`

```python
"""Part 7 完整 Transformer 语言模型测试"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

import torch
import numpy as np
from model_components import TransformerLM
from training import AdamW, train


def test_transformer_lm():
    """TransformerLM: 形状正确 + 参数可训练 + 前向传播"""
    vocab_size = 100
    context_length = 16
    d_model = 32
    num_layers = 2
    num_heads = 4

    model = TransformerLM(vocab_size, context_length, d_model, num_layers, num_heads)

    # 检查内部结构
    assert hasattr(model, 'token_embedding')
    assert hasattr(model, 'layers')
    assert hasattr(model, 'norm')
    assert hasattr(model, 'output')
    assert len(model.layers) == num_layers

    # 前向传播
    token_ids = torch.randint(0, vocab_size, (2, context_length))
    logits = model(token_ids)
    assert logits.shape == (2, context_length, vocab_size)

    # 梯度可传播
    loss = logits.sum()
    loss.backward()
    for p in model.parameters():
        assert p.grad is not None

    print("  [PASS] TransformerLM")


def test_transformer_lm_different_sizes():
    """不同配置的 TransformerLM"""
    configs = [
        (50, 8, 16, 1, 2),    # 小模型
        (200, 32, 64, 3, 8),  # 中等模型
    ]
    for vocab_size, ctx, d_model, n_layers, n_heads in configs:
        model = TransformerLM(vocab_size, ctx, d_model, n_layers, n_heads)
        x = torch.randint(0, vocab_size, (1, ctx))
        out = model(x)
        assert out.shape == (1, ctx, vocab_size), f"配置 {(vocab_size, ctx, d_model, n_layers, n_heads)} 失败"

    print("  [PASS] TransformerLM different sizes")


def test_train_loop():
    """训练循环: loss 应该下降"""
    model = TransformerLM(vocab_size=64, context_length=16, d_model=32, num_layers=1, num_heads=4)
    optimizer = AdamW(model.parameters(), lr=1e-3)
    data = np.random.randint(0, 64, 500)

    # 记录初始 loss
    model.eval()
    with torch.no_grad():
        x_init = torch.randint(0, 64, (4, 16))
        y_init = torch.randint(0, 64, (4, 16))
        init_logits = model(x_init)
        init_loss = torch.nn.functional.cross_entropy(
            init_logits.view(-1, 64), y_init.view(-1)
        ).item()

    # 训练
    train(model, data, optimizer, batch_size=4, context_length=16,
          device='cpu', max_iters=50, log_interval=25)

    # 训练后 loss 应该更低
    model.eval()
    with torch.no_grad():
        final_logits = model(x_init)
        final_loss = torch.nn.functional.cross_entropy(
            final_logits.view(-1, 64), y_init.view(-1)
        ).item()

    # loss 应该下降（至少不会上升）
    assert final_loss < init_loss, f"loss 未下降: {init_loss:.4f} → {final_loss:.4f}"

    print(f"  [PASS] train loop (loss: {init_loss:.4f} → {final_loss:.4f})")


if __name__ == "__main__":
    print("Part 7 完整 Transformer 语言模型测试")
    print("=" * 45)
    test_transformer_lm()
    test_transformer_lm_different_sizes()
    test_train_loop()
    print("=" * 45)
    print("全部测试通过!")
```

---

## 常见陷阱

| # | 陷阱 | 正确做法 |
|---|------|---------|
| 1 | 忘记最后的 RMSNorm | Pre-Norm 架构需要最终归一化 |
| 2 | output Linear 用了 bias | 与所有 Linear 一致，无 bias |
| 3 | loss 计算没有 view(-1, V) | 需要展平为 (B*S, V) 和 (B*S,) |
| 4 | gradient_clipping 在 backward 之前 | 应该在 backward 之后、step 之前 |
| 5 | 忘记 optimizer.zero_grad() | PyTorch 默认累加梯度 |
| 6 | 训练时没调 model.train() | 虽然当前无 dropout/batchnorm，但好习惯 |

---

## 完成标志

运行 `python tests/test_part7.py` 全部 3 项通过后，输入 `提交作业`。
