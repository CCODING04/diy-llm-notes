# Assignment 1 - Part 8：文本生成与端到端验证

> 📍 作业进度：Assignment 1，第 8 / 9 部分
> 📅 生成时间：2026-07-08
> 📎 原作业参考：`coursework/assignment1-basics/CS336_Assignment1_Transformer.ipynb`

---

## 目标与要求

实现文本生成函数，并通过端到端训练验证整个 Transformer 的正确性。

**实现文件**：
- `homework/assignment1/scripts/model_components.py` — 追加 `generate` 方法到 `TransformerLM`
- `homework/assignment1/scripts/training.py` — 追加 `evaluate` 函数

### 2 个组件清单

| # | 组件 | 类型 | 说明 |
|---|------|------|------|
| 1 | `TransformerLM.generate` | 方法 | 自回归文本生成：给定 prompt，逐 token 采样 |
| 2 | `evaluate` | 函数 | 计算验证集上的 perplexity |

---

## Step 1：generate（文本生成）

### 规格

```
TransformerLM.generate(token_ids, max_new_tokens, temperature=1.0):
  token_ids: (B, S) — 初始 token 序列（prompt）
  max_new_tokens: 生成的额外 token 数量
  temperature: 采样温度（>1 更随机，<1 更确定，0=贪心）

  for _ in range(max_new_tokens):
    1. logits = self(token_ids)              # (B, S, V)
    2. logits = logits[:, -1, :]             # 只看最后一个位置
    3. logits = logits / temperature         # 温度缩放
    4. probs = softmax(logits, dim=-1)       # 概率分布
    5. next_token = multinomial(probs, 1)    # 采样
    6. token_ids = cat([token_ids, next_token], dim=1)
  return token_ids
```

### 关键点

- **为什么只取 `logits[:, -1, :]`？** 自回归生成时，我们只关心下一个 token 的预测。最后一个位置的 logits 包含了整个序列的上下文信息。
- **温度的作用**：
  - `temperature=1.0`：原始分布，不缩放
  - `temperature→0`：趋向贪心（argmax），输出确定性最高
  - `temperature>1.0`：分布更平坦，输出更随机
  - 数学上：`softmax(z/T)` 中 T 越大，分布越均匀
- **`torch.multinomial`**：从概率分布中采样一个 token。`num_samples=1` 返回形状 `(B, 1)`。
- **`torch.cat` 而非 `torch.stack`**：因为是在 seq_len 维度上拼接，不是创建新维度。

### 生成流程图

```
prompt: [t1, t2, t3]  (B, 3)
    │
    ▼ model → logits[:, -1, :] → /temperature → softmax → sample
    │
    ▼ next_token = t4
    │
    ├──→ [t1, t2, t3, t4]  (B, 4)
    │
    ▼ model → logits[:, -1, :] → /temperature → softmax → sample
    │
    ▼ next_token = t5
    │
    └──→ [t1, t2, t3, t4, t5]  (B, 5)
    ...
```

### 提示

```python
@torch.no_grad()
def generate(self, token_ids: torch.Tensor, max_new_tokens: int, temperature: float = 1.0) -> torch.Tensor:
    self.eval()
    for _ in range(max_new_tokens):
        logits = self(token_ids)[:, -1, :] / temperature
        probs = softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        token_ids = torch.cat([token_ids, next_token], dim=1)
    return token_ids
```

### 验证

```python
model = TransformerLM(vocab_size=50, context_length=32, d_model=32, num_layers=1, num_heads=4)
prompt = torch.randint(0, 50, (1, 5))
output = model.generate(prompt, max_new_tokens=10, temperature=1.0)
assert output.shape == (1, 15)  # 5 + 10
# 贪心生成（temperature→0）应该确定性输出
output_greedy = model.generate(prompt, max_new_tokens=5, temperature=1e-8)
```

---

## Step 2：evaluate（困惑度评估）

### 规格

```
evaluate(model, data, batch_size, context_length, device, num_batches=10):
  model.eval()
  total_loss = 0
  for _ in range(num_batches):
    x, y = get_batch(data, batch_size, context_length, device)
    logits = model(x)
    loss = F.cross_entropy(logits.view(-1, V), y.view(-1))
    total_loss += loss.item()
  avg_loss = total_loss / num_batches
  perplexity = exp(avg_loss)
  return perplexity
```

### 关键点

- **困惑度（Perplexity）**：语言模型的核心评估指标。`PPL = exp(cross_entropy_loss)`。直觉上，PPL=k 表示模型在每个位置平均有 k 个等概率的选择。PPL 越低越好。
- **为什么用 `model.eval()`？** 关闭 dropout（虽然当前没有）和 batchnorm 的训练模式，确保评估结果确定性。
- **`@torch.no_grad()`**：评估不需要计算梯度，节省内存和计算。

### 公式

$$\text{PPL} = \exp\left(\frac{1}{N}\sum_{i=1}^{N} -\log p(x_i | x_{<i})\right) = \exp(\text{CE Loss})$$

### 提示

```python
@torch.no_grad()
def evaluate(model, data, batch_size, context_length, device, num_batches=10):
    model.eval()
    total_loss = 0.0
    for _ in range(num_batches):
        x, y = get_batch(data, batch_size, context_length, device)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        total_loss += loss.item()
    avg_loss = total_loss / num_batches
    return {"loss": avg_loss, "perplexity": math.exp(avg_loss)}
```

### 验证

```python
model = TransformerLM(vocab_size=50, context_length=16, d_model=32, num_layers=1, num_heads=4)
data = np.random.randint(0, 50, 500)
result = evaluate(model, data, batch_size=4, context_length=16, device='cpu', num_batches=5)
assert "loss" in result and "perplexity" in result
assert result["perplexity"] > 1.0  # 未训练的模型 PPL 应该较高
```

---

## 完整测试

将 `generate` 追加到 `TransformerLM`，将 `evaluate` 追加到 `training.py`，然后运行：

```bash
cd homework/assignment1
python tests/test_part8.py
```

### 测试脚本 `tests/test_part8.py`

```python
"""Part 8 文本生成与端到端验证测试"""
import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

import torch
import numpy as np
from model_components import TransformerLM, softmax
from training import AdamW, train, evaluate, get_batch


def test_generate():
    """generate: 输出形状 + 温度影响"""
    model = TransformerLM(vocab_size=50, context_length=32, d_model=32, num_layers=1, num_heads=4)
    prompt = torch.randint(0, 50, (1, 5))

    # 基本生成
    output = model.generate(prompt, max_new_tokens=10, temperature=1.0)
    assert output.shape == (1, 15), f"形状错误: {output.shape}"
    # 前 5 个 token 应该和 prompt 一致
    assert torch.equal(output[:, :5], prompt)

    # 贪心生成（temperature→0 应该确定性）
    out1 = model.generate(prompt, max_new_tokens=5, temperature=1e-8)
    out2 = model.generate(prompt, max_new_tokens=5, temperature=1e-8)
    assert torch.equal(out1, out2), "贪心生成应该确定性"

    # batch 生成
    prompt_batch = torch.randint(0, 50, (3, 5))
    out_batch = model.generate(prompt_batch, max_new_tokens=5)
    assert out_batch.shape == (3, 10)

    print("  [PASS] generate")


def test_evaluate():
    """evaluate: 返回 loss 和 perplexity"""
    model = TransformerLM(vocab_size=50, context_length=16, d_model=32, num_layers=1, num_heads=4)
    data = np.random.randint(0, 50, 500)

    result = evaluate(model, data, batch_size=4, context_length=16, device='cpu', num_batches=5)
    assert "loss" in result
    assert "perplexity" in result
    assert result["loss"] > 0
    assert result["perplexity"] > 1.0  # PPL > 1

    print(f"  [PASS] evaluate (loss={result['loss']:.4f}, ppl={result['perplexity']:.2f})")


def test_end_to_end():
    """端到端: 训练后 PPL 应该下降"""
    vocab_size = 50
    model = TransformerLM(vocab_size, context_length=16, d_model=32, num_layers=1, num_heads=4)
    optimizer = AdamW(model.parameters(), lr=1e-3)
    data = np.random.randint(0, vocab_size, 1000)

    # 训练前 PPL
    ppl_before = evaluate(model, data, batch_size=4, context_length=16, device='cpu', num_batches=5)["perplexity"]

    # 训练
    train(model, data, optimizer, batch_size=4, context_length=16,
          device='cpu', max_iters=100, log_interval=50)

    # 训练后 PPL
    ppl_after = evaluate(model, data, batch_size=4, context_length=16, device='cpu', num_batches=5)["perplexity"]

    assert ppl_after < ppl_before, f"PPL 未下降: {ppl_before:.2f} → {ppl_after:.2f}"

    # 生成测试
    prompt = torch.randint(0, vocab_size, (1, 5))
    output = model.generate(prompt, max_new_tokens=10, temperature=0.8)
    assert output.shape == (1, 15)

    print(f"  [PASS] end_to_end (PPL: {ppl_before:.2f} → {ppl_after:.2f})")


if __name__ == "__main__":
    print("Part 8 文本生成与端到端验证测试")
    print("=" * 45)
    test_generate()
    test_evaluate()
    test_end_to_end()
    print("=" * 45)
    print("全部测试通过!")
```

---

## 常见陷阱

| # | 陷阱 | 正确做法 |
|---|------|---------|
| 1 | generate 没用 `@torch.no_grad()` | 生成不需要梯度，会浪费大量内存 |
| 2 | generate 用 `argmax` 替代 `multinomial` | `multinomial` 支持随机采样，`argmax` 是贪心（用 temperature=0 控制） |
| 3 | temperature=0 导致除零 | 用 `temperature=1e-8` 近似贪心 |
| 4 | evaluate 没调 `model.eval()` | 虽然当前无 dropout，但好习惯 |
| 5 | perplexity 计算用 `loss` 而非 `exp(loss)` | PPL = exp(CE_loss) |

---

## 完成标志

运行 `python tests/test_part8.py` 全部 3 项通过后，输入 `提交作业`。
