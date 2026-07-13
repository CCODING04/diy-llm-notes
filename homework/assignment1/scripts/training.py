"""Assignment 1 Part 6: 训练基础设施

从零实现的 5 个训练组件：AdamW、余弦调度、梯度裁剪、数据加载、检查点。
"""
import math
import torch
import torch.nn.functional as F
import numpy as np
from torch.optim import Optimizer


# ============================================================
# 1. AdamW — 解耦权重衰减的 Adam 优化器
# ============================================================
class AdamW(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        # TODO: 参数校验（lr>0, eps>0, 0<=beta<1）
        # TODO: defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        # TODO: super().__init__(params, defaults)
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def _get_options(self, group):
        lr = group['lr']
        betas = group['betas']
        eps = group['eps']
        weight_decay = group['weight_decay']
        return lr, betas, eps, weight_decay

    @torch.no_grad()
    def step(self, closure=None):
        # TODO: for group in self.param_groups:
        #   for p in group['params']:
        #     if p.grad is None: continue
        #     state = self.state[p]
        #     if len(state) == 0:
        #       state['step'] = 0
        #       state['exp_avg'] = torch.zeros_like(p)      # 一阶矩 m
        #       state['exp_avg_sq'] = torch.zeros_like(p)   # 二阶矩 v
        #
        #     state['step'] += 1
        #     t = state['step']
        #     m, v = state['exp_avg'], state['exp_avg_sq']
        #     β1, β2 = group['betas']
        #
        #     # 更新矩估计
        #     m.mul_(β1).add_(p.grad, alpha=1-β1)
        #     v.mul_(β2).addcmul_(p.grad, p.grad, value=1-β2)
        #
        #     # 偏置修正
        #     bias_corr1 = 1 - β1**t
        #     bias_corr2 = 1 - β2**t
        #
        #     # 参数更新
        #     step_size = lr * sqrt(bias_corr2) / bias_corr1
        #     p.addcdiv_(m, v.sqrt().add_(eps), value=-step_size)
        #
        #     # 解耦权重衰减（在 Adam 更新之后！）
        #     if weight_decay != 0:
        #       p.add_(p, alpha=-lr * weight_decay)

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr, (beta1, beta2), eps, wd = self._get_options(group)
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]

                if len(state) == 0:
                    state['t'] = 0
                    state['m'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['v'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                m, v, t = state['m'], state['v'], state['t']
                t += 1

                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                bias_corr1 = 1 - beta1 ** t
                bias_corr2 = 1 - beta2 ** t
                m_hat = m / bias_corr1
                v_hat = v / bias_corr2

                if wd != 0:
                    p.mul_(1 - lr * wd)

                denom = v_hat.sqrt().add_(eps)
                p.addcdiv_(m_hat, denom, value=-lr)

        return loss


# ============================================================
# 2. get_lr_cosine_schedule — 余弦退火学习率调度
# ============================================================
def get_lr_cosine_schedule(t: int, alpha_max: float, alpha_min: float, T_w: int, T_c: int) -> float:
    # TODO: if t < T_w: linear warmup
    # TODO: elif t <= T_c: cosine decay
    # TODO: else: alpha_min
    if t < T_w:
        return (t / T_w) * alpha_max
    elif t <= T_c:
        return (1 + math.cos(math.pi * (t - T_w) / (T_c - T_w))) * 0.5 * (alpha_max - alpha_min) + alpha_min
    else:
        return alpha_min


# ============================================================
# 3. gradient_clipping — 全局梯度范数裁剪
# ============================================================
def gradient_clipping(parameters, max_norm: float, eps: float = 1e-6):
    # TODO: 收集有梯度的参数
    # TODO: 计算全局梯度范数 total_norm
    # TODO: clip_coeff = max_norm / (total_norm + eps)
    # TODO: if clip_coeff < 1: 所有梯度 *= clip_coeff
    params_with_grad = [p for p in parameters if p.grad is not None]
    if not params_with_grad:
        return
    total_norm = torch.sum(torch.stack([torch.norm(p.grad.detach(), 2) ** 2 for p in params_with_grad])) ** 0.5
    clip_coeff = max_norm / (total_norm + eps)
    if clip_coeff < 1:
        for p in params_with_grad:
            p.grad.detach().mul_(clip_coeff)
    


# ============================================================
# 4. get_batch — 从 token 序列中采样训练 batch
# ============================================================
def get_batch(data: np.ndarray, batch_size: int, context_length: int, device: str):
    # TODO: max_idx = len(data) - context_length
    # TODO: ix = torch.randint(0, max_idx, (batch_size,))
    # TODO: x = stack([data[i:i+context_length] for i in ix])
    # TODO: y = stack([data[i+1:i+context_length+1] for i in ix])
    # TODO: return x.to(device), y.to(device)
    max_idx = len(data) - context_length
    ix = torch.randint(0, max_idx, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i + context_length].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1:i + context_length + 1].astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)


# ============================================================
# 5. save_checkpoint / load_checkpoint — 检查点
# ============================================================
def save_checkpoint(model, optimizer, iteration: int, out: str):
    # TODO: checkpoint = {'model_state_dict': ..., 'optimizer_state_dict': ..., 'iteration': ...}
    # TODO: torch.save(checkpoint, out)
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'iteration': iteration
    }
    torch.save(checkpoint, out)


def load_checkpoint(src: str, model, optimizer):
    # TODO: checkpoint = torch.load(src, map_location='cpu')
    # TODO: model.load_state_dict(checkpoint['model_state_dict'])
    # TODO: optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # TODO: return checkpoint['iteration']
    checkpoint = torch.load(src, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['iteration']


# ============================================================
# 6. train — 训练循环
# ============================================================
def train(model, data, optimizer, batch_size, context_length, device, max_iters, log_interval=100, max_norm=1.0):
    # TODO: model.train()
    # TODO: for step in range(max_iters):
    #   1. x, y = get_batch(data, batch_size, context_length, device)
    #   2. logits = model(x)
    #   3. loss = F.cross_entropy(logits.view(-1, V), y.view(-1))
    #   4. loss.backward()
    #   5. gradient_clipping(model.parameters(), max_norm)
    #   6. optimizer.step()
    #   7. optimizer.zero_grad()
    #   8. if step % log_interval == 0: print loss
    for step in range(max_iters):
        x, y = get_batch(data, batch_size, context_length, device)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        loss.backward()
        gradient_clipping(model.parameters(), max_norm)
        optimizer.step()
        optimizer.zero_grad()
        if step % log_interval == 0:
            print(f"Step {step}: loss={loss.item():.4f}")


# ============================================================
# 7. evaluate — 计算困惑度
# ============================================================
@torch.no_grad()
def evaluate(model, data, batch_size, context_length, device, num_batches=10):
    # TODO: model.eval()
    # TODO: total_loss = 0
    # TODO: for _ in range(num_batches):
    #   x, y = get_batch(...)
    #   logits = model(x)
    #   loss = F.cross_entropy(logits.view(-1, V), y.view(-1))
    #   total_loss += loss.item()
    # TODO: avg_loss = total_loss / num_batches
    # TODO: return {"loss": avg_loss, "perplexity": math.exp(avg_loss)}
    model.eval()
    total_loss = 0
    for _ in range(num_batches):
        x, y = get_batch(data, batch_size, context_length, device)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        total_loss += loss.item()
    avg_loss = total_loss / num_batches
    return {"loss": avg_loss, "perplexity": math.exp(avg_loss)}


# ============================================================
# 快速自测（python training.py）
# ============================================================
if __name__ == "__main__":
    print("测试 AdamW...")
    model = torch.nn.Linear(3, 2, bias=False)
    opt = AdamW(model.parameters(), lr=1e-3)
    loss = model(torch.randn(4, 3)).sum()
    loss.backward()
    opt.step()
    opt.zero_grad()
    print("  OK")

    print("测试 get_lr_cosine_schedule...")
    lr = get_lr_cosine_schedule(50, 1e-3, 1e-5, 100, 1000)
    print(f"  lr={lr:.6f}")

    print("测试 gradient_clipping...")
    p = torch.tensor([1.0, 2.0], requires_grad=True)
    p.grad = torch.tensor([3.0, 4.0])
    gradient_clipping([p], max_norm=1.0)
    print(f"  grad_norm={torch.norm(p.grad).item():.4f}")

    print("测试 get_batch...")
    data = np.arange(100, dtype=np.int64)
    x, y = get_batch(data, 4, 10, 'cpu')
    print(f"  x.shape={x.shape}, y.shape={y.shape}")

    print("测试 save/load_checkpoint...")
    import tempfile
    path = tempfile.mktemp(suffix='.pt')
    save_checkpoint(model, opt, 42, path)
    model2 = torch.nn.Linear(3, 2, bias=False)
    opt2 = AdamW(model2.parameters(), lr=1e-3)
    it = load_checkpoint(path, model2, opt2)
    print(f"  iteration={it}")
    import os; os.remove(path)

    print("\n所有框架自测通过（完整测试请运行 python tests/test_part6.py）")
