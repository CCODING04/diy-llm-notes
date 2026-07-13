"""Assignment 1 Part 4: Transformer 基础算子

从零实现的 7 个基础组件，不依赖 nn.Linear / nn.Embedding 等 PyTorch 内置层。
"""
import math
import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np


# ============================================================
# 1. Linear — 自定义线性层（无 bias）
# ============================================================
class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # TODO: 创建 weight Parameter，形状 (out_features, in_features)
        self.weight = nn.Parameter(torch.empty((out_features, in_features), device=device, dtype=dtype))
        # TODO: 调用 reset_parameters()
        self.reset_parameters()

    def reset_parameters(self):
        # TODO: σ = sqrt(2 / (in_features + out_features))
        # TODO: trunc_normal_(weight, mean=0, std=σ, a=-3σ, b=3σ)
        sigma = math.sqrt(2 / (self.in_features + self.out_features))
        init.trunc_normal_(self.weight, mean=0.0, std=sigma, a=-3*sigma, b=3*sigma)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: x @ weight.t()
        return x @ self.weight.t()


# ============================================================
# 2. Embedding — token ID → 行向量
# ============================================================
class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        # TODO: 创建 weights Parameter，形状 (num_embeddings, embedding_dim)
        # TODO: 调用 reset_parameters()
        self.weights = nn.Parameter(torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype))
        self.reset_parameters()

    def reset_parameters(self):
        # TODO: trunc_normal_(weights, mean=0, std=1.0, a=-3.0, b=3.0)
        init.trunc_normal_(self.weights, mean=0.0, std=1.0, a=-3.0, b=3.0)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # TODO: 返回 self.weights[token_ids]
        return self.weights[token_ids]


# ============================================================
# 3. RMSNorm — 仅缩放、不平移的归一化
# ============================================================
class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps=1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        # TODO: 创建 weight Parameter，形状 (d_model,)，初始化为全 1
        self.weight = nn.Parameter(torch.ones((d_model, ), device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: 保存原始 dtype
        # TODO: 转 float32
        # TODO: rms = rsqrt(mean(x², dim=-1) + eps)
        # TODO: x * rms * weight → 转回原始 dtype
        ori_dtype = x.dtype
        x = x.to(torch.float32)
        rms = torch.rsqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return (x * rms * self.weight).to(ori_dtype)


# ============================================================
# 4. SwiGLU — SiLU 门控 + GLU
# ============================================================
class SwiGLU(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        # TODO: 计算 d_ff = ceil64(int(8/3 * d_model))
        # TODO: self.w_gate = Linear(d_model, d_ff)
        # TODO: self.w_up   = Linear(d_model, d_ff)
        # TODO: self.w_down = Linear(d_ff, d_model)

        ceil64 = lambda x: ((x + 63) // 64) * 64
        self.w_gate = Linear(d_model, ceil64(int(8/3 * d_model)))
        self.w_up = Linear(d_model, ceil64(int(8/3 * d_model)))
        self.w_down = Linear(ceil64(int(8/3 * d_model)), d_model)



    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: gate = w_gate(x)
        # TODO: swish = gate * sigmoid(gate)
        # TODO: return w_down(swish * w_up(x))
        gate = self.w_gate(x)
        swish = gate * torch.sigmoid(gate)
        return self.w_down(swish * self.w_up(x))


# ============================================================
# 5. RotaryPositionalEmbedding — 旋转位置编码
# ============================================================
class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int):
        super().__init__()
        self.d_k = d_k
        self.theta = theta
        # TODO: register_buffer("cos", None, persistent=False)
        # TODO: register_buffer("sin", None, persistent=False)
        self.register_buffer("cos", None, persistent=False)
        self.register_buffer("sin", None, persistent=False)

    def _build_cache(self, seq_len, device, dtype):
        # TODO: 只在 cos 为 None 或 seq_len 超过缓存时才重新计算
        # TODO: powers = arange(0, d_k, 2) → inv_freq = theta^(-powers/d_k)
        # TODO: t = arange(seq_len) → freqs = outer(t, inv_freq)
        # TODO: register_buffer("cos", cos(freqs).to(dtype), persistent=False)
        # TODO: register_buffer("sin", sin(freqs).to(dtype), persistent=False)
        
        if self.cos is None or self.cos.shape[0] < seq_len:
            powers = torch.arange(0, self.d_k, 2, device=device)
            inv_freq = self.theta ** (-powers / self.d_k)
            t = torch.arange(seq_len, device=device)
            freqs = torch.outer(t, inv_freq)
            self.register_buffer("cos", torch.cos(freqs).to(dtype), persistent=False)
            self.register_buffer("sin", torch.sin(freqs).to(dtype), persistent=False)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, H, S, d_k)
        # TODO: assert d_k % 2 == 0
        # TODO: 调用 _build_cache(S, x.device, x.dtype)
        # TODO: cos/sin 取前 S 行 → reshape 适配广播
        # TODO: x_even = x[..., 0::2], x_odd = x[..., 1::2]
        # TODO: 旋转: x_even*cos - x_odd*sin, x_even*sin + x_odd*cos
        # TODO: 交错还原为 (B, H, S, d_k)
        B, H, S, d_k = x.shape
        assert d_k % 2 == 0, "d_k must be even"
        self._build_cache(S, x.device, x.dtype)
        x = x.reshape(-1, S, d_k)
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        cos = self.cos[:S, :].unsqueeze(0)  # (1, S, d_k//2)
        sin = self.sin[:S, :].unsqueeze(0)  # (1, S, d_k//2)
        x_rotated_even = x_even * cos - x_odd * sin
        x_rotated_odd = x_even * sin + x_odd * cos
        x_rotated = torch.stack((x_rotated_even, x_rotated_odd), dim=-1).reshape(B, H, S, d_k)
        return x_rotated


# ============================================================
# 6. softmax — 数值稳定的 softmax
# ============================================================
def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    # TODO: max_val = x.max(dim, keepdim=True)
    # TODO: e_x = exp(x - max_val)
    # TODO: return e_x / e_x.sum(dim, keepdim=True)
    max_val = x.max(dim=dim, keepdim=True).values
    e_x = torch.exp(x - max_val)
    return e_x / e_x.sum(dim=dim, keepdim=True)


# ============================================================
# 7. cross_entropy — 从零实现的交叉熵（numpy）
# ============================================================
def cross_entropy(logits: np.ndarray, targets: np.ndarray) -> float:
    """logits: (batch_size, vocab_size), targets: (batch_size,)"""
    # TODO: max_val = logits.max(axis=-1, keepdims=True)
    # TODO: shifted = logits - max_val
    # TODO: log_sum_exp = log(sum(exp(shifted), axis=-1))
    # TODO: target_logits = shifted 每行取 targets 位置的值
    # TODO: loss_i = log_sum_exp - target_logits
    # TODO: return mean(loss_i)
    # probs = softmax(torch.tensor(logits), dim=-1).numpy()
    max_val = logits.max(axis=-1, keepdims=True)
    shifted = logits - max_val
    log_sum_exp = np.log(np.sum(np.exp(shifted), axis=-1, keepdims=True))
    target_logits = shifted[np.arange(len(targets)), targets]
    loss_i = log_sum_exp.squeeze() - target_logits
    return loss_i.mean()


# ============================================================
# 8. scaled_dot_product_attention — 缩放点积注意力
# ============================================================
def scaled_dot_product_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask=None):
    """
    q: (..., seq_len, d_k)
    k: (..., seq_len, d_k)
    v: (..., seq_len, d_v)
    mask: (seq_len, seq_len) bool, True=保留, False=屏蔽
    返回: (output, attn_weights)
    """
    # TODO: d_k = q.size(-1)
    # TODO: scores = q @ k^T / sqrt(d_k)
    # TODO: if mask: scores.masked_fill(mask==False, -inf)
    # TODO: attn_weights = softmax(scores, dim=-1)
    # TODO: output = attn_weights @ v
    # TODO: return output, attn_weights
    d_k = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=q.dtype, device=q.device))
    if mask is not None:
        scores = scores.masked_fill(mask == False, float('-inf'))
    attn_weights = softmax(scores, dim=-1)
    output = torch.matmul(attn_weights, v)
    return output, attn_weights


# ============================================================
# 9. MultiHeadAttention — 因果多头注意力
# ============================================================
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"
        # TODO: self.d_model, self.num_heads, self.d_k
        # TODO: self.w_q, self.w_k, self.w_v, self.w_o = Linear(d_model, d_model)
        # TODO: self.rope = RotaryPositionalEmbedding(10000.0, d_k=self.d_k)
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        # self.qkv = Linear(d_model, d_model * 3)
        self.w_q = Linear(d_model, d_model)
        self.w_k = Linear(d_model, d_model)
        self.w_v = Linear(d_model, d_model)
        self.w_o = Linear(d_model, d_model)
        self.rope = RotaryPositionalEmbedding(10000.0, d_k=self.d_k)

    def forward(self, x: torch.Tensor, mask=None):
        # x: (B, S, d_model)
        # TODO: 1. Q = w_q(x) → view(B,S,H,d_k) → transpose(1,2) → (B,H,S,d_k)
        # TODO: 2. K = w_k(x) → 同上
        # TODO: 3. V = w_v(x) → 同上
        # TODO: 4. Q = rope(Q), K = rope(K)   # V 不做 RoPE
        # TODO: 5. out, _ = scaled_dot_product_attention(Q, K, V, mask)
        # TODO: 6. out = transpose(1,2) → contiguous() → view(B, S, d_model)
        # TODO: 7. return w_o(out)
        B, S, _ = x.shape
        q, k, v = self.w_q(x), self.w_k(x), self.w_v(x)  # (B, S, d_model)
        q = q.view(B, S, self.num_heads, self.d_k).transpose(1, 2)  # (B, H, S, d_k)
        k = k.view(B, S, self.num_heads, self.d_k).transpose(1, 2)  # (B, H, S, d_k)
        v = v.view(B, S, self.num_heads, self.d_k).transpose(1, 2)  # (B, H, S, d_k)
        q = self.rope(q)
        k = self.rope(k)
        out, _ = scaled_dot_product_attention(q, k, v, mask)
        out = out.transpose(1, 2).contiguous().view(B, S, self.d_model)  # (B, S, d_model)
        return self.w_o(out)

# ============================================================
# 10. TransformerBlock — Pre-Norm 残差块
# ============================================================
class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        # TODO: self.attention = MultiHeadAttention(d_model, num_heads)
        # TODO: self.ffn = SwiGLU(d_model)
        # TODO: self.norm1 = RMSNorm(d_model)
        # TODO: self.norm2 = RMSNorm(d_model)
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.ffn = SwiGLU(d_model)
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)

    def forward(self, x: torch.Tensor, mask=None):
        # TODO: Pre-Norm + Attention + 残差
        # TODO: Pre-Norm + FFN + 残差
        x = x + self.attention(self.norm1(x), mask)
        x = x + self.ffn(self.norm2(x))
        return x


# ============================================================
# 11. TransformerLM — 完整语言模型
# ============================================================
class TransformerLM(nn.Module):
    def __init__(self, vocab_size: int, context_length: int, d_model: int, num_layers: int, num_heads: int):
        super().__init__()
        # TODO: self.token_embedding = Embedding(vocab_size, d_model)
        # TODO: self.layers = nn.ModuleList([TransformerBlock(d_model, num_heads) for _ in range(num_layers)])
        # TODO: self.norm = RMSNorm(d_model)
        # TODO: self.output = Linear(d_model, vocab_size)
        self.token_embedding = Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([TransformerBlock(d_model, num_heads) for _ in range(num_layers)])
        self.norm = RMSNorm(d_model)
        self.output = Linear(d_model, vocab_size)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # token_ids: (B, S)
        # TODO: x = token_embedding(token_ids)       → (B, S, d_model)
        # TODO: for layer in self.layers: x = layer(x)
        # TODO: x = self.norm(x)
        # TODO: logits = self.output(x)              → (B, S, vocab_size)
        # TODO: return logits
        x = self.token_embedding(token_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        logits = self.output(x)
        return logits

    @torch.no_grad()
    def generate(self, token_ids: torch.Tensor, max_new_tokens: int, temperature: float = 1.0) -> torch.Tensor:
        # TODO: self.eval()
        # TODO: for _ in range(max_new_tokens):
        #   1. logits = self(token_ids)[:, -1, :] / temperature
        #   2. probs = softmax(logits, dim=-1)
        #   3. next_token = torch.multinomial(probs, num_samples=1)
        #   4. token_ids = torch.cat([token_ids, next_token], dim=1)
        # TODO: return token_ids
        self.eval()
        for _ in range(max_new_tokens):
            logits = self(token_ids)[:, -1, :] / temperature
            probs = softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            token_ids = torch.cat([token_ids, next_token], dim=1)
        return token_ids


# ============================================================
# 快速自测（python model_components.py）
# ============================================================
if __name__ == "__main__":
    print("测试 Linear...")
    model = Linear(6, 3)
    tw = torch.randn(3, 6)
    model.load_state_dict({'weight': tw})
    out = model(torch.randn(1, 6))
    assert out.shape == (1, 3), f"Linear 形状错误: {out.shape}"
    print("  OK")

    print("测试 Embedding...")
    emb = Embedding(10, 3)
    w = torch.randn(10, 3)
    emb.load_state_dict({'weights': w})
    out = emb(torch.tensor([[2, 5]]))
    assert out.shape == (1, 2, 3)
    print("  OK")

    print("测试 RMSNorm...")
    norm = RMSNorm(8)
    out = norm(torch.randn(2, 4, 8))
    assert out.shape == (2, 4, 8)
    print("  OK")

    print("测试 SwiGLU...")
    swiglu = SwiGLU(64)
    out = swiglu(torch.randn(2, 5, 64))
    assert out.shape == (2, 5, 64)
    print("  OK")

    print("测试 RoPE...")
    rope = RotaryPositionalEmbedding(10000.0, 8)
    out = rope(torch.randn(1, 1, 5, 8))
    assert out.shape == (1, 1, 5, 8)
    print("  OK")

    print("测试 softmax...")
    s = softmax(torch.tensor([[1.0, 2.0, 1000.0]]), dim=-1)
    assert not torch.isnan(s).any()
    print("  OK")

    print("测试 cross_entropy...")
    loss = cross_entropy(
        np.array([[2.0, 1.0, 0.1], [0.5, 2.5, 0.1]]),
        np.array([0, 1])
    )
    print(f"  loss={loss:.4f}")

    print("测试 scaled_dot_product_attention...")
    q = torch.randn(2, 3, 8)
    out, w = scaled_dot_product_attention(q, q, q)
    assert out.shape == (2, 3, 8)
    print("  OK")

    print("测试 MultiHeadAttention...")
    mha = MultiHeadAttention(64, 8)
    out = mha(torch.randn(2, 5, 64))
    assert out.shape == (2, 5, 64)
    print("  OK")

    print("测试 TransformerBlock...")
    block = TransformerBlock(64, 8)
    out = block(torch.randn(2, 5, 64))
    assert out.shape == (2, 5, 64)
    print("  OK")

    print("\n所有框架自测通过（完整测试请运行 python tests/test_part4.py 和 test_part5.py）")
