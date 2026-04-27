# 第 5 章：混合专家模型 — 模块 3：MoE 在 LLM 中的实践

> 📍 学习进度：第 5 章，第 3 / 4 模块
> 📅 生成时间：2026-04-24（重构版）

---

## 学习目标

- 理解 MoE 替换 FFN 而非 Attention 的原因
- 掌握 Mini LLM + MoE 每个组件的设计决策（为什么这样设计、有哪些替代方案）
- 能逐行追踪 MoELayer 的完整数据流（展平 → 路由 → 分发 → 专家计算 → 加权合并）
- 理解 Pre-Norm 残差结构如何保证被丢弃 token 的信息不丢失

---

## 核心内容

### 一、MoE 在 Transformer 中的位置

MoE 不是独立模型，而是替换 Transformer Block 中的 **FFN（前馈网络）**，保留 Attention 不变：

```
标准 Transformer Block:              MoE Transformer Block:
━━━━━━━━━━━━━━━━━━━━━━━              ━━━━━━━━━━━━━━━━━━━━━━━

  x → LayerNorm                       x → LayerNorm
    → Self-Attention                    → Self-Attention       ← 不变
    → 残差 + Dropout                    → 残差 + Dropout
  → LayerNorm                         → LayerNorm
    → FFN (d → 4d → d)                → MoE Layer            ← 替换这里
    → 残差 + Dropout                    → 残差 + Dropout

FFN 参数量:  2 × d × 4d = 8d²
MoE 参数量:  N × 8d²     (N 个专家)
激活参数量:  k × 8d²     (每次只激活 k 个)
```

**为什么替换 FFN 而不是 Attention？**

这个问题值得深思。Transformer 中有两个核心子层——Attention 和 FFN——它们承担完全不同的职责：

```
Attention（注意力层）:
  作用: Token 之间的信息交换——"谁该看谁"
  计算: Q·K^T → softmax → × V，O(n²d) 复杂度
  性质: 本质是路由/调度，不存储知识

FFN（前馈层）:
  作用: 对每个 token 独立做非线性变换——"如何理解这个 token"
  计算: x → W₁ → activation → W₂，O(nd²) 复杂度
  性质: 本质是知识存储，参数量占全层 ~2/3
```

关键洞察：**FFN 是知识的"仓库"，MoE 的目标是扩大仓库容量**。用 N 个专家替代 1 个 FFN，每个专家可以专门存储某类知识（如数学、语法、多语言），路由器按需调用。而 Attention 负责的是"token 间通信"，把它换成 MoE 没有意义——所有 token 都需要 attend 到其他 token。

实验也验证了这一点：GShard（Google, 2020）和 Switch Transformer（Google, 2022）都只替换 FFN，保留 Attention。后续的 Mixtral 8x7B、DeepSeek-V2 也遵循这个设计。

**哪些层用 MoE？**

代码中的 `use_moe_layer_index` 参数控制这一选择：

```python
# 来自 [Mini LLM+MoE.py](<../Mini LLM+MoE.py>)
model = MiniMoELLModel(
    ...
    use_moe_layer_index=[1,3],   # 第 1、3 层用 MoE（交替策略）
    moe_params=moe_params
)
```

三种常见策略的权衡：

```
方案 A：全部层用 MoE
  Layer 0~3: 全 MoE
  风险: 浅层处理基础语法（词性、句法），专家分化不明显
       → MoE 的参数开销可能浪费，训练不稳定

方案 B：交替使用（本代码的选择）✅
  Layer 0: FFN    ← 浅层用标准 FFN 学通用特征
  Layer 1: MoE    ← 深层用 MoE 做专家分工
  Layer 2: FFN
  Layer 3: MoE
  理由: 深层的表征更抽象，更需要专家分化

方案 C：仅最后 N/2 层用 MoE
  被 DeepSeek-V2/V3、Mixtral 等主流模型采用
  理由: 浅层特征（语法、词形）各语言/领域差异小，不需要专家
```

> 💡 **补充（Context7 / PyTorch）**：PyTorch 已原生支持 MoE 的分布式通信原语 `torch.ops.symm_mem.all_to_all_vdev_2d`，用于多 GPU 间的 token 分发。这说明 MoE 已从研究热点变成框架级原生支持的架构。

---

### 二、ByteTokenizer：最简单的分词方案

来自 [Mini LLM+MoE.py](<../Mini LLM+MoE.py>)：

```python
class ByteTokenizer:
    def __init__(self):
        self.vocab_size = 259       # 256 字节 + 3 特殊 token
        self.bos = 256              # <bos> 序列开始
        self.eos = 257              # <eos> 序列结束
        self.pad = 258              # <pad> 批处理填充

    def encode(self, text, add_bos=True, add_eos=True):
        b = text.encode('utf-8', errors='surrogatepass')
        ids = list(b)               # 每个字节 → token id (0~255)
        if add_bos:
            ids = [self.bos] + ids
        if add_eos:
            ids = ids + [self.eos]
        return ids
```

**为什么用字节级分词而不是 BPE？**

| 方案 | 词表大小 | OOV | 序列长度 | 实现复杂度 | 需要预训练？ |
|------|---------|-----|---------|-----------|-----------|
| 传统字符级 BPE | 32K~100K | 极罕见（字符级有兜底） | 短 | 需要训练合并规则 | 是 |
| 现代字节级 BPE（GPT-2/4） | 32K~100K | **零**（256 字节兜底） | 短 | 需要训练合并规则 | 是 |
| 字节级（本代码） | 259 | **零** | 长（中文 1 字 = 3 token） | 极简（30 行） | **否** |

> ⚠️ **纠正**：上表中"BPE 有 OOV"的说法不够准确。现代字节级 BPE（GPT-2/3/4 使用的方案）以 256 个字节为基础词表，对任何未见过的词会退化为单字节序列，因此**也没有 OOV**。两者的核心差异不是 OOV，而是**压缩率**和**是否需要预训练**：
>
> - **BPE**：对高频词合并后序列更短（如 "cat" → 1 token），但需要在大规模语料上预训练合并规则
> - **字节级**：永远按字节拆分（如 "cat" → 3 token），但零依赖、零预训练
>
> 本代码选择字节级的理由：教学代码目标是展示 MoE 而非分词。字节级分词器零依赖、代码不超过 30 行，让学习者聚焦 MoE 本身。代价是序列变长——但对教学 demo 的小规模输入（几十个 token）没有影响。

`batch_encode` 方法处理批处理中的长度不一致：

```python
    def batch_encode(self, texts, pad_to=None):
        encs = [self.encode(t) for t in texts]
        maxlen = max(len(x) for x in encs)
        arr = [x + [self.pad] * (maxlen - len(x)) for x in encs]  # 短序列右填充
        lengths = torch.LongTensor([len(x) for x in encs])
        return torch.LongTensor(arr), lengths
```

示例，4 条不同长度的文本批处理：

```
"Hello MoE!"           → [256, 72, 101, 108, 108, 111, 32, 77, 111, 69, 33, 257]  长度 12
"你好！😆"              → [256, 228, 189, 160, 229, 165, 189, ...]  长度更长（UTF-8 多字节）
                         → 短的右边补 <pad>=258 对齐
lengths 保留每条的真实长度，用于评估时忽略 padding
```

---

### 三、SimpleSelfAttention：标准多头注意力

```python
# 来自 [Mini LLM+MoE.py](<../Mini LLM+MoE.py>)
class SimpleSelfAttention(nn.Module):
    def __init__(self, d_model, nhead):
        super().__init__()
        self.nhead = nhead
        self.d_k = d_model // nhead          # 每头维度
        self.qkv = nn.Linear(d_model, d_model * 3)  # 一次算出 Q,K,V
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        B, T, D = x.shape
        qkv = self.qkv(x)                             # [B, T, 3D]
        q, k, v = qkv.chunk(3, dim=-1)                # 各 [B, T, D]
        q = q.view(B, T, self.nhead, self.d_k).transpose(1, 2)  # [B, H, T, d_k]
        k = k.view(B, T, self.nhead, self.d_k).transpose(1, 2)
        v = v.view(B, T, self.nhead, self.d_k).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            attn_mask = (~(mask.bool().unsqueeze(1).unsqueeze(2))) * -1e9
            scores = scores + attn_mask                # padding 位置 → -inf → softmax 后 ≈ 0
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)                    # [B, H, T, d_k]
        out = out.transpose(1, 2).contiguous().view(B, T, D)  # [B, T, D]
        return self.out(out)
```

**关键设计：合并 QKV 投影**

`self.qkv = nn.Linear(d_model, d_model * 3)` 把三个投影合并成一个矩阵乘法。这比分别创建 `self.q_proj, self.k_proj, self.v_proj` 更高效：

```
分开投影（3 次矩阵乘法）:
  Q = x @ W_q    [B,T,D] @ [D,D] → [B,T,D]
  K = x @ W_k    [B,T,D] @ [D,D] → [B,T,D]
  V = x @ W_v    [B,T,D] @ [D,D] → [B,T,D]
  总 FLOPs: 3 × BTD²

合并投影（1 次矩阵乘法 + chunk）:
  QKV = x @ W_qkv  [B,T,D] @ [D,3D] → [B,T,3D]
  Q, K, V = chunk(3)                   各 [B,T,D]
  总 FLOPs: 3BTD²（相同）
  但 GPU 上 1 次大矩阵乘法比 3 次小的快（kernel launch 少、cache 友好）
```

**数据流（d_model=256, nhead=4）**：

```
x: [B, T, 256]
  → qkv: [B, T, 768]                 ← 1 次线性变换
  → chunk: q,k,v 各 [B, T, 256]
  → reshape + transpose: [B, 4, T, 64]  ← 4 个头，每头 64 维
  → scores: [B, 4, T, T]            ← 注意力矩阵（每对 token 的相关性）
  → attn × v: [B, 4, T, 64]
  → concat: [B, T, 256]
  → out: [B, T, 256]                ← 输出维度 = 输入维度
```

---

### 四、MoE 层：核心实现（逐段拆解）

这是整个模型的关键组件。我们将 `MoELayer` 拆成 5 个逻辑步骤逐步讲解。

来自 [Mini LLM+MoE.py](<../Mini LLM+MoE.py>)：

#### 4.1 初始化：门控网络 + 专家网络

```python
class MoELayer(nn.Module):
    def __init__(self, d_model, d_ff, n_experts=4, k=1,
                 capacity_factor=1.25, noisy_gating=True):
        super().__init__()
        self.n_experts = n_experts
        self.k = k                              # Top-1 或 Top-2
        self.capacity_factor = capacity_factor   # 容量系数
        self.noisy_gating = noisy_gating

        # 门控网络（路由器）：d_model 维 → n_experts 个得分
        self.w_gating = nn.Linear(d_model, n_experts, bias=False)
        if noisy_gating:
            self.w_noise = nn.Linear(d_model, n_experts, bias=False)

        # N 个专家网络，每个是标准 FFN
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_ff),    # 升维: d → 4d
                nn.GELU(),                    # 非线性激活
                nn.Linear(d_ff, d_model)     # 降维: 4d → d
            ) for _ in range(n_experts)
        ])
```

**设计决策解析**：

**为什么门控网络没有 bias？** `nn.Linear(d_model, n_experts, bias=False)` —— 因为 bias 会让路由器对某些专家有固有偏好。无 bias 保证路由完全由输入特征决定，训练初期各专家获得均等机会。如果加了 bias，初始化时某个专家的 bias 偏大，所有 token 都会被路由过去，其他专家"饿死"。

**为什么专家用 GELU 而不是 ReLU？**

```
ReLU:   f(x) = max(0, x)
        在 x=0 处不可微，梯度要么 0 要么 1
        → MoE 中大量 token 被路由到同一个专家时，ReLU 的"死神经元"问题更严重

GELU:   f(x) = x · Φ(x)    （Φ 是标准正态 CDF）
        处处平滑可微，x<0 时仍有小梯度
        → 梯度流更稳定，尤其适合 MoE 这种"稀疏激活"场景
```

GELU 是 Transformer 时代的默认激活函数（GPT-2、BERT、LLaMA 都用它）。在 MoE 中更重要：因为每个专家只处理一部分 token，如果用 ReLU 导致某些神经元永久失活，专家的"有效容量"会打折。GELU 的平滑性让每个神经元都能持续接收梯度信号。

> 💡 **补充（Context7 / PyTorch）**：PyTorch 的 `aten.gelu` 和 `aten.gelu_backward` 都有原生 CUDA 实现，在 FP16 混合精度训练中自动使用融合 kernel，性能优于手动实现。

**为什么用 `nn.ModuleList` 而不是普通 Python List？**

```python
# ✅ 正确：nn.ModuleList
self.experts = nn.ModuleList([nn.Sequential(...) for _ in range(n_experts)])

# ❌ 错误：普通 List
self.experts = [nn.Sequential(...) for _ in range(n_experts)]
```

`nn.ModuleList` 会将所有专家注册为模型的子模块，`model.parameters()` 能遍历到它们，`model.to(device)` 能把所有专家搬到 GPU。普通 Python List 不行——专家的参数会"丢失"，不参与训练。

#### 4.2 噪声门控：训练时探索、推理时确定

```python
    def _noisy_logits(self, x):
        logits = self.w_gating(x)                      # [N, n_experts] 路由得分
        if self.noisy_gating and self.training:         # 只在训练时加噪声
            noise_std = torch.sigmoid(self.w_noise(x))  # [N, n_experts], 值域 (0, 1)
            logits = logits + torch.randn_like(logits) * noise_std
        return logits
```

对照 Shazeer 原始论文的噪声门控公式：

```
公式:  H(x)_i = (x · W_g)_i + StandardNormal() · σ(x · W_noise)_i

代码:  logits = w_gating(x) + randn_like(logits) * sigmoid(w_noise(x))

对应关系:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
x · W_g        →  w_gating(x)            可学习路由得分
StandardNormal →  torch.randn_like        标准正态随机数
σ(x·W_noise)   →  sigmoid(w_noise(x))    可学习噪声幅度
```

**为什么噪声只在训练时加？**

噪声的作用是**强制所有专家都能收到 token，防止路由坍缩**。训练时加噪声让路由器"探索"，确保每个专家都得到训练信号。推理时模型已经训练好，路由器应该做出确定性选择——加噪声只会引入随机波动，降低输出质量。

`self.training` 是 PyTorch 的内置标志：`model.train()` 时为 True，`model.eval()` 时为 False。这是标准的训练/推理行为切换机制。

**代码用 Sigmoid 而原始论文用 Softplus**：

```
Sigmoid:  σ(x) = 1/(1+e⁻ˣ)    值域 (0, 1)   ← 本代码的简化版
Softplus: ln(1+eˣ)             值域 (0, +∞)  ← Shazeer 原始论文

差异: Sigmoid 的噪声标准差最大为 1，Softplus 无上界
     → Softplus 给模型更大的自由度：对"确定"的 token 可以学到 σ≈0（几乎无噪声）
     → Sigmoid 下限约 0.5，无法完全关闭噪声
本代码用 Sigmoid 是教学简化，实际工程中建议用 Softplus
```

#### 4.3 Top-1 路由 + 容量控制

这是 MoE 的核心——决定每个 token 去哪个专家，以及每个专家能处理多少 token：

```python
    def forward(self, x, mask=None):
        B, T, D = x.shape
        N = B * T                          # 总 token 数
        x_flat = x.view(N, D)              # 展平为 [N, D]

        logits = self._noisy_logits(x_flat)
        scores = F.softmax(logits, dim=-1)  # [N, n_experts] 每个 token 对各专家的路由概率

        # === Top-1 路由 ===
        top1 = torch.argmax(scores, dim=-1)                         # [N] 每个 token 选 1 个专家
        dispatch_mask = F.one_hot(top1, num_classes=self.n_experts)  # [N, E] one-hot 分配矩阵
        combine_weights = torch.gather(scores, 1, top1.unsqueeze(1)).squeeze(1)  # [N]

        # === 容量控制 ===
        capacity = int((N / self.n_experts) * self.capacity_factor) + 1
```

**Step-by-step 数据流**（4 专家, 10 token, capacity_factor=1.25）：

```
① softmax 路由概率 scores [10, 4]:
  Token 0: [0.35, 0.25, 0.20, 0.20]  → argmax → Expert 0, weight=0.35
  Token 1: [0.10, 0.60, 0.15, 0.15]  → argmax → Expert 1, weight=0.60
  Token 2: [0.40, 0.20, 0.30, 0.10]  → argmax → Expert 0, weight=0.40
  Token 3: [0.05, 0.10, 0.15, 0.70]  → argmax → Expert 3, weight=0.70
  Token 4: [0.30, 0.25, 0.35, 0.10]  → argmax → Expert 2, weight=0.35
  Token 5: [0.45, 0.15, 0.25, 0.15]  → argmax → Expert 0, weight=0.45
  Token 6: [0.10, 0.55, 0.20, 0.15]  → argmax → Expert 1, weight=0.55
  Token 7: [0.20, 0.20, 0.50, 0.10]  → argmax → Expert 2, weight=0.50
  Token 8: [0.35, 0.25, 0.15, 0.25]  → argmax → Expert 0, weight=0.35
  Token 9: [0.15, 0.10, 0.60, 0.15]  → argmax → Expert 2, weight=0.60

② 路由分配结果:
  Expert 0 ← Token 0, 2, 5, 8     (4 个)
  Expert 1 ← Token 1, 6           (2 个)
  Expert 2 ← Token 4, 7, 9        (3 个)
  Expert 3 ← Token 3              (1 个)

③ 容量计算:
  capacity = (10 / 4) × 1.25 + 1 = 3 + 1 = 4
  → 每个专家最多处理 4 个 token
  → 此例恰好无丢弃。如果 Expert 0 有 5 个 token → 第 5 个被丢弃
```

**`combine_weights` 是什么？为什么用 `torch.gather` 而不是 `scores.max`？**

```python
# 代码的做法:
combine_weights = torch.gather(scores, 1, top1.unsqueeze(1)).squeeze(1)

# 你可能想到的替代:
combine_weights = scores.max(dim=-1).values
```

两者结果相同（Top-1 下 argmax 位置的值 = max 值），但 `torch.gather` 更通用：

```
scores = [[0.35, 0.25, 0.20, 0.20],   top1 = [0, 1, ...]
          [0.10, 0.60, 0.15, 0.15],   ...]

torch.gather(scores, 1, top1.unsqueeze(1)):
  对每一行，取出 top1 指定位置的值
  → [0.35, 0.60, ...]

scores.max(dim=-1).values:
  对每一行取最大值
  → [0.35, 0.60, ...]

结果相同！但如果将来改为 Top-2 路由:
  gather 可以取任意位置的值（第 1 大、第 2 大）
  max 只能取最大值
→ gather 更具扩展性
```

`combine_weights` 的作用：**加权合并**。路由器给出的分数不仅是"选谁"的依据，还是"以多大力度混合"的权重。直觉上：如果 Token A 给 Expert 0 的分数是 0.9（很确定），Token B 给 Expert 0 的分数是 0.3（不太确定），那 Expert 0 对 A 的输出应该有更大影响。

#### 4.4 Token 分发 + 专家计算 + 容量截断

```python
        # Step 1: 为每个专家收集输入 token
        expert_inputs = []
        expert_indices = []
        for e in range(self.n_experts):
            # 找出所有路由到专家 e 的 token 索引
            idx = torch.nonzero(dispatch_mask[:, e], as_tuple=False).squeeze(-1)
            if idx.numel() > capacity:           # 超出容量！
                idx = idx[:capacity]              # 截断——丢弃多余的 token
            expert_inputs.append(x_flat[idx])
            expert_indices.append(idx)

        # Step 2: 每个专家独立处理自己的 token
        out_flat = torch.zeros_like(x_flat)       # 初始化为全零 [N, D]
        for e in range(self.n_experts):
            if expert_inputs[e].size(0) == 0:
                continue                          # 该专家无 token → 跳过
            y = self.experts[e](expert_inputs[e]) # 专家 FFN 处理
            out_flat[expert_indices[e]] = y       # 写回对应位置

        # Step 3: 加权输出
        out_flat = out_flat * combine_weights.unsqueeze(1)  # 乘以路由权重
        return out_flat.view(B, T, D)
```

**Token 丢弃的执行细节**（假设 Expert 0 收到 5 个 token，capacity=4）：

```
dispatch_mask[:, 0] = [1, 0, 1, 0, 0, 1, 0, 0, 1, 1]  (Token 0,2,5,8,9 选中了 Expert 0)
                       ↓
idx = [0, 2, 5, 8, 9]   (5 个 token)  →  5 > capacity(4)  → 截断！
idx = [0, 2, 5, 8]       (只取前 4 个)
                       ↓
Token 9 被丢弃：不进入 expert_inputs → 不被 Expert 0 处理
                       ↓
out_flat 初始化为全零 [N, D]
Expert 0 处理后 → out_flat[0], out_flat[2], out_flat[5], out_flat[8] 被写入
out_flat[9] 保持为 0
                       ↓
out_flat = out_flat * combine_weights  (逐元素乘)
out_flat[9] = 0 × weight = 0  ← 仍然为零
                       ↓
返回 view(B, T, D) → 被丢弃 token 的 MoE 输出 = 全零向量
```

**被丢弃 token 的信息去了哪里？** 这需要结合下一节 TransformerBlock 的残差连接来理解。

---

### 五、TransformerBlock：Pre-Norm 残差结构

```python
# 来自 [Mini LLM+MoE.py](<../Mini LLM+MoE.py>)
class TransformerBlock(nn.Module):
    def __init__(self, d_model, nhead, d_ff, use_moe=False, moe_params=None, dropout=0.1):
        super().__init__()
        self.attn = SimpleSelfAttention(d_model, nhead)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.use_moe = use_moe

        if use_moe:
            self.moe = MoELayer(**moe_params)        # MoE 替代 FFN
        else:
            self.ffn = nn.Sequential(                 # 标准 FFN
                nn.Linear(d_model, d_ff),
                nn.GELU(),
                nn.Linear(d_ff, d_model)
            )

    def forward(self, x, mask=None):
        x = x + self.dropout(self.attn(self.ln1(x), mask=mask))     # Attention + 残差
        if self.use_moe:
            x = x + self.dropout(self.moe(self.ln2(x), mask=mask))  # MoE + 残差
        else:
            x = x + self.dropout(self.ffn(self.ln2(x)))              # FFN + 残差
        return x
```

**Pre-Norm 残差结构的数据流**：

```
  x ──→ LayerNorm → Attention/MoE/FFN → Dropout ──→ (+) ──→ 输出
  │                                                  ↑
  └──────────── 残差连接（跳过子层）──────────────────┘

  公式: output = x + Dropout(SubLayer(LayerNorm(x)))
```

**为什么用 Pre-Norm 而不是 Post-Norm？**

```
Post-Norm（原始 Transformer）:
  output = LayerNorm(x + SubLayer(x))
  问题: 残差路径上没有 LayerNorm，深层梯度可能爆炸
       需要精心的学习率 warmup，训练不稳定

Pre-Norm（GPT-2 以来的主流）:
  output = x + SubLayer(LayerNorm(x))
  优势: 残差路径直通，梯度可以无损地从最后一层传到第一层
       LayerNorm 在子层输入端稳定数值
       训练更稳定，对学习率不那么敏感
```

MoE 更需要 Pre-Norm：因为 MoE 的路由是动态的（不同 token 走不同专家），训练初期路由不稳定，如果用 Post-Norm，梯度波动会被 LayerNorm 放大。Pre-Norm 让梯度通过残差路径直接回传，不经过 LayerNorm，更稳定。

**被丢弃 token 的信息保存机制**：

```
假设 Token 9 在 MoE 层被丢弃:

TransformerBlock.forward 执行:
  x = x + dropout(moe(ln2(x)))       ← moe 输出中 Token 9 = 全零向量
  x = x + dropout(0)                 ← Token 9: 原始值 + 0 = 原始值
  x = x                              ← 信息完整保留！

这就是"仅通过残差传递到下一层"的代码实现：
  被丢弃 → MoE 输出 = 0 → 残差连接加上 0 → 原始表征原封不动传到下一层
  下一层的 Attention 仍然能用这个 token 做 Q/K/V
  只是该 token 的表征没有经过 MoE 的非线性变换，语义丰富度不如正常 token
```

---

### 六、完整模型组装：MiniMoELLModel

```python
# 来自 [Mini LLM+MoE.py](<../Mini LLM+MoE.py>)
class MiniMoELLModel(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=4, n_layers=4, d_ff=1024,
                 use_moe_layer_index=None, moe_params=None):
        super().__init__()
        # === 嵌入层 ===
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(4096, d_model)      # 可学习位置编码

        # === Transformer 堆叠 ===
        self.layers = nn.ModuleList()
        use_moe_layer_index = set(use_moe_layer_index or [])

        if moe_params is not None:
            moe_params = moe_params.copy()
            moe_params.setdefault("d_model", d_model)   # 自动注入
            moe_params.setdefault("d_ff", d_ff)

        for i in range(n_layers):
            use_moe = (i in use_moe_layer_index)
            self.layers.append(TransformerBlock(
                d_model=d_model, nhead=nhead, d_ff=d_ff,
                use_moe=use_moe, moe_params=moe_params
            ))

        # === 输出层 ===
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight        # 权重共享！
```

**设计决策解析**：

**为什么 `lm_head.weight = self.tok_emb.weight`（权重共享）？**

```
tok_emb.weight:  [259, 256]  ← 词表中每个 token 的嵌入向量
lm_head.weight:  [259, 256]  ← 预测时从隐藏状态映射回词表

共享权重的数学直觉:
  嵌入: token_id → 向量 (查表)
  输出: 向量 → token_id (点积相似度)
  → 语义相近的 token 应该有相近的嵌入向量
  → 点积相似度在嵌入空间中有意义的前提就是共享权重
  → 不共享 = 两套独立的映射空间，语义不一致

参数节省: 259 × 256 = 66,304 参数（对小模型不算多，对大模型如 LLaMA-7B 节省 ~32M）
```

**`moe_params.setdefault` 的作用**：

```python
moe_params = moe_params.copy()           # 防止修改外部字典
moe_params.setdefault("d_model", d_model)  # 如果用户没传 d_model，自动用模型的
moe_params.setdefault("d_ff", d_ff)         # 同理
```

这保证 MoELayer 的 `d_model` 和 `d_ff` 与 TransformerBlock 一致，避免维度不匹配。

**完整前向传播**：

```python
    def forward(self, idx, mask=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device).unsqueeze(0)
        x = self.tok_emb(idx) + self.pos_emb(pos)   # [B, T, D] = 嵌入 + 位置
        for blk in self.layers:                       # 逐层处理
            x = blk(x, mask=mask)
        x = self.ln_f(x)                             # 最终 LayerNorm
        logits = self.lm_head(x)                      # [B, T, vocab_size]
        return logits
```

```
端到端数据流（"Hello MoE!" 示例）:

"Hello MoE!"
    │
    ↓ ByteTokenizer.encode()
[256, 72, 101, 108, 108, 111, 32, 77, 111, 69, 33, 257]  12 个 token
    │
    ↓ tok_emb + pos_emb
[1, 12, 256]  ← batch=1, seq=12, dim=256
    │
    ↓ Layer 0 (FFN):  x = x + dropout(attn(LN(x))) + dropout(ffn(LN(x)))
    ↓ Layer 1 (MoE):  x = x + dropout(attn(LN(x))) + dropout(moe(LN(x)))
    ↓ Layer 2 (FFN):  ...
    ↓ Layer 3 (MoE):  ...
    │
    ↓ ln_f (LayerNorm)
    ↓ lm_head (Linear 256→259, 共享权重)
[1, 12, 259]  ← 每个位置预测下一个 token 的概率分布
```

---

### 七、LLM 性能评估

测试代码中的评估逻辑：

```python
# 来自 [Mini LLM+MoE.py](<../Mini LLM+MoE.py>)
# 交叉熵损失
ce_loss = -torch.log(probs[0, torch.arange(length-1), target_ids] + 1e-9)
# PPL（困惑度）
ppl_text = math.exp(ce_loss.sum().item() / max(valid_len, 1))
```

**PPL 的含义**：

```
PPL（Perplexity，困惑度）= exp(交叉熵损失)

直觉理解: PPL = 模型在每个位置平均"纠结"于多少个候选 token
  PPL = 1   → 完美预测（只考虑 1 个 token，几乎不可能）
  PPL = 10  → 模型在每个位置平均纠结 10 个候选（不错）
  PPL = 259 → 完全随机（vocab_size=259，所有 token 概率相等）

交叉熵 = -log P(正确token)
PPL = exp(交叉熵) = exp(-log P) = 1/P
→ PPL 越低，模型对正确 token 的预测概率越高
```

**Top-1 和 Top-5 准确率**：

```python
# Top-1: 模型预测概率最高的 token 是否等于真实 token
top1_acc += ((topk_idx[:, 0] == target_ids) * mask).sum().item()

# Top-5: 真实 token 是否在模型预测概率最高的 5 个中
top5_acc += sum([(target_ids[i].item() in topk_idx[i].tolist()) * mask[i].item()
                 for i in range(length-1)])
```

```
类比考试:
  Top-1 = 选择题只选 1 个答案，选对才算对
  Top-5 = 选择题可以选 5 个答案，其中包含正确答案就算对

  未训练模型: Top-1 ≈ 1/259 ≈ 0.39%, Top-5 ≈ 5/259 ≈ 1.93%
  训练后模型: Top-1 和 Top-5 都会显著提升
  PPL 从 259 逐渐下降（越接近 1 越好）
```

---

## 🧠 本模块问题

请在下方回答以下问题后，输入 `提交作业` 提交。

**Q1**：在 MoE Layer 的 `forward` 方法中，被丢弃的 token（超出专家容量的 token）的输出值是什么？它如何通过 TransformerBlock 的残差连接保证信息不丢失？请从 `MoELayer.forward` 追踪到 `TransformerBlock.forward` 的完整数据流。

**Q2**：代码中 `combine_weights` 是通过 `torch.gather(scores, 1, top1.unsqueeze(1)).squeeze(1)` 获取的。为什么不用更简洁的 `scores.max(dim=-1).values`？两者在 Top-1 下结果相同，但 gather 的设计有什么考量？

**Q3**：Mini LLM 的专家网络使用 `nn.Sequential(Linear → GELU → Linear)`。如果把 GELU 换成 ReLU，对 MoE 的训练过程会产生什么影响？（提示：从"死神经元"和"专家有效容量"两个角度思考）

---

<!-- 学习者作答区（请在此处填写你的答案） -->

**A1**：

被 MOE 部分丢弃的 token 输出值是 0。因为 MOE block 有 residual connection，可以保证输入的 token 即使在 moe 部分丢失，也能在下一层保持，只不过没有经过 MOE 部分，特征不够丰富。

x -> layer_norm -> MOE_layer -> x_out + x
  |________________________________|

**A2**：
scores.max 虽然更简洁，但是只适用于 topk = 1 的情况，而 torch.gather 可以适用于多种情况，即使 topk 发生变化（ > 1 ）也能不需要修改适用，更符合多种情况，方便修改 topk 超参。


**A3**：

ReLU 的 公式是 max(x, 0)，当 x < 0 时梯度会变成 0 ，这会造成梯度在该部分无法回传，因此部分神经元坏死无法更新。
会导致 FFN(MOE) 中能存储的“知识” 数量下降，而 GELU 在 0 附近有梯度，不会导致神经元坏死，无法激活的情况。

---

<!-- 教师批改区（提交作业后由导师填写，请勿手动修改） -->

### 📝 批改结果

**Q1 批改**：

✅ 被丢弃 token 的 MoE 输出为 0 — 正确
✅ 残差连接保证信息不丢失 — 正确，核心机制抓住了
✅ 画了数据流示意图，方向正确 — 好习惯
⚠️ 图示不够精确：你的 `x_out` 没有体现"为什么是 0"。完整链路是：
   - `out_flat = torch.zeros_like(x_flat)` → 初始化全零
   - 被丢弃 token 的位置**从未被写入** → 保持为 0
   - `x + dropout(0)` = `x + 0` = `x`
   缺少"out_flat 初始化为零 + 未写入位置保持零"这个关键步骤
⚠️ 图示中的 `x -> layer_norm -> MOE_layer -> x_out + x` 省略了 dropout，正确应为 `x = x + dropout(moe(LN(x)))`。dropout 对零向量无影响（0 仍为 0），但形式上应该包含

— 得分：**7/10**

**Q2 批改**：

✅ 指出 gather 更通用 — 正确
✅ 提到 Top-K > 1 时 gather 仍适用 — 正确
⚠️ 遗漏了一个关键点：在 Top-1 下 `scores.max` 和 `gather` **结果完全相同**，区别不是"max 只能 Top-1"，而是：
   - `gather` 可以取**任意位置**的值（第 1 大、第 2 大、指定索引），代码改 Top-K 时只需改 topk 索引，`gather` 行不用动
   - `max` 只能取最大值，如果改为取第 2 大需要换 API（`topk` 或 `sort`）
   → 代码可维护性更好，不是"能不能用"的问题，是"改起来方不方便"的问题

— 得分：**6/10**

**Q3 批改**：

✅ ReLU 在 x<0 时梯度为 0，导致神经元坏死 — 正确
✅ GELU 在 0 附近有梯度，不会坏死 — 正确
✅ 指出影响专家的"知识存储能力" — 方向正确
⚠️ 可以补充更具体的机制：ReLU 死神经元在 MoE 中比 Dense 模型更严重——因为每个专家只处理一部分 token，如果某个专家有 30% 的神经元坏死，该专家的**有效参数量**就从 8d² 降到 5.6d²，而 Dense 模型所有 token 共享 FFN，坏死神经元的影响被大量正常 token 稀释。MoE 的"专家孤立性"放大了死神经元的影响

— 得分：**7/10**

**综合评价**：20/30。三个问题都抓住了核心要点，对 MoE 的残差保护、代码设计、激活函数选择有清晰理解。主要改进方向：
1. **数据流追踪精度**：描述"输出为 0"时要追溯到 `zeros_like` 初始化 + 未写入位置保持零，不要只说"丢失了"
2. **对比论述的完整性**：说 A 比 B 好时，先确认两者在当前条件下等价，再分析可维护性差异
3. **MoE 特有视角**：讨论 MoE 中的问题时，想想"Dense 模型下这个问题也存在吗？MoE 放大了还是缩小了？"

**批改时间**：2026-04-27
