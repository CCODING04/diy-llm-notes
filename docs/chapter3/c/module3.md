# 第 3 章：PyTorch 与资源核算 — 模块 3：模型构建与训练基础

> 📍 学习进度：第 3 章，第 3 / 3 模块
> 📅 生成时间：2026-04-18

---

## 学习目标

- 掌握 Xavier 初始化的原理和截断正态分布的使用
- 理解 nn.Parameter、nn.Module、nn.ModuleList 的关系和用法
- 掌握数据加载的关键优化（memmap、pin_memory、non_blocking）
- 理解 SGD 和 AdaGrad 优化器的实现原理
- 掌握完整训练循环的各步骤及检查点机制
- 理解混合精度训练的策略和工具

---

## 核心内容

### 一、参数初始化（3.5.1）

#### 为什么初始化很重要

直接用标准正态分布 `torch.randn` 初始化，当维度大时数值会爆炸或消失：

```python
w = nn.Parameter(torch.randn(16384, 256))
x = torch.randn(16384)
output = x @ w  # 输出值约 18.9，太大！
```

#### Xavier 初始化

核心思想：除以 √输入维度，让输出的方差与输入维度无关。

```python
w = nn.Parameter(torch.randn(input_dim, output_dim) / np.sqrt(input_dim))
```

#### 截断正态分布

即使 Xavier 初始化，正态分布的尾部仍可能产生极端值。解决方案：用 `trunc_normal_` 限制范围。

```python
w = nn.Parameter(nn.init.trunc_normal_(
    torch.empty(input_dim, output_dim),
    std=1 / np.sqrt(input_dim),
    a=-3, b=3
))
```

> 截断正态是 LLM 训练中最常用的初始化方式，因为极端权重值会破坏训练稳定性。

### 二、自定义模型（3.5.2）

#### 基本构建块

```python
class Linear(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.weight = nn.Parameter(
            torch.randn(input_dim, output_dim) / np.sqrt(input_dim)
        )

    def forward(self, x):
        return x @ self.weight
```

关键理解：
- **nn.Parameter** 是 Tensor 的子类，告诉 PyTorch "这是可学习的参数"
- **nn.Module** 通过 `state_dict()` 自动收集所有 `nn.Parameter`
- **nn.ModuleList** 用于动态数量的层（普通 Python list 不会被 `state_dict()` 识别）

#### 组合模型

```python
class Cruncher(nn.Module):
    def __init__(self, dim: int, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([
            Linear(dim, dim) for _ in range(num_layers)
        ])
        self.final = Linear(dim, 1)

    def forward(self, x):
        B, D = x.size()
        for layer in self.layers:
            x = layer(x)
        x = self.final(x)
        return x.squeeze(-1)  # (B, 1) → (B,)
```

**查看参数**：

```python
model = Cruncher(dim=64, num_layers=2)
for name, param in model.state_dict().items():
    print(f"{name}: {param.numel()} params")
# layers.0.weight: 4096 (64×64)
# layers.1.weight: 4096 (64×64)
# final.weight: 64  (64×1)
```

### 三、随机性管理（3.5.3）

训练中随机性来源：参数初始化、Dropout、数据打乱等。设置随机种子确保可重现性：

```python
seed = 0
torch.manual_seed(seed)    # PyTorch
np.random.seed(seed)       # NumPy
random.seed(seed)           # Python 标准库
```

> 三个库的随机数生成器独立，需要分别设置。

### 四、数据加载（3.5.4）

#### memmap：处理超大数据集

对于 TB 级数据，不可能全部加载到内存。`numpy.memmap` 创建文件指针，按需读取：

```python
data = np.memmap("data.npy", dtype=np.int32)  # 不占内存，按需读取
```

#### get_batch 的关键优化

```python
def get_batch(data, batch_size, sequence_length, device):
    # 随机采样起始位置
    start_indices = torch.randint(len(data) - sequence_length, (batch_size,))
    x = torch.tensor([data[s:s+sequence_length] for s in start_indices])

    # 关键优化
    if torch.cuda.is_available():
        x = x.pin_memory()              # 1. 固定内存
    x = x.to(device, non_blocking=True) # 2. 异步传输
    return x
```

| 优化 | 原理 | 效果 |
|------|------|------|
| **pin_memory()** | 将 CPU 内存标记为"不可换出"，GPU 可直接访问 | 避免额外拷贝 |
| **non_blocking=True** | 数据传输在后台异步进行 | CPU 和 GPU 并行工作 |

两者结合可实现流水线：GPU 处理当前批次的同时，CPU 加载下一批次。

### 五、优化器（3.5.5）

#### SGD：最简单的优化器

```python
class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=0.01):
        super().__init__(params, dict(lr=lr))

    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                p.data -= lr * p.grad.data  # 直接减去梯度
```

#### AdaGrad：自适应学习率

核心思想：经常更新的参数 → 学习率自动减小；稀疏更新的参数 → 学习率保持较大。

```python
class AdaGrad(torch.optim.Optimizer):
    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                state = self.state[p]
                grad = p.grad.data

                g2 = state.get("g2", torch.zeros_like(grad))
                g2 += grad ** 2          # 累加梯度平方
                state["g2"] = g2

                p.data -= lr * grad / torch.sqrt(g2 + 1e-5)  # 自适应调整
```

**优化器状态 = 额外的内存开销**（每个参数额外的 `g2` 张量）。这就是为什么训练时每个参数占 16 bytes。

> 优化器发展路线：SGD → Momentum → AdaGrad → RMSProp → **Adam/AdamW**（当前主流）

#### zero_grad 的正确用法

```python
optimizer.zero_grad(set_to_none=True)  # 比 set_to_none=False 更高效
```

`set_to_none=True` 将梯度指针设为 None 而非填充零值，节省内存。

### 六、训练循环（3.5.7）

标准训练循环的五个步骤：

```python
for t in range(num_train_steps):
    x, y = get_batch(B=B)              # 1. 获取数据
    pred_y = model(x)                   # 2. 前向传播
    loss = F.mse_loss(pred_y, y)        # 3. 计算损失
    loss.backward()                     # 4. 反向传播（计算梯度）
    optimizer.step()                    # 5. 更新参数
    optimizer.zero_grad(set_to_none=True) # 清空梯度
```

> 注意顺序：step 在 zero_grad 之前。先更新参数，再清空梯度准备下一步。

### 七、检查点 Checkpointing（3.5.8）

训练可能持续数天甚至数月，必须定期保存进度以防中断。

```python
# 保存
checkpoint = {
    "model": model.state_dict(),       # 模型参数
    "optimizer": optimizer.state_dict(), # 优化器状态（动量、方差等）
}
torch.save(checkpoint, "model_checkpoint.pt")

# 加载
loaded = torch.load("model_checkpoint.pt")
model.load_state_dict(loaded["model"])
optimizer.load_state_dict(loaded["optimizer"])
```

**为什么必须保存优化器状态？** 如果只恢复模型参数，优化器的动量/方差会从零开始，导致训练不连续甚至无法收敛。

### 八、混合精度训练（3.5.9）

| 部分 | 精度 | 原因 |
|------|------|------|
| 前向传播（激活值） | BF16 | 节省显存，加速计算 |
| 参数（主副本） | FP32 | 数值稳定性 |
| 梯度 | FP32 | 精度敏感 |
| 优化器状态 | FP32 | 精度敏感 |

核心思路：**低精度用于"消耗大但精度要求不高"的部分（激活值），高精度用于"对精度敏感"的部分（参数和梯度）**。

工具：
- **PyTorch AMP**（`torch.amp`）：自动管理精度转换
- **NVIDIA Transformer Engine**：支持 FP8 精度，面向极致性能

> 🌐 **补充（Web Search）**：2025 年 PyTorch 推荐使用 `torch.amp.autocast` 和 `torch.amp.GradScaler`（仅 FP16 需要，BF16 不需要 scaler）来实现混合精度训练。BF16 因动态范围与 FP32 相同，训练更稳定，已成为 LLM 训练的默认选择。

> 🌐 **补充（Web Search）**：`torch.compile()` 可以进一步优化训练速度——它通过算子融合（operator fusion）减少内存读写次数和 kernel launch 开销，通常能提升 10%–30% 的训练速度。参考 [PyTorch Performance Tuning Guide](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html)。

---

## 🧠 本模块问题

请在下方回答以下问题后，输入 `提交作业` 提交。

**Q1**：假设你有一个 `Cruncher(dim=128, num_layers=3)` 模型（结构与本模块中的定义相同）。请计算该模型的总参数量，并估算在 FP32 精度下使用 AdaGrad 优化器训练时，仅参数+梯度+优化器状态占用多少内存（以 MB 为单位）。

**Q2**：训练循环中，为什么 `optimizer.step()` 必须在 `loss.backward()` 之后、`optimizer.zero_grad()` 之前调用？如果颠倒 `step()` 和 `zero_grad()` 的顺序会怎样？

**Q3**：混合精度训练中，为什么激活值可以用 BF16 而参数的主副本必须用 FP32？如果参数也用 BF16 存储会出现什么问题？

---

<!-- 学习者作答区（请在此处填写你的答案） -->

**A1**：

总参数量 4 * （128 * 128 * 2 + 128 * 1） = 4 * 32896 Bytes = 131584 Bytes
参数+梯度+优化器状态 = 6 * 131584 Bytes = 0.76 MB
优化器使用 AdaGrad 需要有个 state 保存 grad**2


**A2**：
因为梯度回传了，每个需要更新的参数才能获得自己的 梯度，然后 step 之后，才能更新参数值。
如果颠倒 step 和 zero_grad 则参数永远不会更新，因为梯度更新前永远为 0


**A3**：

参数的更新需要维持精度，FP32 是小数部分数值更高的，小数范围更广的选择，如果使用 BF16 则会出现精度丢失，训练过程中精度截断或者失效的情况。

---

<!-- 教师批改区（提交作业后由导师填写，请勿手动修改） -->

### 📝 批改结果

**Q1 批改**：思路正确，但有两个具体错误：

**错误 1：num_layers 算错了**
- 题目是 `Cruncher(dim=128, num_layers=3)`
- 你写的是 `128 * 128 * 2`（用了 2 层），应该是 **3 层**
- 正确参数量：3 × 128 × 128 + 128 × 1 = 49,152 + 128 = **49,280 参数**

**错误 2：乘数的含义混淆了**

对于 AdaGrad（注意不是 Adam！），每个参数的内存构成：

| 组件 | FP32 字节数 |
|------|-----------|
| 参数 | 4 bytes |
| 梯度 | 4 bytes |
| AdaGrad g2 | 4 bytes（仅 1 个状态变量） |
| **合计** | **12 bytes/参数 = 3 × param_bytes** |

你用了 **6×** 乘数（0.76 MB），这对应的是 Adam 的 16 bytes/param 情形，而非 AdaGrad。

正确计算：49,280 × 12 = 591,360 bytes ≈ **0.56 MB**

注意区分：AdaGrad 有 1 个状态变量(g2) → 12B/param；Adam 有 2 个状态变量(m,v) → 16B/param。 — 得分：**4/10**

**Q2 批改**：理解正确！

顺序：`backward()` → `step()` → `zero_grad()`

- backward 计算梯度 → step 用梯度更新参数 → zero_grad 清空梯度准备下一步
- 颠倒 step 和 zero_grad：zero_grad 先清空梯度 → step 看到零梯度 → 参数不更新

补充一个细节：如果 `set_to_none=True`，zero_grad 在 step 之前会导致 `p.grad` 为 None，step 中访问 `p.grad.data` 会直接报错（不只是不更新，而是崩溃）。 — 得分：**8/10**

**Q3 批改**：方向正确，但可以更精确。核心问题是**参数更新时的舍入误差**：

```
W_new = W_old - lr × grad
```

假设 lr = 1e-5，grad 均值 ≈ 1e-3，则更新量 Δ ≈ 1e-8。

- FP32（23 位尾数，精度 ~1e-7）：对 W ≈ 1.0 的参数，1e-8 的更新可以被表示 ✅
- BF16（7 位尾数，精度 ~1e-2）：对 W ≈ 1.0 的参数，1e-8 的更新**远小于最小可分辨差异，直接被舍掉** ❌

这就是为什么参数主副本必须用 FP32：保证微小更新不会被"吞掉"。而激活值是前向传播的中间结果，精度稍低不影响最终收敛。 — 得分：**7/10**

**综合评价**：Q1 需要更仔细审题（num_layers=3 不是 2）并区分 AdaGrad 和 Adam 的内存差异。Q2/Q3 理解正确，Q3 建议记住"微小更新被舍掉"这个具体机制。第 3 章三个模块全部完成，可以进入章节收尾。

**批改时间**：2026-04-18
