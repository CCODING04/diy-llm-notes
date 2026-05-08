# 第 7 章：GPU 高性能编程 — 模块 3：Triton 与 torch.compile

> 📍 学习进度：第 7 章，第 3 / 3 模块
> 📅 生成时间：2026-05-08

---

## 学习目标

- 理解 Triton 的定位：比 CUDA 更高级的 GPU 编程抽象
- 掌握 Triton 的核心概念：以块（Block）为中心的编程模型
- 理解 Triton 与 CUDA 的代码差异（向量化操作 vs 线程级操作）
- 了解 torch.compile 的自动优化能力及其适用边界

---

## 核心内容

### 一、Triton：用 Python 写 CUDA

#### 1.1 Triton 的定位

```
编程抽象层级（从低到高）：

  CUDA C/C++    → 线程级编程，手动管理内存合并、共享内存、线程同步
  Triton        → 块级编程，自动管理内存合并、共享内存
  torch.compile → 图级优化，自动融合算子

Triton 由 OpenAI 于 2021 年开发
核心思想：以"块"（Block）为单位编程，而非以"线程"为单位
```

#### 1.2 Triton 自动管理的细节

```
Triton 自动处理（不需要你关心）：
  ✅ 内存合并（burst mode，从 DRAM 一次获取相邻数据）
  ✅ 共享内存管理（SM 内部的高速缓存）
  ✅ SM 内线程的启动和停止
  ✅ 寄存器分配

Triton 不自动处理（需要你设计）：
  ❌ 跨 SM 的调度
  ❌ 不同 SM 之间的操作协调
  → 你以"块"为单位思考，编译器处理底层细节
```

#### 1.3 Triton vs CUDA 的关键差异

```
CUDA（线程视角）：
  每个线程处理 1 个元素
  int i = blockIdx.x * blockDim.x + threadIdx.x;  // 单个索引
  out[i] = f(in[i]);  // 标量操作

Triton（块视角）：
  每个块处理 BLOCK_SIZE 个元素
  offsets = block_start + tl.arange(0, BLOCK_SIZE)  // 向量索引
  x = tl.load(x_ptr + offsets, mask=mask)           // 向量加载
  y = f(x)                                           // 向量操作
  tl.store(y_ptr + offsets, y, mask=mask)            // 向量存储
```

---

### 二、用 Triton 实现 GELU

#### 2.1 Triton 内核代码

```python
@triton.jit
def triton_gelu_kernel(x_ptr, y_ptr, num_elements, BLOCK_SIZE: tl.constexpr):
    # 1. 计算当前块应该处理哪些元素
    pid = tl.program_id(axis=0)              # 块 ID（类似 blockIdx.x）
    block_start = pid * BLOCK_SIZE            # 块起始位置
    offsets = block_start + tl.arange(0, BLOCK_SIZE)  # 向量偏移量

    # 2. 越界掩码
    mask = offsets < num_elements

    # 3. 向量加载（一次加载整个块的数据）
    x = tl.load(x_ptr + offsets, mask=mask)

    # 4. 计算 GELU（向量操作）
    a = 0.79788456 * (x + 0.044715 * x * x * x)
    exp = tl.exp(2 * a)
    tanh = (exp - 1) / (exp + 1)  # Triton 没有 tl.tanh，手动实现
    y = 0.5 * x * (1 + tanh)

    # 5. 向量存储（一次写回整个块的结果）
    tl.store(y_ptr + offsets, y, mask=mask)
```

#### 2.2 代码逐行解析

```
关键概念对比：

  CUDA                        Triton
  ────────────────────────────────────────────
  blockIdx.x                  tl.program_id(axis=0)
  threadIdx.x                 tl.arange(0, BLOCK_SIZE)（向量！）
  float x = in[i]             x = tl.load(ptr + offsets, mask)（向量！）
  out[i] = result             tl.store(ptr + offsets, y, mask)（向量！）
  if (i < num_elements)       mask = offsets < num_elements

最大区别：
  CUDA：每个线程处理 1 个标量
  Triton：每个块处理 1 个向量（BLOCK_SIZE 个元素）

  → Triton 的代码更简洁（不需要手动管理线程索引）
  → Triton 编译器自动处理向量化、内存合并等底层细节
```

#### 2.3 Wrapper 函数

```python
def triton_gelu(x: torch.Tensor):
    assert x.is_cuda
    assert x.is_contiguous()

    y = torch.empty_like(x)
    num_elements = x.numel()
    block_size = 1024
    num_blocks = triton.cdiv(num_elements, block_size)

    # 启动内核：方括号语法指定 grid（块的数量）
    triton_gelu_kernel[(num_blocks,)](x, y, num_elements, BLOCK_SIZE=block_size)
    return y
```

```
对比 CUDA wrapper 的启动语法：

  CUDA:    gelu_kernel<<<num_blocks, block_size>>>(...)
  Triton:  triton_gelu_kernel[(num_blocks,)](..., BLOCK_SIZE=block_size)

两者本质相同：指定启动多少个块，每个块多少线程/元素
```

---

### 三、四种实现的性能对比

```
GELU 性能对比（dim=16384，A100）：

  实现方式          耗时       内核数    编程难度    性能瓶颈
  ─────────────────────────────────────────────────────────
  手动 PyTorch      8.1 ms     3 个      最低        多次 HBM 往返
  torch.compile     1.47 ms    1 个      最低        自动融合
  Triton            1.85 ms    1 个      中等        块级优化
  手写 CUDA         1.84 ms    1 个      最高        线程级控制
  PyTorch 内置      1.1 ms     1 个      N/A         高度优化的库函数

排序：PyTorch 内置 > torch.compile > CUDA ≈ Triton >> 手动实现
```

```
关键发现：

① 融合是核心：从 3 个内核降到 1 个内核，性能提升 4~7 倍
② CUDA 和 Triton 性能接近：1.84ms vs 1.85ms，说明 Triton 的抽象没有性能损失
③ torch.compile 竞争力强：1.47ms，接近手写 CUDA，且不需要写任何底层代码
④ PyTorch 内置最快：1.1ms，说明 PyTorch 团队的库函数经过了极致优化
```

---

### 四、torch.compile：自动优化

#### 4.1 基本用法

```python
# 原始手动实现（3 个内核，8.1ms）
def manual_gelu(x: torch.Tensor):
    return 0.5 * x * (1 + torch.tanh(0.79788456 * (x + 0.044715 * x * x * x)))

# 编译优化（自动融合为 1 个内核，1.47ms）
compiled_gelu = torch.compile(manual_gelu)
```

```
torch.compile 做了什么：
  ① 追踪计算图：识别 manual_gelu 中的所有操作
  ② 图优化：将连续的逐元素操作融合为一个内核
  ③ JIT 编译：生成优化的 CUDA 代码
  ④ 缓存：编译结果缓存，后续调用直接使用

一行代码，从 8.1ms 降到 1.47ms
```

#### 4.2 torch.compile 的适用边界

```
torch.compile 擅长的场景：
  ✅ 简单的算子融合（如 GELU、SwiGLU）
  ✅ 已知矩阵形状时自动选择最优 GEMM 内核
  ✅ 常规的逐元素和归约操作
  → 编译器能"免费"带来 10~30% 的加速

torch.compile 不擅长的场景：
  ❌ FlashAttention 级别的复杂优化（需要硬件底层知识）
  ❌ 利用特定硬件特性（如 H100 的异步执行、WGMMA）
  ❌ 需要算法层面创新的优化（如 Online Softmax）
  → 这些需要人工理解和设计
```

#### 4.3 何时手写 vs 依赖编译器

```
决策框架：

  ① 先用 torch.compile 试试 → 如果性能够用，不需要手写
  ② 如果 torch.compile 不够好 → 用 Triton 重写关键内核
  ③ 如果 Triton 也不够好 → 用 CUDA C++ 重写（极少见）

核心观点：
  不用为每个模块都手写 CUDA 内核，这很可能是在浪费时间
  但如果遇到复杂模块，GPU 利用率不理想且有优化空间
  → 值得使用 Triton
```

---

### 五、GELU 优化总结

```
完整工作流（本章的核心方法论）：

  ① Benchmark → 发现"GELU 很慢"（8.1ms）
  ② Profile   → 发现"3 个 CUDA 内核，多次 HBM 往返"
  ③ 分析      → 瓶颈在 DRAM ↔ SM 通信，不是计算量
  ④ 优化      → 融合为 1 个内核
  ⑤ 实现方式：
     - torch.compile（最简单，1.47ms）
     - Triton（Python 编写，1.85ms）
     - CUDA C++（手动控制，1.84ms）
  ⑥ Benchmark → 验证优化效果

这个工作流适用于任何 GPU 性能优化任务
```

---

### 六、补充知识（交叉验证）

> 📅 2026-05-08 | 通过网络搜索交叉验证后补充

#### 6.1 torch.compile 进阶用法

```
区域编译（Regional Compilation）：
  torch.compile.region(model.lm_head)
  → 只编译模型的某个子模块，而非整个模型
  → 编译延迟从 67.4s 降到 9.6s（7× 编译加速）
  → 适用场景：LoRA 热切换、推理服务中部分模块需要频繁重编译

动态形状（Dynamic Shapes）：
  compiled_fn = torch.compile(fn, dynamic=True)
  → 当 batch size 或序列长度变化时，避免触发重编译
  → 默认情况下，输入 shape 变化会触发重新编译

fullgraph 模式：
  compiled_fn = torch.compile(fn, fullgraph=True)
  → 如果计算图中存在"图断裂"（graph break），直接报错
  → 用于验证整个函数是否可以被完全编译
```

#### 6.2 Triton 生态发展

```
教学内容中未涉及但值得关注的趋势：

TritonBench（2025）：
  → 评估 LLM 自动生成 Triton 内核能力的基准测试
  → 反映 Triton 已成为 GPU 编程的事实标准之一

AMD ROCm 支持：
  → Triton 已扩展支持 AMD GPU（ROCm 平台）
  → 不再局限于 NVIDIA CUDA 生态

与 torch.compile 的关系：
  → torch.compile 的 Inductor 后端核心就是自动生成 Triton 代码
  → 你写的 Triton 内核 vs 编译器自动生成的 Triton 内核
  → 部分场景下编译器生成的内核可以与手写代码持平甚至更优
```

> 💡 **边界比教学描述更灵活**：教学中说 torch.compile "不擅长 FlashAttention 级别的复杂优化"，但随着 Inductor 后端的持续改进，这个边界在不断缩小。实际选择时仍建议先试 torch.compile，不够再考虑手写。

---

## 🧠 本模块问题

请在下方回答以下问题后，输入 `提交作业` 提交。

**Q1**：Triton 以"块"为单位编程，CUDA 以"线程"为单位编程。请用 GELU 的例子说明：在 Triton 中，`offsets = block_start + tl.arange(0, BLOCK_SIZE)` 生成的是一个向量而非标量，这与 CUDA 中 `int i = blockIdx.x * blockDim.x + threadIdx.x` 有什么本质区别？

**Q2**：四种 GELU 实现的性能对比：手动 8.1ms、torch.compile 1.47ms、CUDA 1.84ms、Triton 1.85ms。请解释：为什么 torch.compile 比手写 CUDA 还快？什么时候手写 CUDA/Triton 仍然有价值？

**Q3**：torch.compile 的一行代码就能将手动 GELU 从 8.1ms 降到 1.47ms。请简述 torch.compile 的工作原理，并回答：为什么我们还需要学习 Triton/CUDA？

---

<!-- 学习者作答区（请在此处填写你的答案） -->

**A1**：



**A2**：



**A3**：



---

<!-- 教师批改区（提交作业后由导师填写，请勿手动修改） -->
