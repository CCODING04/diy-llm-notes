# 第 7 章：GPU 高性能编程 — 模块 2：Kernel Fusion 与手写 CUDA 内核

> 📍 学习进度：第 7 章，第 2 / 3 模块
> 📅 生成时间：2026-05-08

---

## 学习目标

- 理解 Kernel Fusion 的核心原理（为什么手动实现比 PyTorch 慢 7.4 倍）
- 掌握 CUDA 内核的基本结构（`__global__`、`blockIdx`、`blockDim`、`threadIdx`）
- 理解 CUDA 内核启动流程（wrapper 函数、grid/block 维度、越界检查）
- 了解 Python-CUDA 绑定方式（`load_inline`）

---

## 核心内容

### 一、Kernel Fusion 的核心问题

#### 1.1 问题：为什么手动 GELU 比 PyTorch 慢 7.4 倍？

```
手动实现 GELU：
  y = 0.5 * x * (1 + torch.tanh(0.79788456 * (x + 0.044715 * x * x * x)))

这行代码包含的操作：
  x * x * x      → 立方运算
  0.044715 * ...  → 乘法
  x + ...         → 加法
  0.79788456 * ... → 乘法
  tanh(...)       → tanh 运算
  1 + ...         → 加法
  0.5 * x * ...   → 乘法

每个操作 → 一个独立的 CUDA 内核
总计：3 次 CUDA 内核启动（profiling 结果）
```

```
时间对比（dim=16384）：

  手动实现：  8.1 ms  （3 个 CUDA 内核）
  PyTorch：   1.1 ms  （1 个融合内核）
  手写 CUDA： 1.8 ms  （1 个自定义内核）
  Triton：    1.85 ms （1 个 Triton 内核）
  torch.compile：1.47 ms（自动融合）
```

<img src="https://raw.githubusercontent.com/datawhalechina/diy-llm/main/docs/chapter7/images/7-14-两个gelu的时间.png" width="800" alt="手动 vs PyTorch GELU 时间对比">

<img src="https://raw.githubusercontent.com/datawhalechina/diy-llm/main/docs/chapter7/images/7-15-手动gelu的性能分析.png" width="800" alt="手动 GELU 性能分析：3 个 CUDA 内核">

<img src="https://raw.githubusercontent.com/datawhalechina/diy-llm/main/docs/chapter7/images/7-16-pytorch的gelu的性能分析.png" width="800" alt="PyTorch GELU 性能分析：1 个融合内核">

#### 1.2 性能差距的根本原因

```
手动实现的执行过程：

  HBM → SM：读取 x
  SM 计算 x³ → 结果写回 HBM
  HBM → SM：读取 x³
  SM 计算乘法 → 结果写回 HBM
  HBM → SM：读取中间结果
  SM 计算 tanh → 结果写回 HBM
  ...（共 3 次 HBM ↔ SM 往返）

PyTorch 融合内核的执行过程：

  HBM → SM：读取 x（1 次）
  SM 内部：x³ → 乘法 → tanh → 最终结果（全部在寄存器/SRAM）
  SM → HBM：写回 y（1 次）

差距：3 次 HBM 往返 vs 1 次 HBM 往返
每次 HBM 往返 ≈ 2~3ms 的延迟
```

> 💡 **关键洞察**：手动实现慢的原因不是"Python 开销"，而是**每个子操作触发一次独立的 CUDA 内核 → 每次内核都要从 HBM 读取数据再写回**。这是 DRAM ↔ SM 之间的通信成本，不是 CPU ↔ GPU 的通信成本。

#### 1.3 Fusion 在不同算子上的表现

```
算子          手动实现    融合内核    加速比    原因
──────────────────────────────────────────────────
GELU          8.1 ms     1.1 ms     7.4×     逐元素操作，融合收益最大
Softmax       类似       类似       ~5×      涉及 exp/sum/div，融合后省多次 HBM 往返
matmul        N/A        N/A        N/A      已经是单一内核，无需融合
```

---

### 二、CUDA 内核的基本结构

#### 2.1 线程层级模型

```
Grid（网格）
  └── Block 0, Block 1, Block 2, ...（线程块集合）
        └── Thread 0, Thread 1, ..., Thread 1023（每个块内的线程）

术语对应：
  Grid  = 一个任务
  Block = 任务的一个分块（分配给一个 SM）
  Thread = 处理一个数据元素的最小单元

定位一个线程需要三个参数：
  blockIdx.x   → 该线程在哪个块
  blockDim.x   → 每个块有多少线程
  threadIdx.x  → 该线程在块内的编号

全局坐标：i = blockIdx.x * blockDim.x + threadIdx.x
```

#### 2.2 GELU 的 CUDA 内核代码

```cpp
// 第一部分：内核函数（运行在 GPU 上）
__global__ void gelu_kernel(float* in, float* out, int num_elements) {
    // 计算全局坐标
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // 越界检查（末尾的线程可能超出数组范围）
    if (i < num_elements) {
        // 每个线程处理一个元素
        out[i] = 0.5 * in[i] * (1.0 + tanh(
            0.79788456 * (in[i] + 0.044715 * in[i] * in[i] * in[i])
        ));
    }
}
```

```
__global__ 关键字：
  标识这是 CUDA 内核函数，由 CPU 调用，在 GPU 上执行

坐标计算（所有 CUDA 代码中最常见的模式）：
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  例：block_size=1024
    Block 0 的 Thread 0 → i = 0*1024 + 0 = 0
    Block 0 的 Thread 1 → i = 0*1024 + 1 = 1
    Block 1 的 Thread 0 → i = 1*1024 + 0 = 1024
    Block 1 的 Thread 511 → i = 1*1024 + 511 = 1535

越界检查：
  总元素 3000，block_size=1024 → 需要 3 个 Block（3072 个线程）
  Block 2 的 Thread 972~1023 → 对应 i=2972~3023 → 超出范围
  → if (i < num_elements) 跳过这些线程
```

#### 2.3 Wrapper 函数（运行在 CPU 上）

```cpp
inline unsigned int cdiv(unsigned int a, unsigned int b) {
    return (a + b - 1) / b;  // 向上取整
}

// 第二部分：wrapper 函数（运行在 CPU 上，协调内核启动）
torch::Tensor gelu(torch::Tensor x) {
    // 1. 检查输入
    TORCH_CHECK(x.device().is_cuda());   // 确保 x 在 GPU 上
    TORCH_CHECK(x.is_contiguous());       // 确保 x 是连续内存

    // 2. 分配输出空间
    torch::Tensor y = torch::empty_like(x);  // empty_like 比 zeros_like 快（后续会覆盖）

    // 3. 计算参数
    int num_elements = x.numel();          // 总元素数
    int block_size = 1024;                  // 每个块 1024 个线程
    int num_blocks = cdiv(num_elements, block_size);  // 需要多少个块

    // 4. 启动内核
    gelu_kernel<<<num_blocks, block_size>>>(
        x.data_ptr<float>(), y.data_ptr<float>(), num_elements
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return y;
}
```

```
关键步骤解析：

① is_contiguous()：矩阵在内存中必须连续存放
   连续：  A[0,0] A[0,1] A[0,2] A[0,3] A[1,0] A[1,1] ...  ✅
   不连续：转置后的矩阵按列存储 → 元素间有跳跃 → 无法简单用 i 索引

② empty_like vs zeros_like：
   empty_like：分配内存但不初始化（后续会覆盖）→ 更快
   zeros_like：分配 + 清零 → 多一次内核启动

③ <<<num_blocks, block_size>>>：尖括号语法指定内核启动参数
   这告诉 GPU："启动 num_blocks 个块，每个块 1024 个线程"

④ cdiv（向上取整）：确保所有元素都被处理
   3000 个元素 / 1024 块大小 = 2.93 → 取整为 3 个块
```

---

### 三、Python 绑定与 Benchmark

#### 3.1 使用 load_inline 编译 CUDA 代码

```python
from torch.utils.cpp_extension import load_inline

def create_cuda_gelu():
    cuda_gelu_src = open("gelu.cu").read()
    cpp_gelu_src = "torch::Tensor gelu(torch::Tensor x);"

    module = load_inline(
        cuda_sources=[cuda_gelu_src],
        cpp_sources=[cpp_gelu_src],
        functions=["gelu"],
        extra_cflags=["-O2"],
        name="inline_gelu",
    )
    cuda_gelu = getattr(module, "gelu")
    return cuda_gelu
```

```
调试技巧：
  os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
  → 强制同步执行，牺牲性能换取准确的错误信息
  → 写 CUDA 代码时必备
```

#### 3.2 性能分析结果

```
CUDA GELU 的 profiling：

  调用：gelu_kernel（1 个内核）
  Self CUDA 时间：100%（单个内核占用了全部 GPU 时间）

  对比手动实现（3 个内核）：
    vectorized_elementwise_kernel4（立方运算）
    vectorized_elementwise_kernel4（乘法 + tanh）
    vectorized_elementwise_kernel4（最终乘法）
  → 每个内核都有启动开销 + HBM 往返

结论：
  融合 = 所有操作在 SM 内部完成，只在开始和结束时各访问一次 HBM
```

---

## 🧠 本模块问题

请在下方回答以下问题后，输入 `提交作业` 提交。

**Q1**：手动实现 GELU（3 个 CUDA 内核）和 PyTorch 版本（1 个融合内核）的性能差距是 7.4 倍。请解释：为什么 3 个内核比 1 个内核慢这么多？瓶颈在哪里？

**Q2**：在 CUDA 内核代码中，`int i = blockIdx.x * blockDim.x + threadIdx.x` 这行代码的作用是什么？如果总元素数是 3000，block_size=1024，请写出每个 block 处理的元素范围，并说明越界检查的必要性。

**Q3**：在 wrapper 函数中，为什么用 `empty_like` 而不是 `zeros_like` 来分配输出张量？这个选择对性能有什么影响？

---

<!-- 学习者作答区（请在此处填写你的答案） -->

**A1**：

每个 内核 都有 HBM 搬运数据到 SM 共享内存的时间，以及数据从 SM 写到 HBM 的时间开销，每次开销延迟 2~5 ms，所以多个 kernel 操作有多个延迟叠加，因此 3 个内核操作相较于 1 个内核操作多了 4 个搬运的开销。


**A2**：

`int i = blockIdx.x * blockDim.x + threadIdx.x` 这行代码根据 块信息、块内线程数量以及线程 id 能定位到 线程的全局位置，方便快速索引数据。
如果总元素数据是 3000， block_size = 1024, 那么需要 3 个 block 来处理所有元素。
block 1 范围：0 ~ 1023
block 2 范围：1024 ~ 2047
block 3 范围： 2048 ~ 3000
如果不做越界检查，i 可以取值到 3071, 会越界访问额外的数据。



**A3**：

选择 `empty_like` 而不是 `zero_like` 是因为 `empty_like` 时间开销较少，只分配了数据空间，而不对内部数据做处理（因为后续会对该部分数据赋值）。
而 `zero_like` 分配了数据空间，并且还需要将该部分数据置 0，多了一步赋值操作，相当于多做了一次 kernel 操作。



---

<!-- 教师批改区（提交作业后由导师填写，请勿手动修改） -->

### 📝 批改结果

**Q1 批改**：核心思路正确——多个内核 = 多次 HBM 往返 = 延迟叠加。但"每次开销 2~5ms"不准确，A100 HBM 带宽 2TB/s，单次往返延迟约在微秒级，8.1ms 中的差距主要来自 3 次内核启动开销 + 3 次 HBM 往返的总和，而非每次 2~5ms。另外应该是多了 2 次额外往返（3次 vs 1次），不是 4 次。 — 得分：**7/10**

<details>
<summary>📖 Q1 参考答案</summary>

**3 个内核 vs 1 个内核的性能差距**：

```
手动实现（3 个内核）的执行流程：

  内核 1（立方运算）：
    HBM → SM：读取 x
    SM：计算 x³
    SM → HBM：写回 x³

  内核 2（乘法 + tanh）：
    HBM → SM：读取 x 和 x³
    SM：计算 0.044715*x³ + x → 0.79788456*... → tanh
    SM → HBM：写回中间结果

  内核 3（最终乘法）：
    HBM → SM：读取 x 和中间结果
    SM：计算 0.5 * x * (1 + tanh)
    SM → HBM：写回 y

  总计：3 次内核启动 + 3 次完整的 HBM ↔ SM 往返
```

```
PyTorch 融合内核（1 个内核）的执行流程：

  融合内核：
    HBM → SM：读取 x（1 次）
    SM 内部：x³ → 乘法 → 加法 → tanh → 最终乘法
             （全部在寄存器/SRAM 中完成，不经过 HBM）
    SM → HBM：写回 y（1 次）

  总计：1 次内核启动 + 1 次 HBM ↔ SM 往返
```

**瓶颈分析**：
- 每次 HBM ↔ SM 往返有固定的延迟开销（内存访问延迟）
- 每次内核启动有 CPU 调度开销（cudaLaunchKernel）
- 3 次 vs 1 次 = 多了 2 次额外 HBM 往返 + 2 次额外内核启动开销
- 瓶颈是 **DRAM ↔ SM 的通信成本**，不是计算量（两种方式计算量完全相同）

</details>

---

**Q2 批改**：完全正确。全局坐标公式、Block 范围划分、越界检查原因都答对了。 — 得分：**9/10**

<details>
<summary>📖 Q2 参考答案</summary>

**全局坐标公式的作用**：

`int i = blockIdx.x * blockDim.x + threadIdx.x` 计算当前线程的全局索引，让每个线程知道自己应该处理数组中的哪个元素。

**3000 个元素、block_size=1024 的情况**：

```
需要的 Block 数 = ⌈3000/1024⌉ = 3

Block 0：i = 0*1024 + threadIdx.x，threadIdx.x ∈ [0, 1023]
  → 处理元素范围：[0, 1023]（1024 个元素）

Block 1：i = 1*1024 + threadIdx.x，threadIdx.x ∈ [0, 1023]
  → 处理元素范围：[1024, 2047]（1024 个元素）

Block 2：i = 2*1024 + threadIdx.x，threadIdx.x ∈ [0, 1023]
  → 处理元素范围：[2048, 3071]，但数组只到 2999
  → threadIdx.x ∈ [0, 951] 对应 i ∈ [2048, 2999] ✅
  → threadIdx.x ∈ [952, 1023] 对应 i ∈ [3000, 3071] ❌ 越界！
```

**越界检查的必要性**：
- Block 数量是向上取整得到的，最后一个 Block 会有多余的线程
- 不做越界检查：i=3000~3071 的线程会访问数组越界内存 → 未定义行为（可能崩溃或数据损坏）
- `if (i < num_elements)` 让越界线程直接跳过，不执行任何操作

</details>

---

**Q3 批改**：完全正确。empty_like 只分配不初始化，zeros_like 分配 + 清零多一次内核启动，对后续会被覆盖的输出张量来说是无意义的额外开销。 — 得分：**9/10**

<details>
<summary>📖 Q3 参考答案</summary>

**empty_like vs zeros_like 的区别**：

```
empty_like(x)：
  → 调用 GPU 内存分配器，分配一块与 x 同大小的内存
  → 不初始化，内存内容是未定义的（垃圾值）
  → 操作：1 次（仅分配）

zeros_like(x)：
  → 分配内存
  → 启动一个 CUDA 内核将所有字节清零
  → 操作：2 次（分配 + 清零内核）
```

**性能影响**：
- zeros_like 的清零操作会触发一次额外的 CUDA 内核启动
- 这次内核启动有调度开销（CPU 侧）和计算开销（GPU 侧）
- 对于 GELU 这样的逐元素操作，输出 y 的每个位置都会被覆盖，清零毫无意义
- 省去这次内核启动 = 减少一次无意义的 GPU 操作

**类似优化思路**：
- 如果确定输出的每个位置都会被写入，用 `empty_like`
- 如果只写入部分位置（如 scatter 操作），需要 `zeros_like` 确保未写入位置为 0

</details>

---

**综合评价**：三道题核心概念都掌握了，Q2 和 Q3 几乎满分。Q1 的原理理解正确，但数值描述不够精确。整体可以继续。

**批改时间**：2026-05-08
