# 第 7 章：GPU 高性能编程 — 模块 1：Benchmark 与 Profiling

> 📍 学习进度：第 7 章，第 1 / 3 模块
> 📅 生成时间：2026-05-08

---

## 学习目标

- 理解 Benchmark 和 Profiling 的区别与互补关系
- 掌握 Benchmark 的两个关键点：预热（warmup）和 CUDA 同步
- 学会使用 torch.profiler 定位算子级别的性能瓶颈
- 理解不同算子的 profiling 特征（add、matmul、GELU、softmax、cdist）
- 了解 Nsight Systems 的用途

---

## 核心内容

### 一、高层原则：先分析，再优化

```
最重要的原则：
  要编写高性能代码，就必须持续进行性能分析。

常见错误：
  主观认定某个部分是瓶颈 → 花 3 小时优化 → 发现根本不是瓶颈 → 浪费时间

正确流程：
  基准测试（benchmark）→ 性能分析（profile）→ 定位瓶颈 → 针对性优化
```

> 💡 **核心思想**：关于 GPU 执行细节或如何编写 softmax 内核的方法可能不断演变，甚至可以依赖编译器的自动优化。但**性能分析的重要性永远不会随工具改变而改变**。

---

### 二、Benchmark（基准测试）：测量端到端耗时

#### 2.1 Benchmark 的两个关键点

**关键点 1：预热（Warmup）**

```
首次运行 PyTorch 代码比后续迭代慢很多，因为：
  - JIT 编译 CUDA 代码
  - 向 GPU 发送指令的初始化开销
  - 缓存预热

预热的目的：确保测量的是"稳态"性能，而非启动速度
```

**关键点 2：CUDA 同步**

```
GPU 和 CPU 是两个独立的计算单元，可以并行运行：

  CPU: 发送内核 → 继续执行下一行代码（不等 GPU）
  GPU: 异步执行计算

问题：如果 GPU 异步执行而 CPU 在计时，计时会提前结束！

  CPU: | 启动计时 → 发送内核 → 停止计时 | → 结果：1ms（错误！）
  GPU:            | ====== 执行计算 ======= |   实际：5ms

解决方案：torch.cuda.synchronize()

  CPU: | 启动计时 → 发送内核 → synchronize() → 停止计时 | → 结果：5ms（正确）
  GPU:            | ====== 执行计算 =======                |
```

#### 2.2 Benchmark 代码

```python
def benchmark(description: str, run: Callable, num_warmups: int = 1, num_trials: int = 3):
    # 1. 预热
    for _ in range(num_warmups):
        run()
    if torch.cuda.is_available():
        torch.cuda.synchronize()  # 等待 CUDA 线程完成

    # 2. 多次测量取平均
    times: list[float] = []
    for trial in range(num_trials):
        start_time = time.time()
        run()
        if torch.cuda.is_available():
            torch.cuda.synchronize()  # 等待 CUDA 线程完成
        end_time = time.time()
        times.append((end_time - start_time) * 1000)  # 转为毫秒

    mean_time = mean(times)
    return mean_time
```

```
记住两个关键点：
  ① 进行预热操作
  ② 调用 CUDA 同步
  如果忘记执行，可能会得到极其异常的数据
  （比如大型矩阵乘法瞬间完成，这显然不符合事实）
```

#### 2.3 Benchmark 的局限

```
Benchmark 能告诉你：
  ✅ 代码运行了多久（端到端时间）
  ✅ 不同实现哪个更快
  ✅ 性能如何随参数缩放（线性？超线性？）

Benchmark 不能告诉你：
  ❌ 时间具体消耗在哪个函数
  ❌ 底层调用了哪些 CUDA 内核
  ❌ CPU 和 GPU 时间的分配比例

→ 需要 Profiling 来补充
```

---

### 三、Profiling（性能分析）：定位时间消耗的位置

#### 3.1 Profiler 代码

```python
def profile(description: str, run: Callable, num_warmups: int = 1, with_stack: bool = False):
    # 预热
    for _ in range(num_warmups):
        run()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # 使用 torch.profiler
    with torch.profiler.profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            with_stack=with_stack) as prof:
        run()
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    # 输出表格（按 CUDA 总时间排序）
    table = prof.key_averages().table(sort_by="cuda_time_total",
                                      max_name_column_width=80,
                                      row_limit=10)
    return table
```

#### 3.2 各算子的 Profiling 特征

**矩阵加法（aten::add）**：

```
调用链：
  Python: a + b
    → aten::add（C++ 接口层）
      → vectorized_elementwise_kernel4（CUDA 内核，实际执行加法）
    + cuLaunchKernel（内核启动开销）
    + cudaDeviceSynchronize（等待 GPU 完成）

结果：CPU 1.4ms，CUDA 17μs
→ GPU 极快，但内核启动开销相对显著
```

**矩阵乘法（aten::matmul）**：

```
小矩阵（128×128）：
  → 直接调用 xmma_gemm（特定瓦片尺寸的 GEMM 内核）
  → 没有经过 Cutlass

大矩阵（2048×2048）：
  → 调用 Cutlass 库（NVIDIA 高性能线性代数库）
  → 分派到 cutlass_80simtt_sgemm（包含分块尺寸等参数化配置）

关键洞察：
  矩阵尺寸不同 → 底层调用完全不同的内核
  同一个 "a @ b" 操作，底层可能是不同的矩阵乘法原语
```

**torch.cdist（复杂操作的分解）**：

```
调用链：
  aten::cdist
    → aten::euclidean_dist
      → aten::matmul（占 GPU 78% 时间）
      + aten::pow（占 5%）
      + aten::sum（占 3%）
      + 数组复制（占 6%）

→ 一个 Python 操作在底层分解为多个 CUDA 内核
→ 优化方向：应集中精力在 matmul（占比最大）
```

**GELU 和 Softmax（融合内核的观察）**：

```
手动实现 GELU = 0.5 * x * (1 + tanh(...))
  → 3 个 CUDA 内核启动（乘法、加法、tanh 各一个）
  → 耗时 8.1ms

PyTorch GELU = torch.nn.functional.gelu(x)
  → 1 个融合 CUDA 内核
  → 耗时 1.1ms（7.4× 加速）

Softmax 也有专门编写的融合内核
  → GPU 不是执行基础原语，而是一次性完成所有计算
  → 这就是第 6 章讲的"算子融合"
```

#### 3.3 Nsight Systems（专业级工具）

```
torch.profiler 的局限：
  - 只显示 PyTorch 层级的调用
  - 复杂操作时可视化不够直观（self CUDA 时间分配不清）

Nsight Systems 的能力：
  - CPU/GPU 时间线可视化（并排显示两者的活动）
  - NVTX 标注：在代码中添加注释标记，分析器识别对应代码块
  - 精确到每个 CUDA 内核的执行时间
  - 可以看到 CPU 和 GPU 之间的协作机制

使用方式：
  with nvtx.range("define_model"):
      model = MLP(dim, num_layers).to(get_device())

→ 分析器运行时能识别 "define_model" 代码块的耗时
```

---

### 四、Benchmark vs Profiling 互补关系

```
工作流：
  ① Benchmark → 发现"这段代码很慢"（端到端时间）
  ② Profile   → 发现"时间花在 matmul 上，占 78%"（细粒度定位）
  ③ 优化      → 针对 matmul 进行优化
  ④ Benchmark → 验证优化效果

类比：
  Benchmark = 体检发现"肝功能异常"
  Profile   = 进一步检查定位"转氨酶偏高，因为熬夜"
  优化      = 调整作息
  Benchmark = 复查确认指标恢复
```

---

### 五、补充知识（交叉验证）

> 📅 2026-05-08 | 通过网络搜索交叉验证后补充

```
业界推荐的 Profiling 逐层深入方法：

  第 1 层：PyTorch Profiler（torch.profiler）
    → 步骤级 trace，定位"哪个算子慢"
    → 适合日常开发

  第 2 层：NVTX 标注 + Nsight Systems
    → CPU/GPU 时间线可视化，定位"哪段代码慢"
    → 用 nvtx.range("label") 标注关键代码段

  第 3 层：Nsight Compute
    → 单个 CUDA 内核的寄存器、共享内存、occupancy 分析
    → 极致优化时使用

教学内容中介绍的 torch.profiler + Nsight Systems 覆盖了前两层
第三层（Nsight Compute）在需要极致内核优化时才会用到
```

---

## 🧠 本模块问题

请在下方回答以下问题后，输入 `提交作业` 提交。

**Q1**：Benchmark 中为什么要进行"预热"和"CUDA 同步"？如果忘记做其中任何一个，可能出现什么问题？

**Q2**：对 `torch.cdist(a, b)` 进行 profiling 后，发现底层分解为 aten::matmul（78%）、aten::pow（5%）、aten::sum（3%）等。如果要优化这个操作，应该优先优化哪个部分？请用"算术强度"的概念解释为什么。

**Q3**：手动实现 GELU（`0.5 * x * (1 + tanh(...))`）在 dim=16384 时耗时 8.1ms，而 PyTorch 版本 `torch.nn.functional.gelu` 只需 1.1ms。请用 profiling 结果解释 7.4 倍性能差距的原因。

---

<!-- 学习者作答区（请在此处填写你的答案） -->

**A1**：



**A2**：



**A3**：



---

<!-- 教师批改区（提交作业后由导师填写，请勿手动修改） -->
