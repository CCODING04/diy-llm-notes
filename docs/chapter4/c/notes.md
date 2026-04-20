# 第 4 章 学习笔记

> 记录学习过程中的临时提问与解答，供复习参考。

---

## 模块 1：标准 Transformer 架构回顾 — QA 记录
> 📅 2026-04-20

**Q**：现代大语言模型（如 GPT 系列）通常只使用解码器部分，训练不需要编码器吗，推理也应该需要对 input token 编码吧？

**A**：原始 Transformer 是为**机器翻译**设计的，源语言和目标语言是不同序列，所以需要编码器-解码器。现代 LLM 做**自回归语言建模** $P(x_t | x_1, ..., x_{t-1})$，输入和输出是同一个序列。解码器的**因果自注意力（Causal Self-Attention）** 本身就在"编码"输入信息——输入 token 经过多层 transformer block 的因果自注意力，每层构建越来越丰富的表征。**解码器的因果自注意力 = 编码 + 生成一体化**，不需要单独的编码器。编码器-解码器模型（如 T5、BART）仍在翻译等特定任务中使用，但通用对话/生成任务 decoder-only 成为主流。

---

**Q**：问题 Q1 中"高频和低频维度"指的是什么部分的高频和低频？指代的是什么？

**A**：指正弦/余弦函数的**振荡频率**，由维度索引 $i$ 控制。角频率 $\omega_i = 1/10000^{2i/d_{model}}$：$i$ 小（如 $i=0$）时频率为 1（高频），sin/cos 振荡快，能精细区分相邻位置（局部信息）；$i$ 大（如 $i=255$）时频率为 1/10000（低频），变化缓慢，捕捉长距离位置关系（全局信息）。这是一个**多尺度编码体系**——不同维度分工覆盖从局部到全局的位置信息。

---

## 模块 2：现代变体（归一化与激活函数）— QA 记录
> 📅 2026-04-20

**Q**：RMSNorm 全称是什么，是不是类似 BatchNorm 里面使用指数移动平均里面求 running_var 和 running_mean，但是这里为什么没有平移部分的设计，只有对原有数据的拉伸？

**A**：RMSNorm 全称 **Root Mean Square Layer Normalization**。它与 BatchNorm **完全不同**——BatchNorm 跨 batch 维度用 EMA 累积 running mean/var；RMSNorm 继承的是 LayerNorm 的方式，对**单个样本的特征维度**实时计算统计量。RMSNorm 从 LayerNorm 简化而来：去掉减均值、去掉偏置 β，只保留 γ 缩放。β 不需要是因为后续线性层本身就能隐式完成平移功能，且不去减均值对 ReLU/SwiGLU 等激活函数效果无影响，重要的是数值相对大小可控。

---

**Q**：简单介绍下 torch.nn.RMSNorm 模块

**A**：PyTorch 2.4+ 提供 `torch.nn.RMSNorm(normalized_shape, eps=1e-8, elementwise_affine=True)`。与 LayerNorm 对比：①无 bias 参数（只有 weight/γ）②不减均值 ③默认 eps=1e-8（非 LayerNorm 的 1e-5）。`elementwise_affine=False` 时连 γ 都不学，变成纯归一化。

---

**Q**：教程中关于 GeLU 以及其详细公式并没有介绍，请补充

**A**：GeLU 公式 $\text{GeLU}(x) = x \cdot \Phi(x)$，其中 $\Phi(x) = \frac{1}{2}[1 + \text{erf}(x/\sqrt{2})]$ 是标准正态分布 CDF。本质是用概率做"软开关"：$x$ 大时 $\Phi(x) \approx 1$ 完全通过，$x$ 小时 $\Phi(x) \approx 0$ 几乎截断。相比 ReLU 的硬开关 $\mathbb{1}_{x>0}$，GeLU 在原点处平滑过渡（处处可微）。实践中用 tanh 近似避免昂贵的 erf 计算。PyTorch 中 `nn.GELU(approximate='none')` 为默认精确计算，`approximate='tanh'` 为快速近似。

---

**Q**：GeLU 具体如何简单实现（不使用 torch 已有的 GeLU 实现）？

**A**：精确版：`x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))`；tanh 近似版：`0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0/math.pi) * (x + 0.044715 * x**3)))`。核心就一行：$x \cdot \Phi(x)$。

---

**Q**：torch.erf 是什么函数，有什么作用（说是计算误差，计算什么误差？）？

**A**：erf（误差函数）的"误差"指**统计学中正态分布的测量误差**，不是 loss/mistake。定义为 $\text{erf}(x) = \frac{2}{\sqrt{\pi}} \int_0^x e^{-t^2} dt$，形状是从 -1 到 1 的 S 型曲线。它在 GeLU 中的作用是计算 $\Phi(x)$（标准正态分布 CDF），即输入 $x$ 有多大概率被保留。名字来自 19 世纪研究天文测量误差分布时发现需要积分 $e^{-t^2}$，故命名为"误差函数"。
