# 第 1 章：工具使用（WandB）— 模块 1：核心工作流

> 📍 学习进度：第 1 章，第 1 / 2 模块
> 📅 生成时间：2026-04-13

---

## 学习目标

- 理解 WandB 在 LLM 实验中解决的核心问题
- 掌握 `wandb.init()` 的参数配置和最佳实践
- 能够用 `wandb.log()` 记录训练指标并在 UI 中查看

---

## 核心内容

### 什么是 WandB，为什么需要它？

在训练大模型时，你会面临这些痛点：
- 跑了十几个实验，忘记哪个超参数组合效果最好
- 换了机器，之前的 loss 曲线找不到了
- 和同学协作，实验结果无法共享

**Weights & Biases（W&B）** 就是为此而生的实验管理平台，核心能力：

| 功能 | 说明 |
|------|------|
| 实验追踪 | 自动记录超参数、指标、代码快照 |
| 可视化 | 实时绘制 loss/accuracy 曲线 |
| 对比分析 | 多次实验一键对比 |
| 模型版本管理 | Artifact 管理 checkpoint |
| 超参数搜索 | Sweeps 自动化调参 |

---

### 安装与登录

```bash
pip install wandb
wandb login   # 首次使用，输入 API key
```

API key 在 [wandb.ai/settings](https://wandb.ai/settings) 获取。无外网环境直接跳过登录，使用离线模式（见模块 2）。

---

### `wandb.init()` — 初始化实验

每次训练开始时调用一次，创建一个 **run**（实验记录单元）：

```python
import wandb

wandb.init(
    project="cs336-a5-sft-v2",          # 项目名，同项目的 run 会归组在一起
    entity="your-team-or-username",     # 团队/用户名（可选）
    name="wanda_sft",                   # 本次 run 的可读名称
    config={
        "model": "Qwen2.5-Math-1.5B",
        "dataset_tag": "raw",
        "batch_size": 64,
        "learning_rate": 2e-5,
        "seed": 2026,
    }
)
```

`config` 是这次实验的"身份证"，建议把**所有超参数、数据路径、模型版本**都放进去，方便后续在 UI 中筛选和对比。

> 💡 **补充资料（来源：Context7 / wandb 官方文档）**
>
> 更推荐使用 `with` 语句管理 run 的生命周期，确保即使训练中途报错也能正确结束 run、不产生"僵尸 run"：
>
> ```python
> with wandb.init(project="my-project", config={"lr": 3e-4}) as run:
>     # 训练代码
>     run.log({"loss": 0.1})
> # 退出 with 块时自动调用 run.finish()
> ```
>
> 也可以动态更新 config（例如在数据加载后才知道实际 token 数）：
> ```python
> run.config.update({"actual_tokens": dataset.total_tokens})
> ```

---

### `wandb.log()` — 记录训练指标

在训练循环中，每步/每 epoch 调用一次：

```python
for step, batch in enumerate(dataloader):
    loss = model(batch)
    wandb.log({
        "train/loss": loss.item(),
        "train/lr": scheduler.get_last_lr()[0],
        "step": step
    })
```

`wandb.log()` 支持记录多种类型：

| 类型 | 示例 |
|------|------|
| 标量 | `{"loss": 0.3}` |
| 图像 | `{"sample": wandb.Image(img_tensor)}` |
| 表格 | `{"results": wandb.Table(data=rows)}` |
| 直方图 | `{"weights": wandb.Histogram(param.data)}` |

> 📌 **命名技巧**：使用 `/` 作为命名空间分隔符（如 `train/loss`、`eval/loss`），W&B UI 会自动将同前缀的指标分组显示，让 Dashboard 更整洁。

---

### 完整最小示例

以下是一个可直接运行的最小训练脚本结构：

```python
import wandb

with wandb.init(
    project="diy-llm-training",
    name="run-001",
    config={"lr": 1e-4, "batch_size": 32, "epochs": 10}
) as run:
    for epoch in range(run.config.epochs):
        train_loss = train_one_epoch(...)   # 你的训练逻辑
        eval_loss  = evaluate(...)

        run.log({
            "train/loss": train_loss,
            "eval/loss":  eval_loss,
            "epoch": epoch,
        })
```

登录 [wandb.ai](https://wandb.ai) 即可看到实时 loss 曲线。

---

## 🧠 本模块问题

请在下方作答区填写答案，完成后输入 `提交作业` 提交。

**Q1**：`wandb.init()` 中的 `project` 和 `name` 参数分别代表什么？两个 run 可以属于同一个 project 吗？

**Q2**：为什么推荐将所有超参数放入 `config` 而不是直接写在代码里？这样做在多实验管理中有什么优势？

**Q3**：`wandb.log({"train/loss": 0.5, "eval/loss": 0.6})` 中用 `/` 分隔命名空间有什么实际效果？如果改成 `{"train_loss": 0.5, "eval_loss": 0.6}` 会有什么不同？

---

<!-- 学习者作答区（请在此处填写你的答案） -->

**A1**：project 代表了项目名称，name 代表了项目下的实验名称，或者简单理解就是 项目目录下的 子目录名称，用于区分同一个项目的不同实验。两个run可以属于同一个项目，但是不能有同一个 name。

**A2**：超参数的修改可能影响实验的最终结果，因此如果用来对比实验为什么结果出现差异，可以很直观得看出来哪些超参数不同导致得。而保存在 config 里面，可以通过 wandb 对比处差异，不写在 config 里，wandb 无法记录则无法直观得通过dashboard 区别开来。

**A3**：使用 train/loss 能将 train 的指标都放在一个组, 而 train_loss 不会在 前缀相同时放到同一组，只会放到不同的卡片中

---

<!-- 教师批改区（提交作业后由导师填写，请勿手动修改） -->

### 📝 批改结果

**Q1 批改**：主体正确，project 是项目容器，name 是 run 的可读标签，理解方向对。需要补充一点：**WandB 中同一 project 下多个 run 是可以重名的**——name 只是人类可读的标签，系统用的是自动生成的唯一 `run_id`。所以"不能有同一个 name"这个限制实际上不存在，不需要担心重名冲突。 — 得分：**8/10**

**Q2 批改**：非常好，抓住了核心价值点：config 存入 wandb 后可以在 dashboard 中横向对比多次实验的超参数差异，直接定位"是哪个超参数导致了结果不同"。可以再延伸一点：config 写入后，wandb 的 **Sweeps（自动调参）** 功能也依赖 config 结构来定义搜索空间，所以规范写 config 是后续使用 Sweeps 的前提。 — 得分：**9/10**

**Q3 批改**：理解正确。补充一下更精确的效果：使用 `/` 命名空间后，WandB UI 会将 `train/*` 的所有指标自动归入同一个 **Panel Group**（面板组），可以折叠/展开，也方便对比 `train/loss` 和 `eval/loss` 在同一张图上的走势。而 `train_loss` / `eval_loss` 这种命名会分散到各自独立的小卡片，当指标数量多时（例如同时记录 20+ 个指标）界面会非常凌乱。一个实用规则：`{阶段}/{指标名}`，如 `train/loss`、`eval/bleu`、`test/perplexity`。 — 得分：**8/10**

**综合评价**：整体掌握扎实，三道题都抓住了核心概念，思路清晰。主要需要注意 Q1 中 name 的唯一性认知，以及 Q3 可以更精确描述 Panel Group 的效果。可以继续进入模块 2。

**批改时间**：2026-04-13
