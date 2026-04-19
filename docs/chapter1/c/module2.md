# 第 1 章：工具使用（WandB）— 模块 2：进阶功能

> 📍 学习进度：第 1 章，第 2 / 2 模块
> 📅 生成时间：2026-04-13

---

## 学习目标

- 掌握离线模式的配置与后续同步方法（无外网服务器场景）
- 理解 Artifact 的概念，能用它管理模型 checkpoint 的版本
- 能快速定位并解决 WandB 常见报错

---

## 核心内容

### 离线模式（Offline Mode）

在无外网的服务器（如大多数 GPU 集群）上训练时，不能实时上传日志，需要切换到离线模式：

```python
wandb.init(mode="offline", project="diy-llm", ...)
```

所有日志会保存在本地 `wandb/` 目录下，格式为：

```
wandb/
└── offline-run-20260116_113519-vc1rtokn/
    ├── files/
    └── run-vc1rtokn.wandb
```

#### 训练结束后同步到云端

将 `wandb/` 目录拷贝到有网络的机器，执行：

```bash
# 同步整个目录下所有离线 run
wandb sync wandb/

# 只同步某一个 run
wandb sync wandb/offline-run-20260116_113519-vc1rtokn
```

> ⚠️ **注意**：目标机器需先 `wandb login`，且对应 project 需存在（或会自动创建）。

#### 三种 mode 对比

| mode | 行为 | 适用场景 |
|------|------|----------|
| `"online"`（默认） | 实时上传 | 本地开发、有外网 |
| `"offline"` | 本地保存，手动同步 | GPU 集群、无外网 |
| `"disabled"` | 完全静默，无任何副作用 | 调试代码、CI 测试 |

---

### 保存模型与工件（Artifacts）

**Artifact** 是 WandB 的版本化文件管理系统，适合管理：
- 模型 checkpoint（`.pt`、`.safetensors`）
- 数据集
- 评估结果

#### 上传 Artifact

```python
artifact = wandb.Artifact(
    name="llama3-70b-wanda-c4",   # artifact 名称
    type="model"                   # 类型标签：model / dataset / result
)
artifact.add_file("checkpoints/model.safetensors")
wandb.log_artifact(artifact)
```

上传后，WandB 会自动给这个版本打标签（`v0`、`v1`……），可在 UI 中浏览历史版本。

#### 下载并复用 Artifact

```python
# 在另一个实验中引用已有模型
artifact = run.use_artifact("llama3-70b-wanda-c4:latest")
artifact_dir = artifact.download()   # 下载到本地，返回目录路径
```

> 💡 **补充资料（来源：Context7 / wandb 官方文档）**
>
> 除了 `Artifact`，也可以用 `run.save()` 做轻量级文件同步，支持多种上传策略：
>
> ```python
> run.save("checkpoints/*.pt")               # 训练结束时上传
> run.save("logs/train.log", policy="live")  # 实时同步（文件追加写入时适用）
> run.save("results.json", policy="now")     # 立即上传
> ```
>
> **Artifact vs save 如何选择**：
> - 需要版本管理、跨 run 复用 → 用 **Artifact**
> - 只是备份某次 run 的附属文件 → 用 **save**

> 🔒 **大模型注意**：大模型参数文件（如 70B 模型）体积巨大，上传 Artifact 会消耗大量存储配额，**不建议上传模型权重**，可改为只上传 config 和训练日志。

---

### 常见问题速查

#### Q：初始化超时（`Run initialization has timed out`）

网络不稳定或服务器访问 wandb.ai 受限时出现：

```python
# 方案1：增加超时时间
wandb.init(settings=wandb.Settings(init_timeout=120), ...)

# 方案2：直接切换离线模式
wandb.init(mode="offline", ...)
```

#### Q：调试时不想产生 wandb 记录怎么办？

```python
wandb.init(mode="disabled")  # 完全静默，所有 wandb.log() 调用变为 no-op
```

适合写单元测试或快速验证代码逻辑时使用。

---

### 第 1 章小结

| 知识点 | 关键 API |
|--------|----------|
| 初始化实验 | `wandb.init(project=, name=, config=)` |
| 记录指标 | `wandb.log({"train/loss": ...})` |
| 离线模式 | `mode="offline"` + `wandb sync` |
| 版本化文件 | `wandb.Artifact` + `log_artifact` |
| 调试静默 | `mode="disabled"` |

---

## 🧠 本模块问题

请在下方作答区填写答案，完成后输入 `提交作业` 提交。

**Q1**：在无外网的 GPU 服务器上训练，应该怎么做才能最终把实验数据同步到 wandb.ai？请写出完整流程（至少 3 步）。

**Q2**：Artifact 和直接用 `run.save()` 保存文件有什么核心区别？什么时候该选 Artifact？

---

<!-- 学习者作答区（请在此处填写你的答案） -->

**A1**：
1. 先使用 wandb 离线保存 wandb.init(mode='offline')
2. 拷贝 wandb 文件夹到有网络连接（且 wandb 登录的主机上）
3. 执行 wandb sync ./wandb 文件夹


**A2**：
Artifact 是 跨 run 保存的，也就是其他项目/实验也可以 download 使用，而且会自动有版本管理，v1 v2 …… 等。
save 只是该轮实验下的便捷备份，只在当前实验保存。
---

<!-- 教师批改区（提交作业后由导师填写，请勿手动修改） -->

### 📝 批改结果

**Q1 批改**：完全正确，三步流程清晰准确。补充一个细节：`wandb sync` 时也可以只同步单个 run（`wandb sync wandb/offline-run-xxx`），在离线跑了很多实验但只想上传某一次结果时很有用。 — 得分：**10/10**

**Q2 批改**：核心区别抓住了——版本管理和跨 run 复用是 Artifact 的本质优势。补充两点让理解更完整：① Artifact 的"跨项目"能力不只是 download，还可以作为另一个实验的**输入依据**（`run.use_artifact()`），WandB 会自动建立实验间的血缘图（lineage graph），便于追溯"这个模型是用哪个数据集训练的"；② `run.save()` 的文件虽然也上传到云端，但没有独立的命名和版本号，在 UI 里只能在当次 run 的 Files 标签页找到，无法被其他实验引用。选择原则：**需要被引用或复用 → Artifact；只是备份附件 → save**。 — 得分：**9/10**

**综合评价**：两题都回答到位，对 WandB 核心机制理解清晰。第 1 章全部完成，可以进入第 2 章了。

**批改时间**：2026-04-13
