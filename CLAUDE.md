# CLAUDE.md — 仓库约定与配置

## 项目简介

本仓库是一个自学 LLM 的教程，包含 15 章课程内容和 6 个课后作业。

- **课程文档**：`docs/chapter{N}/`
- **课后作业（原始）**：`coursework/assignment{N}-*/`（来自 Stanford CS336，不修改）
- **课后作业（实现）**：`homework/assignment{N}/`（用户代码和笔记）

---

## 作业目录约定

### 目录结构

```
homework/assignment{N}/
├── tutorials/            # 分步教程（由导师生成）
│   ├── tutorial_part1.md
│   ├── tutorial_part2.md
│   └── ...
├── scripts/              # 用户实现代码
│   └── *.py
├── tests/                # 测试用例和 fixtures
│   ├── test_*.py
│   └── fixtures/
├── notes.md              # QA 记录 + 作业批改（逐字保留，只修改排版）
├── suggestion.md         # 完成后生成的学习建议（基于错误分析）
└── README.md             # 作业完成总结
```

### 关键文件说明

| 文件 | 用途 | 规则 |
|------|------|------|
| `notes.md` | 对话 QA + 批改记录 | **逐字保留**，只允许修改排版，不允许删减内容 |
| `suggestion.md` | 每个 part 通过后生成 | 基于错误次数/内容 + QA 记录分析知识薄弱点 |
| `tutorials/` | 分步教程 | 引导式，不直接给答案 |
| `tests/fixtures/` | 测试数据 | 大型文件通过 `.gitignore` 排除，README 中给出下载链接 |

### 作业工作流概览

详细流程见 `.claude/skills/repo-tutor/SKILL.md`，核心步骤：
1. 分析原作业 → 拆分为 2-4 个 part
2. 为每个 part 生成教程 → 用户实现 → 测试验证
3. part 通过后生成 `suggestion.md`
4. 全部 part 完成 → 汇总整合 → 更新进度

---

## 教学笔记目录约定

使用 `/repo-tutor` skill 学习时，每章内容会按模块拆分，生成增强版教学文档。

### 存储路径格式

```
docs/chapter{N}/c/module{M}.md
```

- `{N}`：章节编号（1-15，与 `docs/chapterN` 对应）
- `{M}`：模块编号（从 1 开始，简单章节可能只有 1 个模块）
- 目录 `c/` 专用于存放生成的教学笔记，不存放原始课程文件

### 示例

```
docs/chapter2/c/module1.md   # 第2章 BPE算法基础
docs/chapter2/c/module2.md   # 第2章 WordPiece与Unigram
docs/chapter4/c/module1.md   # 第4章 Transformer架构
```

### 模块文档结构

每个模块文档包含以下区域：

```markdown
## 模块 M：[标题]

### 核心概念
### 代码解析（如有对应 .py 文件）
### 🧠 本模块问题
<!-- 学习者作答区 -->
<!-- 教师批改区（提交作业后填写） -->
```

---

### 学习笔记文件（notes.md）

每章的 `c/` 目录下还会生成一个 `notes.md`，用于归档学习过程中的**临时问答记录**。

```
docs/chapter{N}/c/notes.md
```

**与模块文档的区别**：

| 文件 | 内容 | 写入时机 |
|------|------|---------|
| `module{M}.md` | 正式教学内容 + Q1/Q2/Q3 作业批改 | 生成模块时 / 提交作业后 |
| `notes.md` | 学习中临时提问 QA + 章节完成后的完整归档 | 进入下一模块前追加 / 章节完成后整理 |

**章节完成后 notes.md 整理规则**：
- 将所有模块文档中的正式 QA（Q1/Q2/Q3 + 批改）、延申思考批改、课程间隙临时 QA 合并到 notes.md
- **不修改原有模块文档中的 QA 内容**
- 在 notes.md 开头添加学习总结和课下建议（基于所有 QA 的作答情况和批改评分）

**notes.md 格式示例**：

```markdown
# 第 N 章 学习笔记

## 模块 1：[标题] — QA 记录
> 📅 2026-04-13

**Q**：[用户提问原文]

**A**：[导师回答要点]

---

## 模块 2：[标题] — QA 记录
...
```

**用途**：章节学习完成后，`notes.md` 是该章知识盲点的完整记录，可用于复习和总结。

---

## 章节 → 作业映射

| 章节 | 内容 | 对应作业 |
|------|------|----------|
| 第1章 | WandB 工具使用 | — |
| 第2章 | 分词器 | `assignment1-basics`（BPE 部分） |
| 第3章 | PyTorch 与资源核算 | `assignment1-basics`（基础部分） |
| 第4章 | 语言模型架构与训练细节 | `assignment1-basics`（Transformer 实现） |
| 第5章 | 混合专家模型（MoE） | — |
| 第6章 | GPU 与相关优化 | `assignment2-systems` |
| 第7章 | GPU 高性能编程 | `assignment2-systems` |
| 第8章 | 分布式训练 | `assignment2-systems` |
| 第9章 | Scaling Laws | `assignment3-scaling` |
| 第10章 | 推理 | — |
| 第11章 | 数据工程 | `assignment4-data` |
| 第12章 | 评估与基准测试 | `assignment6-evaluation` |
| 第13章 | 大模型基本训练流程 | `assignment5-alignment` |
| 第14章 | 可验证奖励的强化学习 | `assignment5-alignment` |
| 第15章 | 扩展内容 | — |

---

## 学习进度文件

进度保存在 `.claude/tutor-progress.json`，记录：
- 当前章节和模块
- 每模块问答历史
- 已完成的作业标记
