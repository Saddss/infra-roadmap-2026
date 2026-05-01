# 推理工程师 6 个月冲刺工作区

> 24 周 · 每周 15h · 中文资料优先 · 目标头部大厂推理 infra 岗

这个工作区是按计划搭好的脚手架。每个 `Mx-*/` 目录对应计划中的一个阶段。每个阶段目录内含：

- `README.md` — 本阶段目标、每周拆解、核心资料链接
- `resources.md` — 完整资料库（带超链接，按主题分类）
- `weekly-tracker.md` — 每周打卡模板（学了什么 / 卡在哪 / 下周做什么）
- 项目脚手架（如 `mini-llama/`、`triton-flashattn/`）— 直接 `cd` 进去开始写代码

## 24 周进度看板

| 阶段 | 周次 | 主题 | 关键产出 | 状态 |
|---|---|---|---|---|
| **M1** | W1-W4 | Transformer / Attention 全家桶 | mini-llama 仓库 + 1 篇博客 | 🟡 进行中 |
| **M2** | W5-W8 | CUDA + Triton FlashAttention | triton-flash-attn 仓库 + benchmark | ⚪ 待开始 |
| **M3** | W9-W12 | vLLM 源码精读 | 2-3 个 Tier 0/1 PR + 3 篇博客 | ⚪ 待开始 |
| **M4** | W13-W16 | SGLang/TRT-LLM 横向 + 工程优化 | 1-2 个 Tier 2/3 PR + benchmark 报告 | ⚪ 待开始 |
| **M5** | W17-W20 | 并行 + V4/Qwen3.6 新架构 | 架构综述博客 + V4/Qwen3.6 跑通 | ⚪ 待开始 |
| **M6** | W21-W24 | RL 训推一体 + 多模态 + 面试 | RL 跑通 + 投简历 | ⚪ 待开始 |

每开始一个阶段，把状态改为 🟡；完成后改为 ✅。

## 总资源清单

→ 见 [`RESOURCES.md`](./RESOURCES.md)（按主题分类的所有学习链接）

## 反焦虑机制（每天必看）

1. **每周日 30 分钟**：在每阶段的 `weekly-tracker.md` 写一段周记
2. **不和别人比进度**：你 6 个月做到 vLLM PR + 架构综述已经超过 80% 同行
3. **不要用 AI 帮你写本来要练手的代码**：M1-M2 所有手撕代码必须自己写一遍，AI 只用来 debug 和 review
4. **每月模拟面试一次**：哪怕投简历去面别家"练手"

## 工作流约定

```bash
cd ~/sss/infra-roadmap-2026/M1-transformer-fundamentals  # 切到当前阶段
code .                                                 # 用 Cursor 打开
```

每周日：

1. 打开当前阶段的 `weekly-tracker.md` 填本周打卡
2. 把博客草稿放在 `notes/` 目录
3. 代码 commit 到对应的子项目仓库

## 阶段切换检查表

进入下一阶段前，确认：

- [ ] 本阶段的产出物（仓库 / 博客 / PR）已经发出去（不是躺在本地）
- [ ] `weekly-tracker.md` 4 周记录都填了
- [ ] 把简历对应那段更新一下

## 工作区脚手架已就绪

> 这个工作区共 111 个文件、55 个目录，按月分块。所有"待你填空"的代码都标了 `TODO Wx:` 注释。

### 已经能跑的部分（验证过）

```bash
# M1: mini-llama 测试结构（23 个测试可被 pytest 发现）
cd M1-transformer-fundamentals/mini-llama && pytest --collect-only

# M5: DSA Lightning Indexer demo（已验证因果性正确）
cd M5-parallelism-architectures/w19-w20-architectures/deepseek-v3-v32-v4
CUDA_VISIBLE_DEVICES="" python dsa_lightning_indexer.py

# M5: Gated DeltaNet demo（已验证 O(1) 状态）
cd M5-parallelism-architectures/w19-w20-architectures/qwen3-qwen36
CUDA_VISIBLE_DEVICES="" python gated_deltanet.py

# M6: R1 reward function（已验证打分正确）
cd M6-rl-multimodal-interview/w21-w22-rl/r1-reproduction
python reward_function.py
```

### 你 W1 第一天就能开始的事情

```bash
# 1. 切到 M1 目录，看 README
cd ~/sss/infra-roadmap-2026/M1-transformer-fundamentals
cat README.md

# 2. 装依赖
cd mini-llama
python -m venv .venv && source .venv/bin/activate
pip install -e .

# 3. 看到所有 TODO 等你填
grep -r "TODO W" mini_llama/

# 4. 开始 W1 第一题：实现 single-head attention
$EDITOR mini_llama/attention/single_head.py

# 5. 跑测试看你的实现对不对
pytest tests/test_w1_single_head.py -v
```

### 5 个核心 GitHub 仓库

执行计划过程中会产出 5 个 GitHub 仓库：

1. **mini-llama**（M1 末 push）—— `M1-transformer-fundamentals/mini-llama/`
2. **triton-flashattn**（M2 末 push）—— `M2-cuda-triton/triton-flashattn/`
3. **vLLM fork**（M3 期间 fork）—— 单独 clone 到本地，PR 走这里
4. **infer-bench-2026**（M4 末 push）—— `M4-inference-frameworks/benchmark-report/`
5. **arch-evolution-2026**（M5 末 push）—— `M5-parallelism-architectures/w19-w20-architectures/`

### 8-10 篇博客的对应位置

| 阶段 | 博客标题 | 模板位置 |
|---|---|---|
| M1 W4 | 我手撕 MHA→GQA→MLA 的笔记 | `M1/resources.md` 末尾 |
| M2 W8 | 我用 Triton 100 行写了 FlashAttention | `M2/resources.md` 末尾 + `notes/fa_v1_v2_diff.md` |
| M3 W10 | PagedAttention 原理与 Block Table 实现 | `M3/blog-drafts/02-...` |
| M3 W11 | vLLM 一个请求的一生 | `M3/blog-drafts/01-...` |
| M3 W12 | vLLM v1 vs v0 架构演进 | `M3/blog-drafts/03-...` |
| M4 W16 | vLLM/SGLang/TRT-LLM 横向 benchmark | `M4/benchmark-report/REPORT_TEMPLATE.md` |
| M5 W20 | 2026 大模型架构两大流派 · V4 vs Qwen3.6 | `M5/.../arch-comparison-blog/blog-template.md` |
| M6 W22 | 我跑通 R1 复现的踩坑记录 | （直接在博客平台写） |
| M6 W24 | 6 个月推理工程师重启计划复盘 | （直接在博客平台写） |
