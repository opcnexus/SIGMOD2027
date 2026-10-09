# RepSpace — 代码与数据工件（匿名评审版）

**Representation Space Decides: A Platform-Independent Measurement and
Generalization-Diagnosis Framework for Interaction-Data Structure Prediction**

本仓库是上述论文的**可复现工件**，供双盲评审使用。仓库内**不含作者身份信息、密钥或
不可再分发的原始语料**，但**完整包含重建论文全部表格与图件所需的结果文件**。

> 快速验证（30 秒，无需任何数据或密钥）：
> ```bash
> python3 scripts/verify_artifact.py     # 完整性 + 合规性 + 单元测试
> python3 scripts/make_paper_tables.py --check   # 重建论文表格并与已发表数值比对
> ```
>
> 两条命令**只需标准库**：链级指标内核 `src/chain_metrics_core.py` 已与实验驱动分离，
> 无需 torch / numpy（缺 torch 时回归测试自动跳过其中的 5 项，其余 23 项照常运行）。

---

## 1. 这个工件能做什么

`scripts/make_paper_tables.py --check` 会从 `results/` 中的结果文件重建论文表格，
并与论文已发表数值逐项比对。当前随包结果的复现结果为 **53/53 项一致**。

| 论文表 | 内容 | 由哪个结果文件重建 |
|---|---|---|
| 表 1 / 图 1 | 六平台表征空间画像（root%、会话/参与者规模、时延） | `results/raw_reduced/representation_space.json` |
| 表 2 | 语料规模与质量门禁 | 同上 + `data_space_analysis.json` |
| 表 4 | 主诊断（5 探针 × 6 数据集 test F1，5 seeds） | `results/full_benchmark_v2/summary.json` |
| 表 5 | 按金标结构分解的边类别 F1 | 同上（`cats.*` 字段） |
| 表 6 | 跨层级迁移衰减矩阵 | `results/raw_reduced/transfer_experiment.json` |
| 表 7 | M3 递归回溯重构准确率 | `results/raw_reduced/backtrack_reconstruction.json` |
| 表 9 | 链级三方对照（ECB vs 边级探针） | `results/raw_reduced/ecb_full.jsonl` |
| 表 10 | 路由原型与 Oracle 上界 | `results/raw_reduced/moe_router.json` |
| §6.7 负结果 | v2 合并调用 + 不动点迭代的逐会话记录 | `results/raw_reduced/ecb_v2_iterations.jsonl` |
| 图 1–7 | 论文全部图件 | `results/figures/*.png` |
| §6.8 受控验证 | 10 算法 × 5 合成语料的粒度鸿沟实验 | `experiments/backtracking_controls/` |

## 2. 目录结构

```
.
├── README.md                    本文件
├── DATA_STATEMENT.md            数据来源、可获得性与再分发限制（必读）
├── ARTIFACT_MANIFEST.md         文件清单 + SHA256
├── LICENSE / CITATION.cff       许可与引用
├── Makefile                     make verify | tables | test
├── src/                         核心管线（39 个模块，含零依赖指标内核 chain_metrics_core.py）
│   └── exploratory/             非论文主张所需的一次性脚本（13 个，已标注）
├── tests/test_p0_fixes.py       链级指标与 LLM 失败模式的回归测试（28 项）
├── scripts/
│   ├── verify_artifact.py       工件完整性/合规性校验
│   └── make_paper_tables.py     论文表格重建 + 与发表值比对
├── results/
│   ├── raw_reduced/             全部结果 JSON/JSONL（<2MB 阈值，见 manifest）
│   ├── tables/                  生成的论文表格 CSV
│   ├── figures/                 论文图件
│   └── full_benchmark_v2/       主基准逐组合检查点
├── data_sample/                 脱敏数据样例（仅结构，无明文）
├── docs/
│   ├── chain_metric_definitions.md   M1/M2/M3 与 metric_version 的精确定义
│   ├── codebook_slack_mention_proxy.md  silver 标注约定与质量门禁
│   └── reproduction_map.md      脚本 → 论文表的复现路径
└── experiments/backtracking_controls/   §6.8 受控验证实验（代码 + 结果）
```

## 3. 从零复现（需要原始数据时）

论文的**表格复现不需要原始数据**（结果已随包）。若要重跑完整流水线：

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# (1) 获取/构建语料 —— 详见 DATA_STATEMENT.md 的获取指引
python3 src/download_data.py            # 公开语料（IRC / Molweni）
python3 src/crawl_hn_text.py            # Hacker News 显式回复树
python3 src/crawl_reddit_text.py        # Reddit 显式回复树
python3 src/slack_silver.py             # Slack mention 代理 silver 标签

# (2) 表征空间画像与可辨识性
python3 src/representation_space.py
python3 src/data_space_analysis.py

# (3) 探针训练与主基准（5 探针 × 6 数据集 × 5 seeds）
python3 src/full_benchmark_v2.py

# (4) 跨层级迁移 + 归因对照
python3 src/transfer_experiment.py
python3 src/transfer_norm.py

# (5) 链级指标与 OCR-APT 式剪枝对照
python3 src/chain_metrics.py
python3 src/ocrapt_style.py

# (6) 路由原型
python3 src/moe_router.py

# (7) LLM 事件链回溯（需 API key，见下）
export DEEPSEEK_API_KEY=sk-...          # 或写入 ~/.repspace/deepseek.json
python3 src/ecb_full.py
```

### 硬件与运行时间

- 探针/迁移/链级指标：Apple M2（16GB 统一内存）或等效 CPU 可复现，无需独立显卡；
  逐组合检查点原子落盘，单 seed 可逐位复现。
- LLM 事件链回溯：`ecb_full.py` 的 101 个会话在 M2 上约 6.6 分钟（3,932 次调用量级见
  `results/raw_reduced/llm_call_log_metadata_only.jsonl`）。
- 语义探针（DistilBERT 端到端微调）是唯一需要长时训练的可选项，论文中已标注其性价比结论。

## 4. LLM 依赖说明

`src/ecb_*.py` 依赖一个 OpenAI 兼容的 Chat Completions 端点。密钥读取顺序：

1. 环境变量 `DEEPSEEK_API_KEY` 或 `OPENAI_API_KEY`；
2. `~/.repspace/deepseek.json`，形如 `{"api_key": "sk-..."}`。

**本仓库不含任何密钥**。`llm_response.py` 将调用失败分为四种互斥模式
（`network` / `output_truncated` / `empty_content` / `malformed_response`），
其中输出截断会先提升 `max_tokens` 重试，仍失败才记录降级——因此**工程失败不会被计为模型错误**。

## 5. 数据可获得性（重要）

六平台语料的**再分发权限不一致**，因此本仓库**不包含原始消息**。请先阅读
[`DATA_STATEMENT.md`](DATA_STATEMENT.md)。要点：

- IRC、Molweni：公开语料，按原发布条款获取；
- Hacker News、Reddit：公开 API 可重新采集（Reddit 需遵守其 ToS）；
- Discord：公开转储，需自行确认使用条款；
- Slack：**内部渠道脱敏导出，不可再分发**；本仓库仅提供 silver 标签的**脱敏结构样例**
  （作者假名化、消息正文移除），见 `data_sample/`。

## 6. 已知限制（与论文一致）

- 提及代理 silver 标签仅覆盖 11.12% 的消息，暗区行为由间接证据推断；
- 零样本迁移的早停采用源域开发集，迁移衰减可能被低估；
- 实例含六个平台，判据在其他任务上的验证属未来工作；
- 层级标签本身含约定成分（留一平台交叉验证宏平均 67%），故同时报告平台级（98.5%）与层级级可辨识性；
- Discord 的 78 个会话中有 25 个（32%）无法产生可评估的路径 F1，已在表 9 脚注披露。

## 7. 许可

代码采用 MIT（见 `LICENSE`）。数据与 silver 标签遵循各自来源的条款，详见
`DATA_STATEMENT.md`；`data_sample/` 中的脱敏样例由本工件提供，可自由用于评审与验证。
