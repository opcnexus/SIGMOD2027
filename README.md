# RepSpace — Code and Data Artifact (Anonymized for Review)

**Representation Space Decides: A Platform-Independent Measurement and
Generalization-Diagnosis Framework for Interaction-Data Structure Prediction**

This repository is the reproducibility artifact for the paper above, prepared for
double-blind review. It contains **no author identity, no credentials, and no raw
corpora that cannot be redistributed**, but it **does contain every result file
needed to rebuild all tables and figures**.

> Fast verification (about 30 seconds, no data and no API key required):
> ```bash
> python3 scripts/verify_artifact.py            # integrity + compliance + unit tests
> python3 scripts/make_paper_tables.py --check  # rebuild tables, compare to published values
> ```
>
> Both commands need **only the standard library**: the chain-level metric kernel
> (`src/chain_metrics_core.py`) is separated from the experiment drivers, so no
> torch/numpy is needed (without torch, 5 of the regression tests skip themselves
> and the remaining 23 run normally).

---

## 1. What this artifact lets you do

`scripts/make_paper_tables.py --check` rebuilds the paper's tables from the result
files in this repository and compares them, item by item, with the published
numbers. **The shipped results currently reproduce 53 of 53 values.**

| Paper item | Content | Rebuilt from |
|---|---|---|
| Table 1 / Figure 1 | Representation-space profiles of the six platforms (root ratio, conversation/participant sizes, delays) | `results/raw_reduced/representation_space.json` |
| Table 2 | Corpus scale and quality gates | same + `data_space_analysis.json` |
| Table 4 | Main diagnosis (5 probes x 6 datasets, test F1, 5 seeds) | `results/full_benchmark_v2/summary.json` |
| Table 5 | Edge-class F1 broken down by gold structure | same (`cats.*` fields) |
| Table 6 | Cross-tier transfer decay matrix | `results/raw_reduced/transfer_experiment.json` |
| Table 7 | M3 recursive-backtracking reconstruction accuracy | `results/raw_reduced/backtrack_reconstruction.json` |
| Table 9 | Chain-level three-way comparison (ECB vs edge-level probe) | `results/raw_reduced/ecb_full.jsonl` |
| Table 10 | Routing prototype and oracle upper bound | `results/raw_reduced/moe_router.json` |
| Section 6.7 negative results | Per-conversation record of the merged-call + fixed-point iteration variant | `results/raw_reduced/ecb_v2_iterations.jsonl` |
| Figures 1-7 | All paper figures | `results/figures/*.png` |
| Section 6.8 controlled validation | 10 algorithms x 5 synthetic corpora, granularity-gap experiment | `experiments/backtracking_controls/` |

## 2. Layout

```
.
├── README.md                    This file
├── DATA_STATEMENT.md            Data sources, availability, redistribution limits (read first)
├── ARTIFACT_MANIFEST.md         File inventory + SHA256 + explicit exclusion list
├── LICENSE / CITATION.cff       License and citation
├── Makefile                     make verify | tables | test
├── src/                         Core pipeline (39 modules)
│   └── exploratory/             One-off scripts not required for the paper's claims (13, labelled)
├── tests/test_p0_fixes.py       Regression tests for chain metrics and LLM failure modes (28)
├── scripts/
│   ├── verify_artifact.py       Integrity and compliance checks
│   └── make_paper_tables.py     Table rebuild + comparison against published values
├── results/
│   ├── raw_reduced/             All result JSON/JSONL (<2MB threshold, see manifest)
│   ├── tables/                  Generated paper tables (CSV)
│   ├── figures/                 Paper figures
│   └── full_benchmark_v2/       Per-configuration checkpoints of the main benchmark
├── data_sample/                 Redacted data sample (structure only, no plaintext)
├── docs/
│   ├── chain_metric_definitions.md       Precise definitions of M1/M2/M3 and metric_version
│   ├── codebook_slack_mention_proxy.md  Silver labeling codebook and quality gates
│   └── reproduction_map.md      Script -> paper table reproduction paths
└── experiments/backtracking_controls/   Section 6.8 controlled validation (code + results)
```

## 3. Full reproduction (requires the original data)

Reproducing the paper's **tables needs no original data** (results are shipped).
To re-run the whole pipeline:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# (1) Obtain / build the corpora -- see DATA_STATEMENT.md
python3 src/download_data.py            # public corpora (IRC / Molweni)
python3 src/crawl_hn_text.py            # Hacker News explicit reply trees
python3 src/crawl_reddit_text.py        # Reddit explicit reply trees
python3 src/slack_silver.py             # Slack mention-proxy silver labels

# (2) Representation-space profiling and discriminability
python3 src/representation_space.py
python3 src/data_space_analysis.py

# (3) Probe training and the main benchmark (5 probes x 6 datasets x 5 seeds)
python3 src/full_benchmark_v2.py

# (4) Cross-tier transfer and its attribution control
python3 src/transfer_experiment.py
python3 src/transfer_norm.py

# (5) Chain-level metrics and the OCR-APT-style pruning control
python3 src/chain_metrics.py
python3 src/ocrapt_style.py

# (6) Routing prototype
python3 src/moe_router.py

# (7) LLM event-chain backtracking (API key required, see below)
export DEEPSEEK_API_KEY=sk-...          # or write ~/.repspace/deepseek.json
python3 src/ecb_full.py
```

### Hardware and runtime

- Probes / transfer / chain metrics: reproducible on Apple M2 (16 GB unified
  memory) or an equivalent CPU; no discrete GPU required. Per-configuration
  checkpoints are written atomically and a single seed reproduces bit-for-bit.
- LLM event-chain backtracking: the 101 conversations of `ecb_full.py` take about
  6.6 minutes on an M2 (call-level evidence in
  `results/raw_reduced/llm_call_log_metadata_only.jsonl`).
- The semantic probe (end-to-end DistilBERT fine-tuning) is the only long-running
  optional stage; the paper reports its cost-effectiveness verdict.

## 4. LLM dependency

`src/ecb_*.py` require an OpenAI-compatible Chat Completions endpoint. The key is
read in this order:

1. environment variable `DEEPSEEK_API_KEY` or `OPENAI_API_KEY`;
2. `~/.repspace/deepseek.json` shaped like `{"api_key": "sk-..."}`.

**This repository contains no credential.** `llm_response.py` classifies call
failures into four mutually exclusive modes (`network`, `output_truncated`,
`empty_content`, `malformed_response`); output truncation first triggers a retry
with a larger output budget, and only a persistent failure is recorded as
degradation -- so **engineering failures are never scored as model errors**.

## 5. Data availability (important)

Redistribution rights differ across the six platforms, so **no raw messages are
included**. Please read [`DATA_STATEMENT.md`](DATA_STATEMENT.md). In short:

- IRC, Molweni: public corpora, obtained from their official releases;
- Hacker News, Reddit: re-collectable via public APIs (Reddit subject to its ToS);
- Discord: public dumps, terms of use must be checked by the user;
- Slack: **internal de-identified export, not redistributable**; this repository
  ships only a **redacted structural sample** of its silver labels (pseudonymous
  authors, message text removed), under `data_sample/`.

## 6. Known limitations (consistent with the paper)

- Mention-proxy silver labels cover only 11.12% of messages; dark-zone behavior
  is inferred from indirect evidence.
- Zero-shot transfer uses the source-domain dev set for early stopping, which may
  underestimate transfer decay.
- The instantiation spans six platforms; validating the criteria on other tasks
  remains future work.
- Tier labels themselves contain a conventional component (leave-one-platform-out
  macro average 67%), so both platform-level (98.5%) and tier-level
  discriminability are reported.
- On Discord, 25 of 78 conversations (32%) admit no evaluable path F1; this is
  disclosed in the caption of Table 9.

## 7. License

Code is MIT-licensed (see `LICENSE`). Data and silver labels are **not** covered
by this license and remain subject to their original providers' terms; see
`DATA_STATEMENT.md`. The redacted sample under `data_sample/` is provided by this
artifact for review and verification purposes.

## 8. Note on remaining non-English content (deliberate)

Four files still contain non-ASCII characters. All of it is intentional and must
not be translated:

| File | What it contains | Why it stays |
|---|---|---|
| `src/exploratory/build_v4_docx.py` | The font names `宋体` / `黑体` and regex anchors that match the Chinese draft of the paper (`## 2 相关工作`, `参考文献`, ...) | The font names are written verbatim into the generated `.docx` XML (`w:eastAsia`); translating them to "SimSun" would change the emitted document. The regexes are tested against the Chinese draft, so translating them would silently break figure insertion, reference detection and caption handling. These are authoring helpers and are not needed to reproduce any result. |
| `src/exploratory/md_to_docx_v4.py` | Same category (one font list plus a few search patterns) | Same reason. |
| `results/raw_reduced/ecb_full.jsonl` | **Japanese** topic strings produced by the LLM | The Slack corpus in our experiments is a Japanese community, so the model labels chains in Japanese. This is experimental data, not documentation, and is preserved verbatim as the record of that run. |

All documentation, code comments, docstrings, user-facing strings, log messages
and result metadata are in English.
