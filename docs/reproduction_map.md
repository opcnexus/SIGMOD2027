# Reproduction Map: Script -> Paper Table / Figure

This file gives the correspondence from scripts to every number in the paper so
that reviewers can verify item by item.

## 0. Reproduction without data (do this first)

```bash
python3 scripts/verify_artifact.py             # integrity + compliance + 28 unit tests
python3 scripts/make_paper_tables.py --check  # rebuild tables and compare with published values
```

The second command prints `[PASS]/[FAIL]` per item; the shipped results currently
give **53/53 consistent**.

## 1. Per-table reproduction paths

| Paper location | Script | Key settings | Output |
|---|---|---|---|
| Section 4.1 platform discriminability (98.5% / 67%) | `src/representation_space.py`, `src/data_space_analysis.py`, `src/profiling_discriminator.py` | conversation-level structural features, 5-fold CV, LOPO for tiers | `results/raw_reduced/representation_space.json`, `data_space_analysis.json` |
| Section 5, Table 2 corpus and gates | `src/slack_silver.py` + gate report | strategies A/B/C, gates G1-G3 | `results/raw_reduced/data_space_analysis.json` |
| Section 5, Table 3 window coverage | built into the protocol (`W=100`) | forward window width | measured values shipped in `results/raw_reduced/` |
| Section 6.2, Table 4 main diagnosis | `src/full_benchmark_v2.py` | 5 probes, 12 metrics, 5 seeds (42-46), W=100, truncation 1200 | `results/full_benchmark_v2/summary.json` |
| Section 6.2, Table 5 edge-class breakdown | same (`cats.*` fields) | -- | same |
| Section 6.3, Table 6 transfer matrix | `src/transfer_experiment.py` | unified probe, char-3gram 64d, MHA | `results/raw_reduced/transfer_experiment.json` |
| Section 6.3 delay-drift falsification | `src/transfer_norm.py` | replace absolute delay by quantile rank | same experiment, control branch (`transfer_norm`) |
| Section 6.5 semantic probe | `src/bert_encode.py`, `src/digat_ft.py` | DistilBERT, bottom 4 layers frozen, top 2 fine-tuned | `results/raw_reduced/bert_ft_official_v2.json` |
| Section 6.6, Table 7 M3 | `src/chain_metrics.py` | `metric_version=ecb-2.0` | `results/raw_reduced/backtrack_reconstruction.json` |
| Section 6.6, Table 8 OCR-APT-style pruning | `src/ocrapt_style.py` | bidirectional causal pruning + staged verification | `results/raw_reduced/ocrapt_*.json` (when present) |
| Section 6.7, Table 9 ECB chain level | `src/ecb_full.py` | window 300, 2 calls/window, gates | `results/raw_reduced/ecb_full.jsonl` |
| Section 6.7 negative result (v2) | `src/ecb_v2.py` | merged call + fixed-point iteration (<= 3 rounds) | `results/raw_reduced/ecb_v2_iterations.jsonl` |
| Section 6.7 negative result (convention prior) | `src/ecb_v3_constraint.py` | mentions parsed into recent-same-author edges | `results/raw_reduced/ecb_v3_constraint.jsonl` |
| Section 7, Table 10 routing | `src/moe_router.py` | linear router over 7 profile features | `results/raw_reduced/moe_router.json` |
| Section 6.8 controlled validation | `experiments/backtracking_controls/code/experiments/0*.py` | 10 algorithms x 5 synthetic corpora, two conditions (mock p=0.90 and real LLM) | `experiments/backtracking_controls/results/metrics_*.json` |

## 2. Figures

| Figure | Source |
|---|---|
| Fig. 1 platform root-ratio profile | `results/figures/fig1_layering.png` (plotted from `representation_space.json`) |
| Fig. 2 three stages and four operators | `results/figures/fig2_pipeline.png` |
| Fig. 3 system architecture | `results/figures/fig3_architecture.png` |
| Fig. 4 main comparison | `results/figures/fig4_full_comparison.png` |
| Fig. 5 chain-length decay (with theoretical $p^{L}$) | `results/figures/fig5_decay.png` (plotted by the Section 6.8 controlled experiment) |
| Fig. 6 edge-level vs chain-level gap | `results/figures/fig6_gap.png` (same) |
| Fig. 7 chain-level EM per corpus | `results/figures/fig7_perdataset.png` (same) |

## 3. Determinism

- Probe training uses fixed seeds 42-46; results are reported as mean +/- standard
  deviation. Per-configuration checkpoints are written atomically and a single seed
  reproduces bit-for-bit.
- LLM sampling temperature is 0.2. Note that `temperature=0` makes
  majority-vote / self-consistency mechanisms degenerate to a single call, which is
  why this artifact consistently uses 0.2.
- The controlled experiment (Section 6.8) is fixed with `seed=20261002`; the mock
  oracle's per-hop accuracy `p=0.90` is configurable.

## 4. Two deliberate reductions in the shipped results

1. `results/raw_reduced/llm_call_log_metadata_only.jsonl`: derived from the full
   LLM call log by **removing the `response` field**. It keeps timestamps, latency,
   `usage`, `finish_reason`, `failure_mode`, retry counts and budget escalation,
   which is what an audit needs, without shipping any conversation content.
2. `data_sample/`: a pseudonymized, text-stripped sample of the silver labels
   (see `DATA_STATEMENT.md`).

Excluded content (the 17 GB feature cache, model weights, the 18 MB v3 log) does
not affect the reproduction of any number in the paper; see
`ARTIFACT_MANIFEST.md` for the full exclusion list.
