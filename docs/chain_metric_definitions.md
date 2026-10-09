# Precise Definitions of the Chain-Level Metrics (`metric_version = ecb-2.0`)

This document is the normative definition of the chain-level metrics used in
Sections 6.6 and 6.7 of the paper. **All chain-level numbers must be computed by
`src/chain_metrics_core.py::conv_chain_metrics`** (the zero-dependency kernel);
experiment scripts must not re-implement a metric under the same name. A previous
incident -- two ECB variants computing the same-named quantity with different
semantics -- is documented in Section 6.

## 0. Terminology

- **message**: one message of a conversation, indexed `0..n-1` in time order.
- **parent set** `parents[i]`: the predicted or gold set of direct upstream indices
  of message `i`.
- **gold**: all parent edges of the truncated conversation (chains may span
  multiple hops beyond the window); predicted edges remain under the window
  protocol.
- **forest**: the directed graph derived from the parent sets (a node may have
  several parents).

## 1. M1 -- Path F1 (event-chain level)

1. Build the forest from the parent sets, obtaining roots and leaves;
2. Enumerate all `root -> leaf` paths (iterative DFS, **capped at 2,000 per
   conversation**);
3. Compute precision / recall / F1 (x100) over **exact path tuples**.

- **Truncation is recorded**: when the cap is reached, `path_gold_truncated` /
  `path_pred_truncated` are returned; conversations hitting the cap should be
  excluded from cross-configuration comparisons.
- **Degenerate inputs**: when the prediction yields no path at all,
  `path_F1 = None` (**never silently zero-filled**); `path_F1_zero_filled` is also
  provided for continuity with earlier runs.

## 2. M2 -- Thread reconstruction (event-tree level)

Based on **weakly connected components** (union-find over all parent edges).

- `m2_component_recall` = |gold components intersect predicted components| /
  |gold components| -- this is the "thread reconstruction rate" reported in the
  paper;
- `m2_component_precision` = |intersection| / |predicted components|.

Both must be reported: recall alone hides "the prediction shatters into many
fragments", precision alone hides "most gold threads are missed". On Slack the two
differ drastically (precision 69.57% vs recall 2.98%), which directly shows that
the predicted structure is occasionally exactly right but almost always
incomplete. Both denominators **include singleton components**, consistent with
the aggregation in `dataset_probe_eval`.

## 3. M3 -- Recursive backtracking reconstruction accuracy

For every message that **has a gold parent**: take the predicted ancestor closure
`A_pred` by recursive backtracking to the root, the gold ancestor closure `A_gold`
similarly, compute their Jaccard similarity, and average over messages (x100).

- Gold roots have no ancestors by definition, so their closure Jaccard is
  undefined and they are **excluded** (this must be stated in metric
  documentation).
- Aggregation is a **macro average across conversations** (`n_anc_msgs` reports the
  sample size).

## 4. The edge-to-chain gap

High edge-level link F1 with low chain-level metrics reflects **cascade failure**:
a single backtracking error misaligns an entire chain. Tables 4 and 7 of the
paper presenting both side by side is the quantitative evidence for this
phenomenon.

## 5. Failure modes: making LLM engineering failures decidable

`src/llm_response.py` classifies call failures into four **mutually exclusive**
modes:

| Mode | Meaning | Handling |
|---|---|---|
| `network` | connection / timeout exception | backoff retry (<= 3) |
| `output_truncated` | `finish_reason == "length"` | **retry with `max_tokens` raised to the cap**, degrade only if it persists |
| `empty_content` | HTTP success but empty body | backoff retry |
| `malformed_response` | structurally invalid response | backoff retry |

`output_truncated` and `malformed_response` must be recorded separately from
`network`, otherwise **engineering failures are scored as model inadequacy**,
systematically understating the method. This separation is what allows Section 6.7
to disclose that "25 of 78 Discord conversations (32%) admit no evaluable path
F1".

## 6. Versioning and regression

`METRIC_VERSION = "ecb-2.0"` is defined in `src/chain_metrics_core.py`. Any change
to a metric definition must:

1. bump `METRIC_VERSION`;
2. update `tests/test_p0_fixes.py` accordingly;
3. explain the version difference in the paper.

Historical lesson: before this artifact existed, the component metric in
`ecb_deepseek_pilot.py` was **precision-type** while the one in
`ecb_deepseek_all.py` was **recall-type** -- same name, different semantics. Both
now delegate to this single implementation, and every result file records its
`metric_version` so that different batches remain distinguishable.
