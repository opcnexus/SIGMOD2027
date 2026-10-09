# Labeling Codebook: Slack Mention-Proxy Silver Labels (Strategy C)

This codebook defines the **only platform that requires a newly defined labeling
convention** (Slack retains no thread structure at all). Strategy A (explicit
platform tree) and Strategy B (message references) need no new convention; they
read platform fields directly.

## 1. When it applies

A platform that retains **no thread or reply relation whatsoever**, but does carry
a high-frequency `@display-name` mention signal.

## 2. Labeling rules (silver)

For each message `m_i`:

1. If the text of `m_i` contains no `<@display-name>` mention, then `m_i` has
   **no proxy parent edge**;
2. If it does, let the mentioned username set be `U(m_i)`. For each `u` in
   `U(m_i)`: among the messages **inside the current window** that precede
   `m_i` and whose author is `u`, take **the most recent one**, `m_j`, and record
   a candidate proxy edge `(m_i -> m_j)`;
3. **Keep multiple parents**: if `|U(m_i)| > 1`, all candidate edges are kept
   (this is the source of Slack's 0.369% multi-parent rate, above the 0.01%-2.71%
   of the human-annotated sets);
4. Windows are opened **weekly** (no edges are created across weeks), which avoids
   mistakenly linking messages from different time slices of the same channel;
5. A message that produces no proxy edge is **not** treated as a chain root:
   an empty `parents` list means "no mention", not "this is a new topic".

## 3. Differences from human annotation (must be disclosed to readers)

| Dimension | Human gold (IRC / Molweni) | Mention proxy (Slack) |
|---|---|---|
| Edge semantics | genuine reply relation | proxy for "a response to the addressee's most recent utterance" |
| Coverage | all messages | only 11.12% of messages produce proxy edges |
| Multi-parent rate | 0.01%-2.71% | 0.369% |
| Median parent-child delay | longer (annotation includes long-range replies) | 238 s (far shorter than Discord's 1,158 s references) |

Slack is therefore a **dark zone**: its silver labels are trustworthy only on the
high-confidence "mention" signal and must not be used to evaluate messages without
mentions. This is exactly why the paper calls Slack a dark zone and discusses its
chain-level metrics separately.

## 4. Quality gates (audit gates)

| Gate | Threshold | Strategy C measured | Action on failure |
|---|---|---|---|
| G1 parse rate | >= 90% | 91.84% | dataset may not enter the benchmark |
| G2 temporal violations | = 0 | 0 | any violation voids the whole corpus |
| G3 multi-parent rate | recorded, manual spot-check | 0.369% | fall back to single parent |

The purpose of the gates is to reconceive silver labels not as "degraded gold" but
as **independent signal sources with audit certificates**. The implementation is
`src/slack_silver.py`; the gate report is written next to the dataset.

## 5. Human reference for annotator agreement

Because strategy C is rule-based, its "consistency" is guaranteed by determinism
(the same input always yields the same output). What genuinely requires human
verification is whether the rule captures real reply relations. We support the
proxy labels with two pieces of evidence:

1. comparison with the delay distribution of Discord's explicit references
   (238 s vs 1,158 s -- complementary rather than conflicting);
2. manual spot-checking of the multi-parent edges.

## 6. How to reproduce

```bash
# requires an authorized Slack export placed in the format expected by data_loader.py
python3 src/slack_silver.py     # writes silver labels + the gate report
```

Each output record has the following shape (a redacted sample is provided under
`data_sample/`):

```json
{"dataset": "slack", "conv_id": "general-w2471",
 "msgs": [{"id": 0, "author": "u_<pseudonym>", "parents": [], "ts": 1494488768.5,
           "text": "<REDACTED>"}]}
```
