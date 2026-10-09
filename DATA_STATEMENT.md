# Data Sources, Availability, and Redistribution Limits

This artifact follows the principle of being **reviewable and reproducible
without over-reaching redistribution rights**. Licensing status differs across the
six platforms, so **no raw message text is included in this repository**. What is
included: result files, a **redacted structural sample** of the silver labels, and
the collection and labeling code needed to regenerate them.

## 1. Per-platform status

| Platform | Tier | Signal strategy | Size | Licensing status | Raw data in this repo |
|---|---|---|---|---|---|
| IRC | annotated conversational | human gold | 49,765 messages | publicly released corpus, used under its original terms | No (obtain from the official source) |
| Molweni | annotated conversational | human gold | 88,303 messages | public dataset (ACL 2020) | No (official GitHub release) |
| Hacker News | forum tree | A: explicit reply tree | 136,327 messages | public API (Firebase) | No (`src/crawl_hn_text.py` re-collects) |
| Reddit | forum tree | A: explicit reply tree | 62,224 messages | public API, **subject to the Reddit ToS** | No (`src/crawl_reddit_text.py` re-collects; must respect the ToS) |
| Discord | production chat | B: message-reference signal | 58,420 messages | public dumps; the user must confirm the terms of use | No (`src/data_loader.py` reads a local dump) |
| Slack | production chat | C: mention-proxy silver | 80,187 messages | **internal de-identified export, not redistributable** | No (redacted structural sample only) |

Additional note: the paper's experiments also include a Bilibili collection
script (`src/exploratory/crawl_bilibili*.py`), whose data does **not** enter any
paper claim and is therefore not listed here.

## 2. Why the Slack silver labels are not shipped either

The Slack mention-proxy silver labels are built from message metadata (usernames,
timestamps, parent edges) and contain **no message text**. Usernames are still
personal data, so this repository ships only:

- `data_sample/slack_mention_proxy_silver.sample.jsonl`: 30 conversations with
  **irreversibly pseudonymized authors** (salted SHA-256) and **all message text
  replaced by a placeholder**;
- the complete labeling rules and quality gates in
  `docs/codebook_slack_mention_proxy.md`;
- the code that produces them, `src/slack_silver.py`.

Reviewers who need to reproduce the Slack labels must obtain an authorized export
themselves; we neither distribute that data nor take responsibility for its
compliance.

## 3. Audit certificate of the silver labels

Strategy C (mention proxy) shares the same quality gates as strategies A/B
(paper Table 2):

| Gate | Threshold | Strategy C measured |
|---|---|---|
| G1 parse rate | >= 90% | 91.84% |
| G2 temporal violations | = 0 | 0 |
| G3 multi-parent rate | recorded and manually spot-checked | 0.369% |

A dataset that fails a gate **may not enter the benchmark**. The median
parent-child delay of 238 s is complementary to Discord's explicit references at
1,158 s, indicating that mentions are a higher-frequency, stronger-intent reply
signal.

## 4. Suggested order for obtaining the original data

1. IRC / Molweni: download from their official release pages (entry points are
   documented in `src/download_data.py`);
2. Hacker News / Reddit: re-collect through the public APIs, respecting their
   terms of use and rate limits;
3. Discord: use a public dump and **confirm the terms of use** before using it for
   research;
4. Slack: **you must obtain authorization yourself**. Without it, the paper's core
   criteria (representation-space measurement, gain locality, transfer asymmetry,
   the missing chain-level metrics) still hold on the remaining five platforms;
   only the platform count in the corresponding text needs to be adjusted.

## 5. Ethics

The corpora used in this study come from public interfaces, publicly released
corpora, and authorized internal de-identified exports. Mention-proxy labeling
uses **only message metadata and involves no redistribution of message content**.
Applying silver-labeling methods to settings such as employee-behaviour analysis
carries a risk of surveillance abuse; usage restrictions accompany the release.

If you find that this repository inadvertently contains material that should not
be redistributed, please contact the review process and we will remove it
promptly.
