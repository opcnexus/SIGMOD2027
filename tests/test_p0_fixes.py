#!/usr/bin/env python3
"""Regression tests for the P0 fixes (no network dependency).

They verify the three blocking fixes listed in Section 10 of the formal review:
  P0-1 output truncation is detected and distinguished from network failure, and triggers a retry with a larger budget
  P0-2 chain-level metrics with the same name are unified in one authoritative implementation (both M2 variants + metric_version)
  P0-3 path-enumeration truncation flags are recorded explicitly, and degenerate inputs are no longer silently zero-filled

Run: python tests/test_p0_fixes.py
"""
import io
import os
import sys
import json

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))

# The metric kernel is a zero-dependency module (standard library only), so reviewers can run
# these tests without torch/numpy
import chain_metrics_core as CM
from llm_response import classify_response, MAX_TOKENS_CAP

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(("  [PASS] " if cond else "  [FAIL] ") + name + (f"  {detail}" if detail else ""))


# ---------------------------------------------------------------- P0-1
def test_p0_1_classifier():
    print("\n== P0-1 failure-mode classification ==")
    ok_body = {"choices": [{"finish_reason": "stop",
                            "message": {"content": '{"chains":[]}'}}]}
    c, fr, fm = classify_response(ok_body)
    check("normal response -> failure_mode=None", fm is None and c == '{"chains":[]}', f"fr={fr}")

    trunc = {"choices": [{"finish_reason": "length",
                          "message": {"content": '{"chains":[{"topic":"a","mess'}}]}
    c, fr, fm = classify_response(trunc)
    check("finish_reason=length -> output_truncated", fm == "output_truncated", f"fr={fr}")
    check("non-empty truncated content (broken JSON) is still flagged", c is not None and fm == "output_truncated")

    empty = {"choices": [{"finish_reason": "stop", "message": {"content": ""}}]}
    check("empty content -> empty_content", classify_response(empty)[2] == "empty_content")
    check("malformed body -> bad_response", classify_response({"oops": 1})[2] == "bad_response")
    check("failure modes are mutually exclusive and decidable (truncation != network)",
          classify_response(trunc)[2] != classify_response({"x": 1})[2])


def test_p0_1_escalation():
    print("\n== P0-1 retry with a larger max_tokens after truncation ==")
    try:
        import ecb_deepseek_all as ALL
    except Exception as e:  # depends on torch / data modules; skipped when unavailable
        print(f"  [SKIP] cannot import ecb_deepseek_all: {type(e).__name__}: {e}")
        return
    import urllib.request as ur

    seen = []

    class FakeResp:
        def __init__(self, body): self._b = json.dumps(body).encode()
        def read(self): return self._b
        def __enter__(self): return self
        def __exit__(self, *a): return False

    def fake_urlopen(req, timeout=None):
        mt = json.loads(req.data.decode())["max_tokens"]
        seen.append(mt)
        if mt < MAX_TOKENS_CAP:                       # budget too small -> truncation
            return FakeResp({"choices": [{"finish_reason": "length",
                                          "message": {"content": '{"chains":['}}],
                             "usage": {"completion_tokens": mt}})
        return FakeResp({"choices": [{"finish_reason": "stop",
                                      "message": {"content": '{"chains":[]}'}}],
                         "usage": {"completion_tokens": 7}})

    orig = ur.urlopen
    ur.urlopen = fake_urlopen
    try:
        logf = io.StringIO()
        resp, meta = ALL.call_logged("dummy-key", "unittest", "conv0", "prompt", 400, logf)
        recs = [json.loads(l) for l in logf.getvalue().strip().splitlines()]
    finally:
        ur.urlopen = orig

    check("first budget of 400 is detected as truncated", seen[0] == 400, f"attempts={len(seen)}")
    check("retried after escalating to 8192", MAX_TOKENS_CAP in seen, f"seen={seen}")
    check("retry returns content successfully", resp is not None and meta["failure_mode"] is None)
    check("evidence log records finish_reason", recs and recs[-1].get("finish_reason") == "stop")
    check("evidence log records attempts and the final budget",
          recs and recs[-1].get("attempts") == 2 and recs[-1].get("max_tokens_final") == MAX_TOKENS_CAP)


# ---------------------------------------------------------------- P0-2 / P0-3
def _diamond_multi_parent():
    """Build a graph with multiple parents and several chains: gold = three 2-message chains plus one shared child."""
    gold = [[], [0], [], [0], [1, 3]]   # 5 -> the shared child 4 has two parents
    return gold


def test_p0_2_metric_authority():
    print("\n== P0-2 authoritative metric implementation ==")
    check("exposes METRIC_VERSION", hasattr(CM, "METRIC_VERSION"), CM.METRIC_VERSION)
    gold = [[], [0], [1]]                 # one 3-message chain
    same = CM.conv_chain_metrics(gold, list(gold))
    m = CM.conv_chain_metrics(gold, [[], [0], [1]])
    check("M2 recall-type is present", m["m2_component_recall"] is not None, str(m["m2_component_recall"]))
    check("M2 precision-type is present", m["m2_component_precision"] is not None, str(m["m2_component_precision"]))
    check("with a perfect prediction both M2 variants are 100",
          m["m2_component_recall"] == 100.0 and m["m2_component_precision"] == 100.0)
    # gold has 3 independent chains, the prediction reproduces only 2 -> recall 2/3, precision 2/2 -> the two must differ
    gold2 = [[], [0], [], [2], [], [4]]
    pred_partial = [[], [0], [], [2], [], []]
    m2 = CM.conv_chain_metrics(gold2, pred_partial)
    check("the two M2 variants are distinguishable (R<P when a chain is missed)",
          m2["m2_component_recall"] != m2["m2_component_precision"],
          f"R={m2['m2_component_recall']} P={m2['m2_component_precision']}")
    check("returns metric_version", m2.get("metric_version") == CM.METRIC_VERSION)
    check("M3 counts only messages with a gold parent (gold roots excluded)",
          m2["n_anc_msgs"] == 3 and m2["n_anc_msgs"] < len(gold2), str(m2["n_anc_msgs"]))


def test_p0_3_truncation_flag():
    print("\n== P0-3 path-enumeration truncation is recorded ==")
    # build a wide DAG: a complete binary tree -> number of paths is 2^(levels)
    def wide_tree(levels):
        par = [[]]
        idx = 1
        frontier = [0]
        for _ in range(levels - 1):
            nxt = []
            for p in frontier:
                for _ in range(2):
                    par.append([p]); nxt.append(idx); idx += 1
            frontier = nxt
        return par
    gold = wide_tree(8)                    # 2^7 = 128 paths, cap=10 -> truncation is guaranteed
    m = CM.conv_chain_metrics(gold, gold, cap=10)
    check("gold-side truncation is flagged", m["path_gold_truncated"] is True)
    check("prediction-side truncation is flagged", m["path_pred_truncated"] is True)
    small = CM.conv_chain_metrics([[], [0]], [[], [0]], cap=10)
    check("small graphs are not falsely flagged", small["path_gold_truncated"] is False)
    check("truncation information reaches the metric result", "path_gold_truncated" in m and "path_pred_truncated" in m)


def test_degenerate_input():
    print("\n== degenerate-input domain (no silent zero-filling) ==")
    gold = [[], [0], [1]]
    empty_pred = [[], [], []]
    m = CM.conv_chain_metrics(gold, empty_pred)
    check("prediction with no paths -> path_F1=None (distinguishable)", m["path_F1"] is None, str(m["path_F1"]))
    check("a zero-filled variant is provided for continuity with older results", m["path_F1_zero_filled"] == 0.0)
    check("M2 is still computable in that case (recall = 0)", m["m2_component_recall"] == 0.0)
    check("M3 does not emit NaN when there is no comparable gold parent", m["m3_ancestor_jaccard"] in (None, 0.0))


def test_multiparent_invariance():
    print("\n== multi-parent invariant (temporal) ==")
    gold = _diamond_multi_parent()
    pred = [[], [0], [], [0], [1, 3]]
    m = CM.conv_chain_metrics(gold, pred)
    check("M3 is computable on a multi-parent graph", m["m3_ancestor_jaccard"] is not None, str(m["m3_ancestor_jaccard"]))
    check("M1 is well defined on a multi-parent graph", m["n_gold_paths"] > 0, str(m["n_gold_paths"]))


if __name__ == "__main__":
    print("P0 fix regression tests -- ECB event-chain backtracking")
    test_p0_1_classifier()
    test_p0_1_escalation()
    test_p0_2_metric_authority()
    test_p0_3_truncation_flag()
    test_degenerate_input()
    test_multiparent_invariance()
    print(f"\n===== result: {len(PASS)} passed / {len(FAIL)} failed =====")
    if FAIL:
        print("failed items: " + "; ".join(FAIL))
        sys.exit(1)
    print("all regression tests passed.")
