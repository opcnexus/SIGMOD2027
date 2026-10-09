#!/usr/bin/env python3
"""**Zero-dependency core** of the chain-level metrics (standard library only).

Separation goal: reviewers can verify the metric definitions and the regression
tests (`tests/test_p0_fixes.py` depends on this module only) without installing
torch / numpy. The experiment driver and the training-related code live in
`chain_metrics.py`; it reuses the functions of this module so there is a
"single definition".

Metric definition version: see METRIC_VERSION (for the lessons learned see
docs/chain_metric_definitions.md).
"""
from collections import defaultdict

PATH_CAP = 2000
# metric definition version (metrics sharing a name must share a version; ECB and
# the baselines share conv_chain_metrics)
METRIC_VERSION = "ecb-2.0"


def build_forest(parents_of):
    n = len(parents_of)
    children = defaultdict(list)
    roots = []
    for i in range(n):
        ps = [p for p in parents_of[i] if 0 <= p < n and p != i]
        if not ps:
            roots.append(i)
        for p in ps:
            children[p].append(i)
    child_set = set(children)
    leaves = [i for i in range(n) if i not in child_set]
    return children, roots, leaves

def enumerate_paths(children, roots, cap=PATH_CAP):
    """All root->leaf paths of the DAG (iterative DFS). Returns (path list,
    whether truncated)."""
    paths, truncated = [], False
    for r in roots:
        stack = [(r, (r,))]
        while stack:
            node, path = stack.pop()
            cs = children.get(node, [])
            if not cs:
                paths.append(path)
                if len(paths) >= cap:
                    return paths, True
            else:
                for c in cs:
                    stack.append((c, path + (c,)))
    return paths, truncated

def components(parents_of):
    n = len(parents_of)
    par = list(range(n))
    def find(x):
        while par[x] != x:
            par[x] = par[par[x]]; x = par[x]
        return x
    for i, ps in enumerate(parents_of):
        for p in ps:
            if 0 <= p < n:
                ra, rb = find(i), find(p)
                if ra != rb:
                    par[ra] = rb
    groups = defaultdict(set)
    for i in range(n):
        groups[find(i)].add(i)
    return set(frozenset(g) for g in groups.values())

def ancestor_closure(parents_of, i):
    seen = set()
    stack = [p for p in parents_of[i] if 0 <= p < len(parents_of)]
    while stack:
        x = stack.pop()
        if x in seen:
            continue
        seen.add(x)
        stack.extend(p for p in parents_of[x] if 0 <= p < len(parents_of))
    return seen

def conv_chain_metrics(gold_par, pred_par, cap=PATH_CAP):
    """[ECB authoritative definition] Single-conversation chain-level metrics --
    the single implementation, so that metrics sharing a name cannot diverge.

    P0-2 fix: M2 returns both the **recall-type** value (share of gold components
    reproduced) and the **precision-type** value (share of predicted components hit),
    with the denominator made explicit (both include singleton components, matching
    the aggregation of dataset_probe_eval).
    P0-3 fix: M1 explicitly returns the truncation flag of the path enumeration (gold
    and pred independently) instead of silently discarding it.

    Degenerate-input convention: when an empty path set makes the metric
    incomputable, `path_F1 = None` (**never silently filled with 0**); a
    `path_F1_zero_filled` variant is also provided for alignment with older results;
    `m3` is computed only over messages that have a gold parent (a gold root has no
    ancestors by definition, so the closure Jaccard is undefined for it).
    """
    gc, gr, _ = build_forest(gold_par)
    pc, pr, _ = build_forest(pred_par)
    gpaths, g_trunc = enumerate_paths(gc, gr, cap)
    ppaths, p_trunc = enumerate_paths(pc, pr, cap)
    gs, ps = set(gpaths), set(ppaths)
    inter = len(gs & ps)
    a = inter / len(ps) if ps else None          # precision-type (predicted paths as denominator)
    b = inter / len(gs) if gs else None          # recall-type (gold paths as denominator)
    f1 = (2 * a * b / (a + b)) if (a is not None and b is not None and (a + b) > 0) else None
    gcomp, pcomp = components(gold_par), components(pred_par)
    m2r = 100 * len(gcomp & pcomp) / len(gcomp) if gcomp else None
    m2p = 100 * len(gcomp & pcomp) / len(pcomp) if pcomp else None
    anc = []
    for i in range(len(pred_par)):
        if gold_par[i]:                            # exclude gold roots (no ancestor closure)
            ga = ancestor_closure(gold_par, i)
            pa = ancestor_closure(pred_par, i)
            if ga:
                anc.append(len(ga & pa) / len(ga | pa))
    return {
        "metric_version": METRIC_VERSION,
        "path_P": round(100 * a, 2) if a is not None else None,
        "path_R": round(100 * b, 2) if b is not None else None,
        "path_F1": round(100 * f1, 2) if f1 is not None else None,
        "path_F1_zero_filled": round(100 * f1, 2) if f1 is not None else 0.0,
        "n_gold_paths": len(gs), "n_pred_paths": len(ps),
        "path_gold_truncated": bool(g_trunc),     # P0-3: truncation recorded explicitly
        "path_pred_truncated": bool(p_trunc),
        "m2_component_recall": round(m2r, 2) if m2r is not None else None,
        "m2_component_precision": round(m2p, 2) if m2p is not None else None,
        "n_gold_components": len(gcomp), "n_pred_components": len(pcomp),
        "m3_ancestor_jaccard": round(100 * (sum(anc) / len(anc)), 2) if anc else None,
        "n_anc_msgs": len(anc),
    }


def conv_chain_metrics(gold_par, pred_par, cap=PATH_CAP):
    """[ECB authoritative definition] Single-conversation chain-level metrics --
    the single implementation, so that metrics sharing a name cannot diverge.

    P0-2 fix: M2 returns both the **recall-type** value (share of gold components
    reproduced) and the **precision-type** value (share of predicted components hit),
    with the denominator made explicit (both include singleton components, matching
    the aggregation of dataset_probe_eval).
    P0-3 fix: M1 explicitly returns the truncation flag of the path enumeration (gold
    and pred independently) instead of silently discarding it.

    Degenerate-input convention: when an empty path set makes the metric
    incomputable, `path_F1 = None` (**never silently filled with 0**); a
    `path_F1_zero_filled` variant is also provided for alignment with older results;
    `m3` is computed only over messages that have a gold parent (a gold root has no
    ancestors by definition, so the closure Jaccard is undefined for it).
    """
    gc, gr, _ = build_forest(gold_par)
    pc, pr, _ = build_forest(pred_par)
    gpaths, g_trunc = enumerate_paths(gc, gr, cap)
    ppaths, p_trunc = enumerate_paths(pc, pr, cap)
    gs, ps = set(gpaths), set(ppaths)
    inter = len(gs & ps)
    a = inter / len(ps) if ps else None          # precision-type (predicted paths as denominator)
    b = inter / len(gs) if gs else None          # recall-type (gold paths as denominator)
    f1 = (2 * a * b / (a + b)) if (a is not None and b is not None and (a + b) > 0) else None
    gcomp, pcomp = components(gold_par), components(pred_par)
    m2r = 100 * len(gcomp & pcomp) / len(gcomp) if gcomp else None
    m2p = 100 * len(gcomp & pcomp) / len(pcomp) if pcomp else None
    anc = []
    for i in range(len(pred_par)):
        if gold_par[i]:                            # exclude gold roots (no ancestor closure)
            ga = ancestor_closure(gold_par, i)
            pa = ancestor_closure(pred_par, i)
            if ga:
                anc.append(len(ga & pa) / len(ga | pa))
    return {
        "metric_version": METRIC_VERSION,
        "path_P": round(100 * a, 2) if a is not None else None,
        "path_R": round(100 * b, 2) if b is not None else None,
        "path_F1": round(100 * f1, 2) if f1 is not None else None,
        "path_F1_zero_filled": round(100 * f1, 2) if f1 is not None else 0.0,
        "n_gold_paths": len(gs), "n_pred_paths": len(ps),
        "path_gold_truncated": bool(g_trunc),     # P0-3: truncation recorded explicitly
        "path_pred_truncated": bool(p_trunc),
        "m2_component_recall": round(m2r, 2) if m2r is not None else None,
        "m2_component_precision": round(m2p, 2) if m2p is not None else None,
        "n_gold_components": len(gcomp), "n_pred_components": len(pcomp),
        "m3_ancestor_jaccard": round(100 * (sum(anc) / len(anc)), 2) if anc else None,
        "n_anc_msgs": len(anc),
    }

