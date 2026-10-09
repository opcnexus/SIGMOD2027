"""Method base class and shared utilities.

The two styles (corresponding to the difference to be explained):
  style='1to1'  : only local per-hop link prediction (each hop independent), chained at evaluation time -> errors accumulate as p^L
  style='chain': joint prediction over chains, subgraphs or search -> can break the independence assumption
"""
from datasets.base import ChainInstance


class Method:
    name = "base"
    style = "1to1"  # or 'chain'
    paper = ""
    dataset_hint = ""

    def predict(self, inst: ChainInstance, llm, rng):
        """Return pred_parent: Dict[int, int] (each non-root node -> its predicted direct upstream)."""
        raise NotImplementedError

    def info(self):
        return {"name": self.name, "style": self.style, "paper": self.paper}


def cands_before(inst, i):
    """All nodes that are earlier in time (idx order is time order)."""
    return list(range(i))


def chain_from(inst, pred_parent, i):
    """Chain the predicted 1-to-1 edges into the full chain: root -> ... -> i, with cycle protection.

    Note: the current node must be appended before fetching its parent, otherwise when the root is node 0 
    (0 is not a key of pred_parent) the walk stops early and the EM of every chain rooted there collapses to 0.
    """
    out, seen = [], set()
    cur = i
    while cur != -1 and cur not in seen:
        seen.add(cur)
        out.append(cur)
        cur = pred_parent.get(cur, -1)
    return list(reversed(out))
