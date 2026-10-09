"""M04 BlackBox-Forensics-ConvAgent (2026) -- black-box forensics of a dialogue agent (whole-trajectory reconstruction).

Style: chain. Walk back from the terminal node while maintaining a visited memory to avoid cycles;
      after the walk, run a chain-level consistency check and re-query any suspicious edge (cycle-forming or already visited).
"""
from .base import Method
from .helpers import cands_full


class M04BlackBoxForensics(Method):
    name = "M04_blackbox_forensics"
    style = "chain"
    paper = "2026-arXiv2026-BlackBox-Forensics-ConvAgent.pdf"
    dataset_hint = "D5_agent_traj"

    def predict(self, inst, llm, rng):
        pred = {}
        # reconstruct the whole trajectory backwards from the terminal node, with visited memory (already reconstructed nodes are no longer candidates),
        # and a forensic re-check at each step (two agreeing votes, with a third vote to break a tie).
        for i in range(1, inst.n):
            chain, cur = [i], i
            while True:
                c = [x for x in cands_full(inst, cur) if x not in chain]
                if not c:
                    pred[cur] = -1
                    break
                v1 = llm.ask_parent(inst, cur, c)
                v2 = llm.ask_parent(inst, cur, c)
                p = v1 if v1 == v2 else llm.ask_parent(inst, cur, c)
                pred[cur] = p
                if p == -1 or len(chain) > inst.n:
                    break
                chain.append(p)
                cur = p
        # chain-level validation: re-query any edge that still lies on a cycle
        for i in range(1, inst.n):
            seen, cur, bad = set(), i, False
            while cur != -1 and cur in pred:
                if cur in seen:
                    bad = True
                    break
                seen.add(cur)
                cur = pred[cur]
            if bad:
                pred[i] = llm.ask_parent(inst, i, cands_full(inst, i))
        return pred
