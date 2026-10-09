"""M07 LATS (2023/ICLR2024) -- language-agent tree search: chain-level joint inference (the key control method).

Style: chain (genuine full-chain joint inference, not per-hop independence followed by chaining).
Mechanism:
  1) draw one LLM policy suggestion s_j per node (the LLM acts as the policy);
  2) sample K candidate chains backwards from the query node i (adopt the suggestion with probability q, otherwise explore randomly);
  3) score each candidate chain with an LLM-endorsement value function and take the argmax as the final chain.

Key point: once the correct chain is sampled its endorsement score is far higher than that of a wrong chain, so
      joint search can beat the p^L ceiling of independent chaining.
"""
from .base import Method
from .helpers import cands_full


class M07LATS(Method):
    name = "M07_lats"
    style = "chain"
    paper = "2023-ICLR2024-LATS.pdf"
    dataset_hint = "D5_agent_traj"

    # In a real API run K directly determines the call volume (~ K x chain length per node), so it is kept small
    # to control cost; the local mock validation used K=16 (no cost pressure). Keep this in mind when comparing runs.
    K = 3
    Q = 0.85    # probability of adopting the LLM suggestion (the rest is exploration)

    def predict(self, inst, llm, rng):
        # 1) policy suggestions: one LLM call per node (the LLM as policy)
        sugg = {0: -1}
        for i in range(1, inst.n):
            sugg[i] = llm.ask_parent(inst, i, cands_full(inst, i))
        # 2) sample K candidate chains, and 3) score them with an independent re-check value function and take the argmax
        pred = {}
        for i in range(1, inst.n):
            best_chain, best_score = None, -1.0
            for _ in range(self.K):
                chain, cur = [i], i
                while len(chain) <= inst.n:
                    c = [x for x in cands_full(inst, cur) if x not in chain]
                    if not c:
                        break
                    if rng.random() < self.Q and sugg.get(cur, -2) in c:
                        nxt = sugg[cur]
                    else:
                        nxt = rng.choice(c)
                    if nxt == -1:
                        chain.append(-1)
                        break
                    chain.append(nxt)
                    cur = nxt
                sc = self._value(inst, chain, llm)
                if sc > best_score:
                    best_score, best_chain = sc, chain
            # Crucial: write back the edges of the **entire** jointly searched chain (not only its first edge),
            # otherwise the advantage of joint chain-level inference is lost, since only the first edge is stored and the rest rely on each node's independent decision.
            if best_chain:
                for a in range(len(best_chain)):
                    child = best_chain[a]
                    if child == -1:
                        continue
                    parent = best_chain[a + 1] if a + 1 < len(best_chain) else -1
                    pred[child] = parent
        return pred

    @staticmethod
    def _value(inst, chain, llm):
        """Value function: **independently re-check** every edge of the chain (a fresh LLM call) and take the mean endorsement rate.

        Crucial: it must be normalized (a mean), otherwise an unnormalized count favours longer wrong chains;
        and the re-check must be an **independent call** -- reusing the suggestions from chain construction degenerates to per-hop chaining, with no joint gain.
        """
        tot = cnt = 0
        for a in range(len(chain)):
            child = chain[a]
            if child == -1:
                continue
            parent = chain[a + 1] if a + 1 < len(chain) else -1
            cands = cands_full(inst, child)
            if parent not in cands:
                cnt += 1
                continue
            v = llm.ask_parent(inst, child, cands)
            tot += 1 if v == parent else 0
            cnt += 1
        return tot / cnt if cnt else 0.0
