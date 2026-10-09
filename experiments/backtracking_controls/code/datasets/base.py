"""Unified data model: all five datasets are expressed as "chain instances".

Unified abstraction (matching the paper formalization m_i -> m_{i-j}):
  - items        : time-ordered nodes (messages / events / entities / reposts / actions)
  - gold_parent  : 1-to-1 gold edges (each node -> its direct upstream node; root = -1)
  - gold_chain(i): full-chain gold (the complete upstream path from the root to i), derived as the transitive closure of the 1-to-1 gold edges
"""
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class Item:
    idx: int
    text: str
    ts: float
    meta: Dict = field(default_factory=dict)


@dataclass
class ChainInstance:
    dataset: str
    instance_id: str
    items: List[Item]
    gold_parent: Dict[int, int]  # idx -> parent idx (root = -1)

    # ---------- full-chain gold (transitive closure of the 1-to-1 gold) ----------
    def gold_chain(self, i: int) -> List[int]:
        """The complete upstream path root -> ... -> i."""
        chain, seen = [], set()
        cur = i
        while cur != -1 and cur not in seen and cur in self.gold_parent:
            seen.add(cur)
            chain.append(cur)
            cur = self.gold_parent[cur]
        return list(reversed(chain))

    def chain_len(self, i: int) -> int:
        return len(self.gold_chain(i))

    @property
    def n(self) -> int:
        return len(self.items)

    def non_root(self) -> List[int]:
        return [i for i in range(self.n) if self.gold_parent.get(i, -1) != -1]

    def to_dict(self):
        return {
            "dataset": self.dataset,
            "instance_id": self.instance_id,
            "items": [
                {"idx": it.idx, "text": it.text, "ts": it.ts, "meta": it.meta}
                for it in self.items
            ],
            "gold_parent": {str(k): v for k, v in self.gold_parent.items()},
        }


def from_dict(d) -> ChainInstance:
    items = [
        Item(idx=it["idx"], text=it["text"], ts=it["ts"], meta=it.get("meta", {}))
        for it in d["items"]
    ]
    gp = {int(k): v for k, v in d["gold_parent"].items()}
    return ChainInstance(
        dataset=d["dataset"], instance_id=d["instance_id"], items=items, gold_parent=gp
    )
