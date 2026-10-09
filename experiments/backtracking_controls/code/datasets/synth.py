"""Small-batch synthetic data generator (all five datasets share an "interleaved chains" structure with known gold).

v2 change (essential for real-LLM runs): the text became a **semantically coherent** topic/event sequence --
  each chain is one topic (e.g. "flight to Paris") advancing through natural stages (announce -> start -> finish -> ...),
  and multiple chains are interleaved in time, so a real LLM can infer from topic membership and stage order;
  the semantically meaningless v1 text made real models degenerate to "pick the most recent message" and produced no meaningful result.

Structure:
  1. each instance holds n_chains chains, linear within a chain (D1/D2/D3/D5) or a tree (D4 cascade);
  2. chains are interleaved in time, which creates the real difficulty of finding the correct upstream among several candidates;
  3. each chain carries a topic, and topic noise is injected into 10% of the nodes (added ambiguity);
  4. the 1-to-1 gold is the in-chain predecessor; the full chain is the root->i path (transitive closure of the 1-to-1 gold).

NOTE: these are local small-batch validation data, not research-conclusion data. Real data is fetched on the server by datasets/fetch_scripts/.
"""
import random

from .base import ChainInstance, Item

# topics (6); each chain is bound to one topic
TOPICS = {
    "alpha": "flight to Paris",
    "beta": "pizza order",
    "gamma": "laptop repair",
    "delta": "student election",
    "epsilon": "database migration",
    "zeta": "weekend weather",
}

# staged templates per domain: node k uses template k (cycled modulo the list)
DOMAINS = {
    "D1_conversation": {
        "branching": 1,
        "tpl": [
            "Hi, I have a question about the {topic}.",
            "Sure - what specifically do you need for the {topic}?",
            "I'd like to confirm the details of the {topic} first.",
            "Got it. The {topic} is scheduled for next week.",
            "Can you double-check the {topic} status?",
            "Yes, the {topic} has been updated.",
            "Great, then we are done with the {topic}.",
            "Thanks for your help with the {topic}.",
        ],
    },
    "D2_eventstoryline": {
        "branching": 1,
        "tpl": [
            "Officials announced plans for the {topic}.",
            "Preparations for the {topic} began.",
            "The first phase of the {topic} was completed.",
            "A review of the {topic} was published.",
            "The {topic} entered its final stage.",
            "Results concerning the {topic} were released.",
            "An audit of the {topic} followed.",
            "The {topic} concluded.",
        ],
    },
    "D3_provenance": {
        "branching": 1,
        "tpl": [
            "process opened the config file for {topic}",
            "process read credentials related to {topic}",
            "process wrote a temp file for {topic}",
            "process connected to the host serving {topic}",
            "process executed the loader for {topic}",
            "process modified the registry entry for {topic}",
            "process uploaded data from {topic}",
            "process closed the handle for {topic}",
        ],
    },
    "D4_cascade": {
        "branching": 2,
        "tpl": [
            "user shared a post about {topic}",
            "user reposted an update on {topic}",
            "user commented on {topic}",
            "user reshared the {topic} thread",
            "user quoted the {topic} post",
            "user replied about {topic}",
            "user forwarded news on {topic}",
            "user amplified the {topic} story",
        ],
    },
    "D5_agent_traj": {
        "branching": 1,
        "tpl": [
            "act: inspect the {topic} target",
            "act: move toward the {topic} area",
            "act: pick up the {topic} item",
            "act: open the {topic} container",
            "act: place the {topic} item inside",
            "act: verify the {topic} state",
            "act: clean the {topic} surface",
            "act: finish the {topic} task",
        ],
    },
}


def _build_chain(rng, cid, theme, n_nodes, branching):
    """Return the in-chain node list: [{k, parent(local index), theme}], with parent=-1 marking the root."""
    nodes = [{"k": 0, "parent": -1, "theme": theme}]
    if branching == 1:
        for i in range(1, n_nodes):
            nodes.append({"k": i, "parent": i - 1, "theme": theme})
    else:  # tree (cascade): each child is attached to an existing node, guaranteeing parent_idx < child_idx
        while len(nodes) < n_nodes:
            p = rng.randrange(len(nodes))
            nodes.append({"k": len(nodes), "parent": p, "theme": theme})
    return nodes


def _text(domain, rng, topic, k):
    tpl = DOMAINS[domain]["tpl"]
    return tpl[k % len(tpl)].format(topic=topic)


def gen_instance(domain, iid, rng, n_chains, len_range):
    cfg = DOMAINS[domain]
    chains = []
    for c in range(n_chains):
        theme = list(TOPICS.keys())[c % len(TOPICS)]
        n_nodes = rng.randint(len_range[0], len_range[1])
        chains.append(_build_chain(rng, c, theme, n_nodes, cfg["branching"]))

    # interleave: repeatedly take the next node from a randomly chosen chain that still has nodes left, preserving in-chain order (ancestors before descendants)
    ptr = [0] * len(chains)
    remaining = [i for i in range(len(chains)) if ptr[i] < len(chains[i])]
    global_nodes = []
    while remaining:
        c = rng.choice(remaining)
        global_nodes.append((c, ptr[c], chains[c][ptr[c]]))
        ptr[c] += 1
        remaining = [i for i in range(len(chains)) if ptr[i] < len(chains[i])]

    loc2g = {(c, pos): g for g, (c, pos, _n) in enumerate(global_nodes)}
    items, gold_parent = [], {}
    for g, (c, pos, node) in enumerate(global_nodes):
        true_theme = node["theme"]
        # 10% topic noise: the topic in the text is swapped for another one (adds ambiguity, avoids a trivial task)
        # note: the noise rate directly determines the accuracy ceiling (a noisy node's topic contradicts its hidden chain, so
        #     no method can recover its true parent from the text), so it should not be set too high.
        shown_theme = true_theme
        if rng.random() < 0.1:
            shown_theme = rng.choice([t for t in TOPICS if t != true_theme])
        items.append(
            Item(
                idx=g,
                text=_text(domain, rng, TOPICS[shown_theme], node["k"]),
                ts=float(g),
                meta={"chain": c, "true_theme": true_theme, "shown_theme": shown_theme},
            )
        )
        p_local = node["parent"]
        gold_parent[g] = loc2g[(c, p_local)] if p_local != -1 else -1
    return ChainInstance(dataset=domain, instance_id=iid, items=items, gold_parent=gold_parent)


def gen_dataset(domain, n_inst, n_chains, len_range, seed):
    rng = random.Random(seed)
    return [
        gen_instance(domain, "%s#%d" % (domain, i), rng, n_chains, len_range).to_dict()
        for i in range(n_inst)
    ]
