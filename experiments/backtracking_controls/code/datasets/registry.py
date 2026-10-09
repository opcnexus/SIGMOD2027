"""Dataset registry: two channels -- synthetic (local small batch) and real (fetched on the server)."""
from . import synth

# Dataset metadata (D1-D5 of reports/datasets_5_gold.md)
REGISTRY = {
    "D1_conversation": {
        "scenario": "S1 conversation disentanglement",
        "gold": "human-annotated reply-to links (real data); for synthetic data the gold is derived from the chain structure",
        "full_data": "Kummerfeld et al. 2018 large-scale IRC disentanglement corpus",
        "fetch": "datasets/fetch_scripts/fetch_d1.sh",
        "methods": "all 22 papers in 01_text_1to1 + M03 + M04",
    },
    "D2_eventstoryline": {
        "scenario": "S5 event chains",
        "gold": "event coreference chains plus temporal/causal relations (real data); for synthetic data the gold is derived from the chain structure",
        "full_data": "Event Storyline Corpus (ESC); alternatives: HiEve / MATRES",
        "fetch": "datasets/fetch_scripts/fetch_d2.sh",
        "methods": "M01 / M02 / M03",
    },
    "D3_provenance": {
        "scenario": "S3 provenance forensics",
        "gold": "red-team attack-activity ground truth (real data); for synthetic data the gold is derived from the chain structure",
        "full_data": "OpTC (with red-team ground truth), a subset; alternative: DARPA TC",
        "fetch": "datasets/fetch_scripts/fetch_d3.sh",
        "methods": "M08 / M09",
    },
    "D4_cascade": {
        "scenario": "S4 opinion cascades",
        "gold": "true repost parents / a synthetic ground-truth tree",
        "full_data": "Cascade-Inference-Problem synthetic and real data (Zenodo 10.5281/zenodo.13994029, 39.8GB, not downloaded in full)",
        "fetch": "datasets/fetch_scripts/fetch_d4.sh",
        "methods": "M10",
    },
    "D5_agent_traj": {
        "scenario": "S10 agent trajectories",
        "gold": "task success/failure plus action trajectories (real data); for synthetic data the gold is derived from the chain structure",
        "full_data": "ALFWorld; alternative: the dialogue-agent trajectories released with BlackBox-Forensics-ConvAgent",
        "fetch": "datasets/fetch_scripts/fetch_d5.sh",
        "methods": "M04 / M05 / M06 / M07",
    },
}


def gen_small(cfg):
    """Generate the small batches of the five datasets according to the config (caching is handled by script 01)."""
    sb = cfg["small_batch"]
    out = {}
    for i, ds in enumerate(cfg["datasets"]):
        out[ds] = synth.gen_dataset(
            ds,
            n_inst=sb["instances_per_dataset"],
            n_chains=sb["chains_per_instance"],
            len_range=sb["chain_len_range"],
            seed=cfg["seed"] + i,
        )
    return out
