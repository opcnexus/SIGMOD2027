"""Build the intermediate-data artifacts manifest: register retained outputs + filesystem timestamps (minute precision)."""
import os, json, datetime

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MANIFEST = os.path.join(ROOT, "data", "artifacts", "manifest.json")

ITEMS = [
    # (path, type, description, generating script)
    ("data/irc-disentanglement/", "raw_data", "Ubuntu IRC raw data, 519 files + glove-ubuntu.txt", "src/download_data.py"),
    ("results/split_8020.json", "split", "80/20 re-split, seed=42 (117/21/35)", "src/digat.py"),
    ("results/data_space_analysis.json", "analysis", "data solution-space statistics (e.g. 2.67% multi-parent)", "src/data_space_analysis.py"),
    ("results/ff_model_glove.pt", "checkpoint", "FF277 pretrained encoder (official 153/10/10 protocol)", "src/ff_reproduction.py"),
    ("results/ff_predictions_test_glove.json", "predictions", "FF277 test predictions", "src/ff_reproduction.py"),
    ("data/bert_feats.npz", "features", "DistilBERT sentence vectors (248k messages, 768-dim, fp16, 330MB)", "src/bert_encode.py"),
    ("results/baselines.json", "results", "heuristic baselines (official protocol)", "src/baselines.py"),
    ("results/baselines_8020.json", "results", "heuristic baselines (80/20 protocol)", "src/baselines_8020.py"),
    ("results/digat_results_digat_glove.json", "results", "DiGAT(GloVe) 80/20", "src/digat.py"),
    ("results/digat_results_pairmlp.json", "results", "PairMLP ablation, 80/20", "src/digat.py"),
    ("results/digat_results_digat_bert.json", "results", "DiGAT+DistilBERT 80/20", "src/digat.py"),
    ("results/digat_digat_glove.pt", "checkpoint", "DiGAT(GloVe) weights", "src/digat.py"),
    ("results/digat_pairmlp.pt", "checkpoint", "PairMLP weights", "src/digat.py"),
    ("results/digat_digat_bert.pt", "checkpoint", "DiGAT+BERT weights", "src/digat.py"),
    ("results/gts_backtrack_results.json", "results", "GTS-Backtrack v1 (official protocol)", "src/gts_backtrack.py"),
]

def ts(path):
    p = os.path.join(ROOT, path)
    if not os.path.exists(p):
        return None
    return datetime.datetime.fromtimestamp(os.path.getmtime(p)).strftime("%Y-%m-%d %H:%M")

manifest = {
    "updated_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
    "purpose": "Intermediate-data retention index: later experiments read the cache directly instead of reprocessing the raw data",
    "items": [
        {"path": p, "type": t, "desc": d, "source": s, "completed_at": ts(p)}
        for p, t, d, s in ITEMS
    ],
}
with open(MANIFEST, "w") as f:
    json.dump(manifest, f, ensure_ascii=False, indent=2)
print("saved:", MANIFEST)
for it in manifest["items"]:
    print(" ", it["path"], "->", it["completed_at"])
