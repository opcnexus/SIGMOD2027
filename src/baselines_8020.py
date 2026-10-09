"""Recompute the heuristic baselines (B1/B2/B3/B4/B5) under the 80/20 re-split protocol; the results are stored in results/baselines_8020.json"""

import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from baselines import (LastMessage, SameSpeaker, Mention, TfidfBaseline,
                       CombinedLinear, build_tfidf, evaluate_split)
from ff_reproduction import DATA_ROOT
from digat import load_split_paths, load_logs_for

RESULTS_DIR = "./results"


def main():
    split = load_split_paths()
    logs = load_logs_for(split["test"])
    print("test files:", len(logs))
    tfidf = build_tfidf(logs)
    out = {}
    for name, pred in (("B1-last", LastMessage()), ("B2-speaker", SameSpeaker()),
                       ("B3-mention", Mention()), ("B4-tfidf", TfidfBaseline(tfidf)),
                       ("B5-combined", CombinedLinear(tfidf))):
        out[name] = {"test": evaluate_split(logs, pred)}
        print(name, out[name]["test"], flush=True)
    with open(os.path.join(RESULTS_DIR, "baselines_8020.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("saved")


if __name__ == "__main__":
    main()
