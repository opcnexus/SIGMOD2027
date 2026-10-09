#!/usr/bin/env python3
"""Download exactly the files needed by the irc-disentanglement dataset (skipping the 85MB auto-annotated bz2)"""
import json
import os
import sys
import time
import urllib.request

REPO = "jkkummerfeld/irc-disentanglement"
BRANCH = "master"
RAW = "https://raw.githubusercontent.com/{}/{}/".format(REPO, BRANCH)
DEST = "./data/irc-disentanglement"

def fetch(url, retries=4):
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "research-dl/1.0"})
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read()
        except Exception as e:
            print("  retry {}/{}: {}".format(i + 1, retries, e), file=sys.stderr)
            time.sleep(2 * (i + 1))
    return None

def main():
    # 1. fetch the file tree
    tree_url = "https://api.github.com/repos/{}/git/trees/{}?recursive=1".format(REPO, BRANCH)
    blob = fetch(tree_url)
    if blob is None:
        sys.exit("FATAL: cannot fetch file tree")
    tree = json.loads(blob).get("tree", [])

    # 2. filter the required files
    keep = []
    for t in tree:
        if t.get("type") != "blob":
            continue
        p = t["path"]
        ok = (
            p in ("README.md", "LICENSE.md", "data/README.md", "data/LICENSE-data.txt")
            or p.startswith("data/train/") and (p.endswith(".tok.txt") or p.endswith(".annotation.txt"))
            or p.startswith("data/dev/") and (p.endswith(".tok.txt") or p.endswith(".annotation.txt"))
            or p.startswith("data/test/") and (p.endswith(".tok.txt") or p.endswith(".annotation.txt"))
            or p.startswith("data/gold.")
            or p.startswith("data/list.ubuntu")
        )
        if ok:
            keep.append(p)

    total = len(keep)
    print("files to download: {}".format(total))
    fail = []
    for i, p in enumerate(keep, 1):
        out = os.path.join(DEST, p)
        if os.path.exists(out) and os.path.getsize(out) > 0:
            continue
        os.makedirs(os.path.dirname(out), exist_ok=True)
        data = fetch(RAW + p)
        if data is None:
            fail.append(p)
            continue
        with open(out, "wb") as f:
            f.write(data)
        if i % 20 == 0:
            print("[{}/{}] ...".format(i, total))
    print("done. failed: {}".format(len(fail)))
    for p in fail:
        print("  FAIL:", p)

if __name__ == "__main__":
    main()
