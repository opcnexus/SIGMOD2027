#!/usr/bin/env python3
"""Re-crawl Reddit (including the comment body text): arctic-shift API.
Output: data/candidates/reddit_threads_text.jsonl (msg gains a text field)"""
import json, time, urllib.request, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "data", "candidates", "reddit_threads_text.jsonl")
BASE = "https://arctic-shift.photon-reddit.com/api"
UA = {"User-Agent": "research-script/0.2"}

SUBS = ["AskReddit", "worldnews", "technology", "gaming", "movies",
        "relationships", "explainlikeimfive", "Seattle", "politics", "Music",
        "todayilearned", "nba", "teenagers", "relationship_advice", "AskMen", "funny"]

def get(url, retries=3):
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=25) as r:
                return json.loads(r.read().decode())
        except Exception:
            time.sleep(3 * (i + 1))
    return None

def main():
    n_msg = n_conv = 0
    with open(OUT, "w") as f:
        for sub in SUBS:
            # crawl 40 heavily-commented posts per subreddit (paginating the default sort)
            before = None
            posts = []
            for page in range(4):
                q = f"{BASE}/posts/search?subreddit={sub}&limit=100&sort=desc&sort_type=created_utc"
                if before:
                    q += f"&before={before}"
                d = get(q)
                rows = (d or {}).get("data") or []
                if not rows:
                    break
                posts += rows
                before = rows[-1]["created_utc"]
                time.sleep(1.0)
            # pick the 40 posts with the largest num_comments and >=30
            posts = [p for p in posts if (p.get("num_comments") or 0) >= 30 and not p.get("over_18")]
            posts.sort(key=lambda p: -(p.get("num_comments") or 0))
            for p in posts[:40]:
                pid, plink = p["id"], p.get("permalink", "")
                title = p.get("title") or ""
                selftext = p.get("selftext") or ""
                cs = []
                after = None
                for page in range(3):
                    q = f"{BASE}/comments/search?link_id={pid}&limit=100&sort=asc"
                    if after:
                        q += f"&after={after}"
                    d = get(q)
                    rows = (d or {}).get("data") or []
                    if not rows:
                        break
                    cs += rows
                    after = rows[-1]["created_utc"]
                    time.sleep(0.8)
                msgs = [{"id": pid, "author": p.get("author"), "parent": None,
                         "ts": p.get("created_utc"), "text": (title + " " + selftext)[:400]}]
                for r in cs:
                    par = r.get("parent_id", "")
                    msgs.append({"id": r["id"], "author": r.get("author"),
                                 "parent": (par[3:] if par.startswith("t1_") else None),
                                 "ts": r.get("created_utc"),
                                 "text": (r.get("body") or "")[:400]})
                ids = {m["id"] for m in msgs}
                for m in msgs:
                    if m["parent"] and m["parent"] not in ids:
                        m["parent"] = None  # a dangling parent edge means the root post
                with_parent = [m for m in msgs if m["parent"]]
                if len(with_parent) < 5:
                    continue
                f.write(json.dumps({"dataset": "reddit", "conv_id": pid,
                                    "title": title[:100], "msgs": msgs},
                                   ensure_ascii=False) + "\n")
                n_msg += len(msgs); n_conv += 1
                time.sleep(1.0)
            print(f"  [{sub}] done, cumulative convs={n_conv} msgs={n_msg}", flush=True)
    print(f"DONE reddit_text: {n_conv} convs {n_msg} msgs -> {OUT}", flush=True)

if __name__ == "__main__":
    main()
