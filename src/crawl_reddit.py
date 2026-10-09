#!/usr/bin/env python3
"""Reddit collection (arctic-shift API, no credentials needed).

Strategy: 10 high-engagement subreddits x ~40 top posts each, pulling all the comments of every post.
Output: data/candidates/reddit_threads.jsonl (unified schema, same as crawl_hn.py)
Total cap ~120k messages.
"""
import json, time, urllib.request, sys, os

BASE = "https://arctic-shift.photon-reddit.com/api"
OUT = os.path.join(os.path.dirname(__file__), "..", "data", "candidates", "reddit_threads.jsonl")
SUBS = sys.argv[1].split(",") if len(sys.argv) > 1 else [
    "AskReddit", "worldnews", "technology", "gaming", "movies",
    "relationships", "explainlikeimfive", "Seattle", "politics", "Music"]
APPEND = len(sys.argv) > 2 and sys.argv[2] == "append"
POSTS_PER_SUB = 40
MAX_TOTAL = 120_000

def get(url, retries=4):
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 research/0.1"})
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read().decode())
        except Exception:
            time.sleep(2.5 * (i + 1))
    return None

def fetch_posts(sub):
    posts, before = [], None
    while len(posts) < POSTS_PER_SUB:
        q = f"{BASE}/posts/search?subreddit={sub}&limit=100&sort=desc&sort_type=created_utc"
        if before:
            q += f"&before={before}"
        d = get(q)
        if not d or not d.get("data"):
            break
        rows = d["data"]
        for r in rows:
            if r.get("num_comments", 0) >= 20:
                posts.append(r)
        before = rows[-1]["created_utc"] - 1
        if len(rows) < 100:
            break
        time.sleep(0.4)
    return posts[:POSTS_PER_SUB]

def fetch_comments(link_id, max_c=1500):
    comments, before = [], None
    while len(comments) < max_c:
        q = f"{BASE}/comments/search?link_id={link_id}&limit=100&sort=asc"
        if before:
            q += f"&before={before}"
        d = get(q)
        if not d or not d.get("data"):
            break
        rows = d["data"]
        for r in rows:
            pid = r.get("parent_id", "")
            comments.append({"id": r["id"], "author": r.get("author"),
                             "parent": link_id if pid.startswith("t3_") else pid[3:],
                             "ts": r.get("created_utc"),
                             "tlen": len(r.get("body") or "")})
        before = rows[-1]["created_utc"] - 1
        if len(rows) < 100:
            break
        time.sleep(0.35)
    return comments

def main():
    n_total = 0
    with open(OUT, "a" if APPEND else "w") as f:
        for sub in SUBS:
            posts = fetch_posts(sub)
            print(f"[{sub}] posts: {len(posts)}", flush=True)
            for p in posts:
                if n_total >= MAX_TOTAL:
                    break
                cs = fetch_comments(p["id"])
                root = {"id": p["id"], "author": p.get("author"), "parent": None,
                        "ts": p.get("created_utc"),
                        "tlen": len(p.get("title") or "") + len(p.get("selftext") or "")}
                msgs = [root] + cs
                ids = {m["id"] for m in msgs}
                for m in msgs:
                    if m["parent"] and m["parent"] not in ids:
                        m["parent"] = None
                f.write(json.dumps({"dataset": "reddit", "conv_id": p["id"],
                                    "title": p.get("title", "")[:80], "msgs": msgs},
                                   ensure_ascii=False) + "\n")
                n_total += len(msgs)
                time.sleep(0.3)
            if n_total >= MAX_TOTAL:
                break
    print(f"DONE reddit: {n_total} msgs -> {OUT}", flush=True)

if __name__ == "__main__":
    main()
