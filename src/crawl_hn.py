#!/usr/bin/env python3
"""Crawl Hacker News comment trees via Algolia API (no auth).

Output: data/candidates/hn_threads.jsonl
Schema per line: {"dataset":"hn","conv_id":..., "title":..., "msgs":[...]}
Each msg: {id, author, parent, ts, tlen}  (ts epoch seconds, tlen = text char len)
"""
import json, time, urllib.request, urllib.parse, sys, os

BASE = "https://hn.algolia.com/api/v1"
OUT = os.path.join(os.path.dirname(__file__), "..", "data", "candidates", "hn_threads.jsonl")
N_STORIES = int(sys.argv[1]) if len(sys.argv) > 1 else 120
MIN_POINTS = 800

def get(url, retries=3):
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "research-crawler/0.1"})
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read().decode())
        except Exception as e:
            time.sleep(2 * (i + 1))
    return None

def search_stories():
    stories = []
    page = 0
    while len(stories) < N_STORIES and page < 20:
        q = urllib.parse.urlencode({
            "tags": "story",
            "numericFilters": f"points>{MIN_POINTS}",
            "hitsPerPage": 50, "page": page})
        d = get(f"{BASE}/search?{q}")
        if not d or not d.get("hits"):
            break
        for h in d["hits"]:
            if h.get("author") and h.get("created_at_i") and h.get("points", 0) >= MIN_POINTS:
                stories.append(h)
        page += 1
        time.sleep(0.3)
    return stories[:N_STORIES]

def walk(node, out, depth=0):
    out.append({
        "id": node.get("id"),
        "author": node.get("author"),
        "parent": node.get("parent_id"),
        "story_id": node.get("story_id"),
        "ts": node.get("created_at_i"),
        "tlen": len(node.get("text") or ""),
        "depth": depth,
    })
    for c in node.get("children") or []:
        walk(c, out, depth + 1)

def main():
    stories = search_stories()
    print(f"stories: {len(stories)}", flush=True)
    n_msg = 0
    with open(OUT, "w") as f:
        for i, s in enumerate(stories):
            d = get(f"{BASE}/items/{s['objectID']}")
            if not d:
                continue
            msgs = []
            walk(d, msgs)
            # story itself as root
            root = {"id": int(s["objectID"]), "author": s["author"], "parent": None,
                    "story_id": int(s["objectID"]), "ts": s["created_at_i"],
                    "tlen": len(s.get("title") or "") + len(s.get("story_text") or ""), "depth": 0}
            msgs = [root] + [m for m in msgs if m["id"] != root["id"]]
            f.write(json.dumps({"dataset": "hn", "conv_id": int(s["objectID"]),
                                "title": s.get("title"), "msgs": msgs}, ensure_ascii=False) + "\n")
            n_msg += len(msgs)
            if (i + 1) % 20 == 0:
                print(f"  {i+1}/{len(stories)} stories, {n_msg} msgs", flush=True)
            time.sleep(0.25)
    print(f"DONE hn: {len(stories)} convs, {n_msg} msgs -> {OUT}", flush=True)

if __name__ == "__main__":
    main()
