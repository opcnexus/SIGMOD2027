#!/usr/bin/env python3
"""Crawl Bilibili comment sections (reply trees) via public API (no auth).

Output: data/candidates/bilibili_threads.jsonl
conv = one video's comment section. Top-level comments are roots (parent=None);
sub-replies have parent=rpid. ts in epoch seconds.
"""
import json, time, urllib.request, sys, os

UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
      "Referer": "https://www.bilibili.com"}
OUT = os.path.join(os.path.dirname(__file__), "..", "data", "candidates", "bilibili_threads.jsonl")
N_VIDEOS = int(sys.argv[1]) if len(sys.argv) > 1 else 40

def get(url, retries=4):
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=20) as r:
                return json.loads(r.read().decode())
        except Exception:
            pass
        time.sleep(2.0 * (i + 1))
    return None

def collect_aids():
    aids = []
    for pn in range(1, 6):
        d = get(f"https://api.bilibili.com/x/web-interface/popular?ps=20&pn={pn}")
        if not d or d.get("code") != 0:
            break
        for v in d["data"]["list"]:
            aids.append((str(v["aid"]), v.get("title", "")))
        time.sleep(0.4)
    return aids[:N_VIDEOS]

def fetch_comments(oid, max_top=160, max_sub=40):
    msgs = {}
    for pn in range(1, 9):
        d = get(f"https://api.bilibili.com/x/v2/reply?type=1&oid={oid}&pn={pn}&ps=20&sort=1")
        if not d or d.get("code") != 0 or not d["data"]["replies"]:
            break
        for r in d["data"]["replies"]:
            root_rpid = r["rpid"]
            msgs[r["rpid"]] = {"id": r["rpid"], "author": str(r["member"]["mid"]),
                               "parent": None, "ts": r["ctime"],
                               "tlen": len(r["content"]["message"] or "")}
            reps = r.get("replies") or []
            for sub in reps:
                msgs[sub["rpid"]] = {"id": sub["rpid"], "author": str(sub["member"]["mid"]),
                                     "parent": root_rpid, "ts": sub["ctime"],
                                     "tlen": len(sub["content"]["message"] or "")}
            # fetch more sub-replies if truncated
            if r.get("rcount", 0) > len(reps) and len(reps) >= 3:
                d2 = get(f"https://api.bilibili.com/x/v2/reply/reply?type=1&oid={oid}&root={root_rpid}&pn=1&ps=20")
                if d2 and d2.get("code") == 0:
                    for sub in (d2["data"].get("replies") or [])[:max_sub]:
                        msgs[sub["rpid"]] = {"id": sub["rpid"], "author": str(sub["member"]["mid"]),
                                             "parent": root_rpid, "ts": sub["ctime"],
                                             "tlen": len(sub["content"]["message"] or "")}
        if len(msgs) >= max_top * 3:
            break
        time.sleep(0.35)
    return list(msgs.values())

def main():
    aids = collect_aids()
    print(f"videos: {len(aids)}", flush=True)
    n_msg = 0
    with open(OUT, "w") as f:
        for i, (aid, title) in enumerate(aids):
            msgs = fetch_comments(aid)
            if len(msgs) < 20:
                continue
            f.write(json.dumps({"dataset": "bilibili", "conv_id": aid, "title": title,
                                "msgs": msgs}, ensure_ascii=False) + "\n")
            n_msg += len(msgs)
            if (i + 1) % 10 == 0:
                print(f"  {i+1}/{len(aids)} videos, {n_msg} msgs", flush=True)
            time.sleep(0.4)
    print(f"DONE bilibili: {n_msg} msgs -> {OUT}", flush=True)

if __name__ == "__main__":
    main()
