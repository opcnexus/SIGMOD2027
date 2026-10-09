#!/usr/bin/env python3
"""Bilibili comment bulk collection (throttled to avoid rate limiting).

Video sources: multiple popular pages + per-channel rankings. For each video, fetch 2 pages of hot comments + 1 page of nested replies for the top 3 hot comments.
Output: data/candidates/bilibili_threads.jsonl (appended; dedup is handled downstream)
"""
import json, time, urllib.request, os

UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
      "Referer": "https://www.bilibili.com"}
OUT = os.path.join(os.path.dirname(__file__), "..", "data", "candidates", "bilibili_threads.jsonl")
OUT_TMP = OUT + ".scale"
TARGET_VIDEOS = 700

def get(url, retries=3):
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=20) as r:
                return json.loads(r.read().decode())
        except Exception:
            time.sleep(2.0 * (i + 1))
    return None

def collect_aids():
    aids = []
    # multiple popular pages
    for pn in range(1, 16):
        d = get(f"https://api.bilibili.com/x/web-interface/popular?ps=20&pn={pn}")
        if not d or d.get("code") != 0:
            break
        aids += [str(v["aid"]) for v in d["data"]["list"]]
        time.sleep(1.0)
    # per-channel rankings
    for rid in (1, 4, 36, 160, 119, 129, 155, 5, 181, 177):
        d = get(f"https://api.bilibili.com/x/web-interface/ranking/v2?rid={rid}&type=all")
        if not d or d.get("code") != 0:
            continue
        for v in d["data"]["list"]:
            aids.append(str(v["aid"]))
        time.sleep(1.0)
    seen, out = set(), []
    for a in aids:
        if a not in seen:
            seen.add(a)
            out.append(a)
    return out

def fetch_video(oid):
    msgs = {}
    for pn in range(1, 3):
        d = get(f"https://api.bilibili.com/x/v2/reply?type=1&oid={oid}&pn={pn}&ps=20&sort=1")
        if not d or d.get("code") != 0 or not d["data"]["replies"]:
            break
        for r in d["data"]["replies"]:
            msgs[r["rpid"]] = {"id": r["rpid"], "author": str(r["member"]["mid"]),
                               "parent": None, "ts": r["ctime"],
                               "tlen": len(r["content"]["message"] or "")}
        time.sleep(1.3)
    # nested replies: only for the top 3 hot comments
    d = get(f"https://api.bilibili.com/x/v2/reply?type=1&oid={oid}&pn=1&ps=20&sort=1")
    if d and d.get("code") == 0:
        for r in (d["data"]["replies"] or [])[:3]:
            if r.get("rcount", 0) > 0:
                d2 = get(f"https://api.bilibili.com/x/v2/reply/reply?type=1&oid={oid}&root={r['rpid']}&pn=1&ps=20")
                if d2 and d2.get("code") == 0:
                    for sub in (d2["data"].get("replies") or []):
                        msgs[sub["rpid"]] = {"id": sub["rpid"], "author": str(sub["member"]["mid"]),
                                             "parent": r["rpid"], "ts": sub["ctime"],
                                             "tlen": len(sub["content"]["message"] or "")}
                time.sleep(1.3)
    return list(msgs.values())

def main():
    done_aids = set()
    if os.path.exists(OUT_TMP):
        for line in open(OUT_TMP):
            try:
                done_aids.add(json.loads(line)["conv_id"])
            except Exception:
                pass
    aids = [a for a in collect_aids() if a not in done_aids]
    print(f"videos to crawl: {len(aids)} (done {len(done_aids)})", flush=True)
    n_msg = 0
    with open(OUT_TMP, "a") as f:
        for i, aid in enumerate(aids):
            msgs = fetch_video(aid)
            if len(msgs) >= 15:
                f.write(json.dumps({"dataset": "bilibili", "conv_id": aid, "title": "",
                                    "msgs": msgs}, ensure_ascii=False) + "\n")
                f.flush()
                n_msg += len(msgs)
            if (i + 1) % 20 == 0:
                print(f"  {i+1}/{len(aids)} videos, +{n_msg} msgs", flush=True)
    print(f"DONE bilibili scale: +{n_msg} msgs -> {OUT_TMP}", flush=True)

if __name__ == "__main__":
    main()
