#!/usr/bin/env python3
"""Bilibili slow-recovery comment collection (no-login, risk-control-aware).

Strategy (lessons learned from the 2026-09-25 risk-control incident):
- random 18-28s gap between requests; sleep 8 minutes every 25 requests
- ps=49 is the per-page maximum; 3 pages of hot comments + 1 page of nested replies for the top 5 (explicit parent)
- keep the comment text (stored as `text` alongside `tlen`, for later feature engineering)
- resumable: read the conv_id values already present in the output file and dedup
Output: data/candidates/bilibili_threads.jsonl.scale (appended)
"""
import json, time, random, urllib.request, os, sys

UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
      "Referer": "https://www.bilibili.com"}
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "data", "candidates", "bilibili_threads.jsonl.scale")
MAX_HOURS = float(sys.argv[1]) if len(sys.argv) > 1 else 10.0

req_count = 0
t_start = time.time()

def get(url, retries=2):
    global req_count
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=20) as r:
                req_count += 1
                # risk-control throttle: sleep 8 minutes every 25 requests
                if req_count % 25 == 0:
                    print(f"  [pause] {req_count} requests, sleep 480s", flush=True)
                    time.sleep(480)
                else:
                    time.sleep(random.uniform(18, 28))
                d = json.loads(r.read().decode())
                # Bilibili risk control returns error codes in the JSON body (-352/-412), not as HTTP status
                if d.get("code") in (-352, -412, -799):
                    print(f"  [risk-control] code={d.get('code')} backing off 900s", flush=True)
                    time.sleep(900)
                    continue
                return d
        except Exception as e:
            if "412" in str(e) or "352" in str(e):
                print(f"  [risk-control] backing off 900s: {e}", flush=True)
                time.sleep(900)
            else:
                time.sleep(5 * (i + 1))
    return None

def collect_aids():
    aids = []
    for pn in range(1, 21):
        d = get(f"https://api.bilibili.com/x/web-interface/popular?ps=20&pn={pn}")
        if not d or d.get("code") != 0:
            break
        aids += [str(v["aid"]) for v in d["data"]["list"]]
    for rid in (1, 4, 36, 160, 119, 129, 155, 5, 181, 177, 231, 21):
        d = get(f"https://api.bilibili.com/x/web-interface/ranking/v2?rid={rid}&type=all")
        if d and d.get("code") == 0:
            aids += [str(v["aid"]) for v in d["data"]["list"]]
    seen, out = set(), []
    for a in aids:
        if a not in seen:
            seen.add(a); out.append(a)
    return out

def fetch_video(oid):
    msgs = {}
    for pn in range(1, 9):
        d = get(f"https://api.bilibili.com/x/v2/reply?type=1&oid={oid}&pn={pn}&ps=20&sort=1")
        if not d or d.get("code") != 0 or not d["data"]["replies"]:
            if d and d.get("code") not in (0,):
                print(f"  [warn] oid={oid} pn={pn} code={d.get('code')} {d.get('message')}", flush=True)
            break
        for r in d["data"]["replies"]:
            msgs[r["rpid"]] = {"id": r["rpid"], "author": str(r["member"]["mid"]),
                               "parent": None, "ts": r["ctime"],
                               "text": (r["content"]["message"] or "")[:200]}
        # nested replies for the top 5 hot comments (explicit parent structure)
        if pn == 1:
            for r in (d["data"]["replies"] or [])[:5]:
                if r.get("rcount", 0) > 0:
                    d2 = get(f"https://api.bilibili.com/x/v2/reply/reply?type=1&oid={oid}&root={r['rpid']}&pn=1&ps=20")
                    if d2 and d2.get("code") == 0:
                        for sub in (d2["data"].get("replies") or []):
                            msgs[sub["rpid"]] = {"id": sub["rpid"], "author": str(sub["member"]["mid"]),
                                                 "parent": r["rpid"], "ts": sub["ctime"],
                                                 "text": (sub["content"]["message"] or "")[:200]}
    return list(msgs.values())

def main():
    done_aids = set()
    if os.path.exists(OUT):
        for line in open(OUT):
            try:
                done_aids.add(json.loads(line)["conv_id"])
            except Exception:
                pass
    aids = [a for a in collect_aids() if a not in done_aids]
    print(f"videos to crawl: {len(aids)} (already done {len(done_aids)})", flush=True)
    n_msg = 0
    with open(OUT, "a") as f:
        for i, aid in enumerate(aids):
            if (time.time() - t_start) > MAX_HOURS * 3600:
                print(f"time budget {MAX_HOURS}h reached, stop at video {i}", flush=True)
                break
            msgs = fetch_video(aid)
            if len(msgs) >= 15:
                f.write(json.dumps({"dataset": "bilibili", "conv_id": aid, "title": "",
                                    "msgs": msgs}, ensure_ascii=False) + "\n")
                f.flush()
                n_msg += len(msgs)
            if (i + 1) % 10 == 0:
                print(f"  {i+1}/{len(aids)} videos, +{n_msg} msgs, {req_count} reqs, "
                      f"elapsed {int((time.time()-t_start)/60)}min", flush=True)
    print(f"DONE bilibili slow: +{n_msg} msgs -> {OUT}", flush=True)

if __name__ == "__main__":
    main()
