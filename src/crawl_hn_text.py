#!/usr/bin/env python3
"""Re-crawl the full HN trees (including the comment text): one items API call per post, parsing the nested children.
Output: data/candidates/hn_threads_text.jsonl (msg gains a text field)"""
import json, time, re, urllib.request, os, html

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "data", "candidates", "hn_threads_text.jsonl")
UA = {"User-Agent": "research-script/0.2"}

TAG_RE = re.compile(r"<[^>]+>")

def strip_html(s):
    if not s:
        return ""
    return " ".join(html.unescape(TAG_RE.sub(" ", s)).split())

def get(url, retries=3):
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=25) as r:
                return json.loads(r.read().decode())
        except Exception:
            time.sleep(3 * (i + 1))
    return None

def walk(node, parent_id, out, story_id):
    """Depth-first collection of (id, author, parent, ts, text, depth)"""
    nid = node.get("id")
    txt = strip_html(node.get("text") or node.get("title") or "")
    out.append({"id": nid, "author": node.get("author"),
                "parent": parent_id, "ts": node.get("created_at_i"),
                "text": txt[:400], "depth": 0})
    for ch in node.get("children") or []:
        walk(ch, nid, out, story_id)

def main():
    # reuse the 120 story ids already crawled
    story_ids = []
    for line in open(os.path.join(HERE, "..", "data", "candidates", "hn_threads.jsonl")):
        d = json.loads(line)
        story_ids.append(d["conv_id"])
    print(f"recrawling {len(story_ids)} stories with text ...", flush=True)
    n_msg = 0
    with open(OUT, "w") as f:
        for k, sid in enumerate(story_ids):
            d = get(f"https://hn.algolia.com/api/v1/items/{sid}")
            if not d:
                print(f"  [skip] {sid}", flush=True)
                continue
            msgs = []
            walk(d, None, msgs, sid)
            # keep only the non-root comments (root = the post itself, parent = None)
            msgs = [m for m in msgs if m["parent"]]
            ids = {m["id"] for m in msgs}
            ids.add(sid)
            for m in msgs:
                if m["parent"] not in ids:
                    m["parent"] = sid if m["parent"] is None else None
                if m["parent"] == sid:
                    m["parent"] = None
            n_valid = sum(1 for m in msgs if m["parent"] in ids or m["parent"] is None)
            f.write(json.dumps({"dataset": "hn", "conv_id": str(sid),
                                "title": strip_html(d.get("title") or ""),
                                "story_id": sid, "msgs": msgs}, ensure_ascii=False) + "\n")
            n_msg += len(msgs)
            if (k + 1) % 10 == 0:
                print(f"  {k+1}/{len(story_ids)} stories, {n_msg} msgs", flush=True)
            time.sleep(1.2)
    print(f"DONE hn_text: {n_msg} msgs -> {OUT}", flush=True)

if __name__ == "__main__":
    main()
