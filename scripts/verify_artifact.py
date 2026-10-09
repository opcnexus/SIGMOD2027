#!/usr/bin/env python3
"""Artifact integrity checks (the first thing a reviewer should run).

Usage: python scripts/verify_artifact.py
Checks:
  1. all required files are present
  2. no leftover secrets / usernames / absolute paths (double-blind compliance)
  3. the redacted sample contains no plaintext usernames or message text
  4. key result files parse and the metric version is consistent
  5. the metric implementation unit tests pass
A non-zero exit code indicates a problem.
"""
import json
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
FAIL = []


def ok(cond, msg, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {msg}" + (f"  {detail}" if detail else ""))
    if not cond:
        FAIL.append(msg)


REQUIRED = [
    "README.md", "LICENSE", "CITATION.cff", "DATA_STATEMENT.md", "requirements.txt",
    "Makefile", "ARTIFACT_MANIFEST.md",
    "src/representation_space.py", "src/slack_silver.py", "src/chain_metrics.py",
    "src/llm_response.py", "src/ecb_full.py", "src/transfer_experiment.py",
    "src/full_benchmark.py", "src/moe_router.py", "src/ocrapt_style.py",
    "tests/test_p0_fixes.py", "scripts/make_paper_tables.py",
    "results/raw_reduced/representation_space.json", "results/raw_reduced/transfer_experiment.json",
    "results/raw_reduced/backtrack_reconstruction.json", "results/raw_reduced/ecb_full.jsonl",
    "results/full_benchmark_v2/summary.json", "data_sample/slack_mention_proxy_silver.sample.jsonl",
]

# only flag realistic key shapes (>=20 chars, no placeholder); the sk-... in the docs is illustrative
SECRET_PATTERNS = [
    (r"sk-[A-Za-z0-9]{20,}", "suspected real API key"),
    (r"(?i)api[_-]?key\s*[=:]\s*[\"'][A-Za-z0-9]{20,}[\"']", "hard-coded credential"),
    (r"huakanglee", "username"),
    (r"/Users/[A-Za-z0-9_.-]+/", "local absolute path"),
    (r"workbuddy", "internal tool path"),
]
SELF = pathlib.Path(__file__).resolve()

TEXT_LEAK = re.compile(r'"text":\s*"(?!<REDACTED)[^"]{20,}')


def main():
    print("== 1. required files ==")
    for f in REQUIRED:
        ok((ROOT / f).exists(), f"present: {f}")

    print("\n== 2. secret / identity scan ==")
    hits = 0
    for p in ROOT.rglob("*"):
        if not p.is_file() or p.suffix.lower() in {".png", ".pdf", ".xlsx", ".npy", ".pt"}:
            continue
        if p.resolve() == SELF:          # this scanner contains the pattern literals itself; skip it
            continue
        if ".git" in p.parts:            # .git is never transferred by push (the reflog is local only)
            continue
        try:
            txt = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for pat, why in SECRET_PATTERNS:
            for m in re.finditer(pat, txt):
                hits += 1
                print(f"      -> {p.relative_to(ROOT)}: {why}: {m.group()[:40]}")
    ok(hits == 0, "no leftover secrets / usernames / absolute paths", f"{hits} hits")

    print("\n== 2b. commit metadata (the identity channel that push actually transfers) ==")
    try:
        import subprocess
        who = subprocess.run(["git", "log", "--format=%an|%ae|%cn|%ce"],
                             cwd=ROOT, capture_output=True, text=True, timeout=30).stdout
        ids = {x for line in who.strip().splitlines() if line for x in line.split("|")}
        leak = {i for i in ids if "huakanglee" in i or i in ("",)}
        ok(not leak, f"commit author/committer identities are anonymized ({len(ids)} distinct)", f"leaked: {sorted(leak)}")
        objs = subprocess.run(["git", "rev-list", "--all", "--objects"],
                              cwd=ROOT, capture_output=True, text=True, timeout=30).stdout
        ok("logs/" not in objs, "no reflog objects queued for push (.git/logs)")
    except Exception as e:
        ok(False, "commit metadata check", f"{type(e).__name__}: {e}")

    print("\n== 3. redacted sample ==")
    s = ROOT / "data_sample" / "slack_mention_proxy_silver.sample.jsonl"
    if s.exists():
        txt = s.read_text(encoding="utf-8")
        ok(TEXT_LEAK.search(txt) is None, "no plaintext message text in the sample")
        ok(re.search(r'"author":\s*"u_[0-9a-f]{8}"', txt) is not None, "authors are pseudonymized")
        n = sum(1 for l in txt.splitlines() if l.strip())
        ok(n > 0, f"sample is non-empty ({n} records)")
        try:
            json.loads(txt.splitlines()[0])
            ok(True, "first sample line parses")
        except Exception as e:
            ok(False, "first sample line parses", str(e))

    print("\n== 4. key results parse and metric version ==")
    for f in ["results/raw_reduced/representation_space.json", "results/raw_reduced/transfer_experiment.json",
              "results/raw_reduced/backtrack_reconstruction.json", "results/full_benchmark_v2/summary.json"]:
        p = ROOT / f
        try:
            json.load(open(p))
            ok(True, f"parses: {f}")
        except Exception as e:
            ok(False, f"parses: {f}", str(e))
    ecb = ROOT / "results/raw_reduced/ecb_full.jsonl"
    if ecb.exists():
        rows = [json.loads(l) for l in ecb.open() if l.strip()]
        summ = [r for r in rows if r.get("window_id") == -1]
        vers = {r.get("metric_version") for r in summ}
        ok(len(summ) == 101, f"ECB conversation-level summary rows = {len(summ)} (paper reports 101)")
        ok(vers == {"ecb-2.0"}, f"metric version is consistent: {vers}")

    print("\n== 5. syntax compilation of every Python file ==")
    import py_compile
    bad = []
    pys = [q for q in ROOT.rglob("*.py") if "__pycache__" not in q.parts]
    for q in pys:
        try:
            py_compile.compile(str(q), doraise=True, cfile="/tmp/_repspace_check.pyc")
        except Exception as e:
            bad.append((q, str(e).splitlines()[0][:80]))
    ok(not bad, f"all {len(pys)} .py files compile", f"{len(bad)} failed")
    for q, e in bad:
        print(f"      -> {q.relative_to(ROOT)}: {e}")

    print("\n== 6. metric implementation unit tests ==")
    r = subprocess.run([sys.executable, str(ROOT / "tests" / "test_p0_fixes.py")],
                       capture_output=True, text=True)
    tail = [l for l in r.stdout.strip().splitlines() if "=====" in l]
    ok(r.returncode == 0, "tests/test_p0_fixes.py passes", tail[-1] if tail else "")

    print("\n===== verification complete =====")
    if FAIL:
        print(f"found {len(FAIL)} problem(s):")
        for f in FAIL:
            print("  -", f)
        sys.exit(1)
    print("all checks passed. Run `make tables` to rebuild the paper tables and compare them with the published values.")


if __name__ == "__main__":
    main()
