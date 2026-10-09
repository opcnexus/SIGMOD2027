"""Real LLM backend (enabled on a Linux GPU server).

Any OpenAI-compatible endpoint is supported (vLLM / LMDeploy / OpenAI / a local Ollama /v1):
    export OPENAI_BASE_URL=http://localhost:8000/v1
    export OPENAI_API_KEY=EMPTY
    export REPRO_LLM_MODEL=Qwen2.5-7B-Instruct
then set llm.backend to "api" in config/config.json.

This module is not used on a local machine (macOS, no GPU, no credential); mock.py covers local validation.
"""
import json
import os
import re
import time
import urllib.request
from pathlib import Path

from .base import BaseLLM


ENV_FILE = Path(__file__).resolve().parents[1] / "config" / ".env.local"


def _load_env_file():
    """Load the credential from config/.env.local (environment variables take precedence).

    Purpose: keep the credential out of command lines and logs. That file should be chmod 600 and never committed.
    """
    if not ENV_FILE.exists():
        return
    for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip()
        if k and not os.environ.get(k):
            os.environ[k] = v


class APILLM(BaseLLM):
    """DeepSeek / OpenAI-compatible endpoint.

    Credential precedence: environment variables OPENAI_API_KEY / DEEPSEEK_API_KEY, then config/.env.local
    (never hard-code the credential in code or logs).
    """

    name = "api"

    def __init__(self, model=None, temperature=0.0, timeout=60, max_retries=3):
        _load_env_file()
        self.base_url = os.environ.get("OPENAI_BASE_URL", "https://api.deepseek.com")
        self.api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("DEEPSEEK_API_KEY", "")
        self.model = model or os.environ.get("REPRO_LLM_MODEL", "deepseek-chat")
        self.temperature = temperature
        self.timeout = timeout
        self.max_retries = max_retries
        self.n_calls = 0
        self.n_fail = 0

    def _chat(self, prompt):
        url = self.base_url.rstrip("/") + "/chat/completions"
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": 16,
        }
        last = None
        for attempt in range(self.max_retries):
            try:
                req = urllib.request.Request(
                    url,
                    data=json.dumps(payload).encode("utf-8"),
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": "Bearer " + self.api_key,
                    },
                )
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    resp = json.loads(r.read().decode("utf-8"))
                return resp["choices"][0]["message"]["content"]
            except Exception as e:  # noqa: BLE001
                last = e
                time.sleep(2 * (attempt + 1))
        self.n_fail += 1
        raise RuntimeError("API failed after %d retries: %s" % (self.max_retries, last))

    def _prompt(self, inst, i, cands):
        L = [
            "Below is a sequence of items (utterances / events / actions) recorded in time order.",
            "Several independent threads or topics are INTERLEAVED in this sequence.",
            "",
            "Task: for item [%d], identify its DIRECT PREDECESSOR within the SAME thread --" % i,
            "that is, the single earlier item that item [%d] directly continues, replies to," % i,
            "or follows from. Use topical continuity and natural stage order as evidence.",
            "If item [%d] starts a NEW thread (no earlier item belongs to its thread), answer -1." % i,
            "",
            "Sequence:",
        ]
        for it in inst.items:
            L.append("[%d] %s" % (it.idx, it.text))
        L.append("")
        L.append("Candidates (choose exactly one): " + ", ".join(str(c) for c in cands))
        L.append("Answer with ONLY the integer id. No explanation.")
        return "\n".join(L)

    def ask_parent(self, inst, i, cands):
        self.n_calls += 1
        if not cands:
            return -1
        try:
            out = self._chat(self._prompt(inst, i, cands))
            m = re.search(r"-?\d+", out)
            if m:
                v = int(m.group())
                if v in cands:
                    return v
        except Exception as e:  # on network/parse failure fall back to the most recent candidate so the pipeline is not interrupted
            print("[APILLM WARN] %s" % e)
        return cands[-1]

    def describe(self):
        return "api(model=%s, url=%s, calls=%d)" % (self.model, self.base_url, self.n_calls)
