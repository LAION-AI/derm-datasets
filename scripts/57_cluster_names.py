"""Name the HDBSCAN concept clusters of 56_concept_map.py with an LLM (OpenAI-compatible chat endpoint).

Endpoint: LLM_BASE_URL (default http://localhost:11434/v1 = Ollama, e.g. via `ssh -N -L 11434:localhost:11434 donv`),
model: LLM_MODEL (default gemma2:latest), optional LLM_API_KEY, LLM_MAX_TOKENS (default 4096, retried once at 8192; room for
<think> blocks of reasoning models, which are stripped), LLM_REASONING_EFFORT (e.g. low).
Used: LLM_BASE_URL=http://localhost:4242/v1 LLM_MODEL=incoai/Qwen3.8-27B-Splash LLM_REASONING_EFFORT=low. The 25 most frequent concept names of each cluster
are sent; nothing else. Temperature 0; names are cached in work/concepts/cluster_names.json (delete to redo).
"""
import json
import os
import re
import urllib.request

from common import WORK

BASE = os.environ.get("LLM_BASE_URL", "http://localhost:11434/v1").rstrip("/")
MODEL = os.environ.get("LLM_MODEL", "gemma2:latest")
PROMPT = ("Below are dermatology terms that form one cluster in a map of diagnosis concepts (most frequent first). "
          "Give a short, specific name for the cluster (2-5 words, English, Title Case), e.g. 'Autoimmune Bullous "
          "Diseases' or 'Fungal Skin Infections'. If the terms are mostly descriptive (morphology, body sites, "
          "imaging methods), name that instead. Answer with the name only.\n\nTerms: ")


def ask(terms, max_tokens=None):
    max_tokens = max_tokens or int(os.environ.get("LLM_MAX_TOKENS", 4096))
    req = {"model": MODEL, "temperature": 0, "max_tokens": max_tokens,
           "messages": [{"role": "user", "content": PROMPT + "; ".join(terms)}]}
    if os.environ.get("LLM_REASONING_EFFORT"):   # e.g. "low" for Qwen3.8-27B-Splash (otherwise it can think > 8k tokens)
        req["reasoning_effort"] = os.environ["LLM_REASONING_EFFORT"]
    body = json.dumps(req).encode()
    req = urllib.request.Request(BASE + "/chat/completions", data=body, headers={
        "Content-Type": "application/json", **({"Authorization": "Bearer " + os.environ["LLM_API_KEY"]}
                                               if os.environ.get("LLM_API_KEY") else {})})
    with urllib.request.urlopen(req, timeout=900) as r:
        txt = json.load(r)["choices"][0]["message"]["content"]
    if not txt:   # reasoning used the whole budget (finish_reason "length"): retry once with twice the budget
        if max_tokens >= 8192:
            raise RuntimeError("no answer within the token budget")
        return ask(terms, 2 * max_tokens)
    txt = re.sub(r"<think>.*?</think>", "", txt, flags=re.S)   # reasoning models (e.g. Qwen3) may think first
    txt = re.sub(r"[*\"'`]|^name:\s*", "", txt.strip().split("\n")[0], flags=re.I).strip(" .")
    return txt[:60]


def main():
    M = json.load(open(WORK / "concepts/concept_map.json"))
    cache = WORK / "concepts/cluster_names.json"
    names = json.load(open(cache)) if cache.exists() else {}
    for c in M["clusters"]:
        k = str(c["c"])
        if k not in names:
            names[k] = {"name": ask(c["top"]), "model": MODEL, "reasoning_effort": os.environ.get("LLM_REASONING_EFFORT")}
            print(k, names[k]["name"], "|", ", ".join(c["top"][:5]), flush=True)
            json.dump(names, open(cache, "w"), indent=1, ensure_ascii=False)
    print(len(names), "clusters named with", MODEL)


if __name__ == "__main__":
    main()
