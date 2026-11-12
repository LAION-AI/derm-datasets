"""Visual-only lesion descriptions for SynthDerm prompts. The lifecycle taxonomy (89_lesion_lifecycle.py) mixes visible
morphology with non-visual clinical facts (Auspitz sign, pruritus, course, histology, drug correlation) and strong colour
words ("black"); FLUX.2-klein renders such text literally (pilot 2026-10-14: painted-on, pitch-black or drawn-line lesions).
Per variant x stage, the Splash LLM rewrites the entry into what a camera would see on Fitzpatrick I-II / III-IV / V-VI
skin: primary lesion, number, size, shape, border, surface, colour relative to the surrounding skin, distribution.
Realistic contrast, no non-visual facts, no diagnosis names. Cached in dataset/synthetic/plan/visual_descriptions.json
(key "<diagnosis>|<variant>|<stage>"); 94_lifecycle_prompts.py uses them when present. LLM draft, expert review pending.
"""
import json
import os
import re
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from common import DATASET

PLAN = DATASET / "synthetic/plan"
OUT = PLAN / "visual_descriptions.json"
BASE = os.environ.get("LLM_BASE_URL", "http://localhost:4242/v1")
MODEL = os.environ.get("LLM_MODEL", "incoai/Qwen3.8-27B-Splash")
GROUPS = ["fst_I_II", "fst_III_IV", "fst_V_VI"]
SCHEMA = {"type": "object", "properties": {g: {"type": "string"} for g in GROUPS}, "required": GROUPS}
RULES = (
    "Rewrite this dermatology entry into three short VISUAL descriptions (35-60 words each) of what a clinical photograph "
    "shows, one each for Fitzpatrick I-II, III-IV and V-VI skin. Describe only what a camera sees: primary lesion type, "
    "number, size in mm or cm, shape, border, surface (scale, crust, vesicles, texture), colour RELATIVE to the surrounding "
    "skin, and arrangement / distribution. Rules: no diagnosis or variant names; no symptoms, signs elicited by touch or "
    "manoeuvres, course, timing, histology, causes or tests; realistic and moderate contrast - on dark skin say 'darker brown', "
    "'grey-brown', 'violaceous' or 'ashy' rather than 'black' unless the lesion is genuinely black (e.g. some melanomas, eschar); "
    "no exaggeration; lesions sit in the skin with normal skin texture around them. If the stage has no visible lesion, describe "
    "the subtle residual change (e.g. faint hyperpigmented macules) honestly.")


def ask(key, dx, va, st):
    txt = (f"{RULES}\n\nDiagnosis: {dx}\nVariant: {va}\nStage: {st['stage']}\nMorphology: {st['morphology']}\n"
           f"On Fitzpatrick I-II: {st['fst_I_II']}\nOn Fitzpatrick III-IV: {st['fst_III_IV']}\nOn Fitzpatrick V-VI: {st['fst_V_VI']}")
    body = {"model": MODEL, "reasoning_effort": "low", "temperature": 0, "max_tokens": 2500,
            "messages": [{"role": "user", "content": txt}],
            "response_format": {"type": "json_schema", "json_schema": {"name": "visual", "schema": SCHEMA}}}
    req = urllib.request.Request(BASE + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    err = None
    for _ in range(3):
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                c = json.load(r)["choices"][0]["message"]["content"]
            v = json.loads(re.sub(r"<think>.*?</think>", "", c or "", flags=re.S).strip())
            if all(len(v.get(g, "")) > 40 for g in GROUPS):
                return key, {**v, "model": MODEL, "status": "llm_draft - dermatologist review pending"}
            err = "short output"
        except (OSError, ValueError, KeyError, TypeError) as e:
            err = repr(e)[:200]
    return key, {"error": err}


def main():
    L = {k: v for k, v in json.load(open(PLAN / "lesion_lifecycle.json")).items() if v.get("variants")}
    cache = json.load(open(OUT)) if OUT.exists() else {}
    cache = {k: v for k, v in cache.items() if "error" not in v}
    todo = [(f"{e['requested_name']}|{va['name']}|{st['stage']}", e["requested_name"], va["name"], st)
            for e in L.values() for va in e["variants"] for st in va["stages"]]
    todo = [t for t in todo if t[0] not in cache]
    print(len(todo), "variant-stage entries to rewrite", flush=True)
    with ThreadPoolExecutor(2) as ex:
        for i, (k, v) in enumerate(ex.map(lambda t: ask(*t), todo)):
            cache[k] = v
            if (i + 1) % 25 == 0:
                print(i + 1, k, flush=True); json.dump(cache, open(OUT, "w"), indent=1, ensure_ascii=False)
    json.dump(cache, open(OUT, "w"), indent=1, ensure_ascii=False)
    print("errors:", sum("error" in v for v in cache.values()))


if __name__ == "__main__":
    main()
