"""Multi-disease LoRA training set for FLUX.2-klein-base-4B from the permissive reference pool (CC BY / CC0, no text,
no watermark; 79 / 82). Per-disease sets of 30-50 images failed (blur, leakage); this uses every clean lesion photograph
whose shorter side is >= --min-side px (journal figures are small: median 274 px), trained at 384 px with bucketing.

1. vision LLM (OpenAI-compatible endpoint) per image, given the article label and source caption: the diagnosis must be named by the
   label or caption (never diagnosed freely; "unknown" -> dropped), plus variant / stage, body site, age group, sex,
   Fitzpatrick type and a one-sentence visual morphology. Resumable cache: work/lora_data/multi/_descriptions.json.
2. Open-tier safety for the training data: intimate / anogenital / breast / buttock sites (94_lifecycle_prompts.site_ok
   on the reported site) and images flagged by the CLIP nudity screen (86) are excluded.
3. Captions use the generation-prompt format of 94 (subject with skin colour in words + "Diagnosis: <dx>, <variant>,
   stage: <stage>. <morphology>"), so the LoRA's conditioning matches how SynthDerm prompts at inference.
4. Writes images (long side <= 1024 px, JPEG) + .txt + attribution.jsonl to work/lora_data/multi/<name>/ and rsyncs the
   set to donv:~/unsloth-studio/assets/datasets/uploads/<name> (Unsloth Studio data_dir "uploads/<name>").
Usage: python 101_lora_multi.py [--min-side 384] [--workers 2] [--no-describe] [--name synthderm_multi_v1]
"""
import argparse
import base64
import importlib
import io
import json
import os
import re
import subprocess
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from PIL import Image

from common import WORK

BASE = os.environ.get("LLM_BASE_URL", "http://localhost:4242/v1")
MODEL = os.environ.get("LLM_MODEL", "incoai/Qwen3.8-27B-Splash")
OUT = WORK / "lora_data/multi"
ROMAN_SKIN = {"I": "very fair, pale white skin", "II": "fair white skin", "III": "light olive to light brown skin",
              "IV": "moderate brown skin", "V": "dark brown skin", "VI": "very dark brown, almost black, deeply pigmented skin"}
SCHEMA = {"type": "object", "properties": {
    "diagnosis": {"type": "string"}, "diagnosis_source": {"type": "string", "enum": ["label", "caption", "unknown"]},
    "variant": {"type": "string"}, "stage": {"type": "string"}, "body_site": {"type": "string"},
    "age_group": {"type": "string", "enum": ["infant", "child", "adolescent", "adult", "elderly", "unknown"]},
    "sex": {"type": "string", "enum": ["female", "male", "unknown"]},
    "fitzpatrick": {"type": "string", "enum": ["I", "II", "III", "IV", "V", "VI", "unknown"]},
    "is_clinical_skin_photo": {"type": "boolean"}, "morphology": {"type": "string"}},
    "required": ["diagnosis", "diagnosis_source", "variant", "stage", "body_site", "age_group", "sex", "fitzpatrick",
                 "is_clinical_skin_photo", "morphology"]}


def img_url(path):
    im = Image.open(path).convert("RGB"); im.thumbnail((768, 768))
    b = io.BytesIO(); im.save(b, "JPEG", quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode()


def describe(path, label, caption):
    text = (f"Clinical photograph from an open-access dermatology article. Article diagnosis label: '{label}'. Source caption: "
            f"'{str(caption)[:700]}'.\nName the diagnosis ONLY if the label or the caption states it (diagnosis_source = label "
            "or caption); if neither names a specific skin condition, answer diagnosis 'unknown' and diagnosis_source "
            "'unknown'. Never diagnose from the image alone. Then describe what is visible: clinical variant / subtype and "
            "stage (e.g. 'acute, oozing', 'chronic, lichenified', 'treated, post-inflammatory hyperpigmentation'), the precise "
            "body site, age group, sex if visible, Fitzpatrick type of the skin, whether this is a clinical photograph of skin "
            "(false for histology, dermoscopy, radiology, diagrams, intraoperative or mucosal images), and one sentence of "
            "visual morphology (lesion type, number, size, colour relative to the surrounding skin, surface, arrangement).")
    body = {"model": MODEL, "reasoning_effort": "low", "temperature": 0, "max_tokens": 3000,
            "messages": [{"role": "user", "content": [{"type": "text", "text": text},
                                                      {"type": "image_url", "image_url": {"url": img_url(path)}}]}],
            "response_format": {"type": "json_schema", "json_schema": {"name": "ref", "schema": SCHEMA}}}
    req = urllib.request.Request(BASE + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    err = None
    for _ in range(3):
        try:
            with urllib.request.urlopen(req, timeout=900) as r:
                txt = json.load(r)["choices"][0]["message"]["content"]
            return json.loads(re.sub(r"<think>.*?</think>", "", txt or "", flags=re.S).strip())
        except (OSError, ValueError, TypeError) as e:
            err = e
    return {"error": repr(err)[:200]}


def caption_of(d):
    sk = d["fitzpatrick"]
    skin = f"{ROMAN_SKIN[sk]} (Fitzpatrick skin type {sk})" if sk in ROMAN_SKIN else "skin of unknown type"
    who = {"infant": "an infant", "child": "a child", "adolescent": "an adolescent", "adult": "an adult",
           "elderly": "an elderly person", "unknown": "a patient"}[d["age_group"]]
    if d["sex"] != "unknown" and d["age_group"] in ("adult", "elderly"):
        who = {"female": "a woman", "male": "a man"}[d["sex"]] if d["age_group"] == "adult" else f"an elderly {'woman' if d['sex'] == 'female' else 'man'}"
    return (f"Clinical dermatology photograph. Close-up of the {d['body_site']} of {who} with {skin}. "
            f"Diagnosis: {d['diagnosis']}, {d['variant']}, stage: {d['stage']}. {d['morphology']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-side", type=int, default=384)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--no-describe", action="store_true")
    ap.add_argument("--name", default="synthderm_multi_v1")
    a = ap.parse_args()
    R = pd.read_parquet(WORK / "refs/permissive_refs.parquet")
    U = R[R.usable & R.lesion_photo & ~R.has_text.fillna(True) & ~R.has_watermark.fillna(True)].copy()
    U["short"] = [min(Image.open(p).size) for p in U.path]
    U = U[U.short >= a.min_side].reset_index(drop=True)
    print(len(U), f"clean lesion photos with shorter side >= {a.min_side} px", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    cf = OUT / "_descriptions.json"
    cache = json.load(open(cf)) if cf.exists() else {}
    cache = {k: v for k, v in cache.items() if "error" not in v}
    todo = [] if a.no_describe else [r for r in U.itertuples() if r.sha256 not in cache]
    print(len(todo), "to describe", flush=True)
    with ThreadPoolExecutor(a.workers) as ex:
        for i, (k, v) in enumerate(ex.map(lambda r: (r.sha256, describe(r.path, r.diagnosis, r.caption)), todo)):
            cache[k] = v
            if (i + 1) % 25 == 0:
                print(i + 1, "of", len(todo), flush=True); json.dump(cache, open(cf, "w"), indent=1, ensure_ascii=False)
    json.dump(cache, open(cf, "w"), indent=1, ensure_ascii=False)

    sys.path.insert(0, str(Path(__file__).parent))
    m94 = importlib.import_module("94_lifecycle_prompts")
    D = U[U.sha256.isin(cache)].copy()
    D["d"] = D.sha256.map(cache)
    D = D[D.d.map(lambda d: "error" not in d)]
    keep = D.d.map(lambda d: d["diagnosis_source"] != "unknown" and d["diagnosis"].lower() != "unknown"
                   and d["is_clinical_skin_photo"] and m94.site_ok(d["body_site"]))
    print("described", len(D), "| kept after diagnosis / photo / site filters", int(keep.sum()))
    D = D[keep]
    # skin diseases only: the stated diagnosis must match the planned concept vocabulary (979 names; 94's normaliser)
    import difflib
    tab = pd.read_csv(m94.PLAN / "conditioning_table.csv")
    vocab = {m94.norm(n): n for n in tab.concept_name}
    def to_vocab(dx):
        n = m94.norm(dx)
        if n in vocab: return vocab[n]
        hit = difflib.get_close_matches(n, list(vocab), n=1, cutoff=0.85)
        return vocab[hit[0]] if hit else None
    D["concept_name"] = D.d.map(lambda d: to_vocab(d["diagnosis"]))
    print("matched to skin-disease vocabulary:", int(D.concept_name.notna().sum()), "of", len(D))
    D = D[D.concept_name.notna()]
    # CLIP nudity screen (same labels and threshold as the gate)
    g86 = importlib.import_module("99_batch_review")
    pe, _, thr = g86.screen(D.path)
    D = D[pe < thr]
    print("after nudity screen", len(D))
    ds = OUT / a.name; ds.mkdir(exist_ok=True)
    att = []
    for r in D.itertuples():
        im = Image.open(r.path).convert("RGB"); im.thumbnail((1024, 1024))
        im.save(ds / f"{r.sha256[:16]}.jpg", quality=94)
        (ds / f"{r.sha256[:16]}.txt").write_text(caption_of(r.d))
        att.append({"file": f"{r.sha256[:16]}.jpg", "attribution": r.attribution, "license": r.license_class,
                    "source": r.doi or r.pmcid, "modification": "resized, re-encoded"})
    with open(ds / "attribution.jsonl", "w") as f:
        for x in att: f.write(json.dumps(x, ensure_ascii=False) + "\n")
    stats = {"images": len(D), "diagnoses": int(D.concept_name.nunique()),
             "fitzpatrick": D.d.map(lambda d: d["fitzpatrick"]).value_counts().to_dict(),
             "top_diagnoses": D.concept_name.value_counts().head(25).to_dict()}
    json.dump(stats, open(ds / "_stats.json", "w"), indent=1)
    print(json.dumps(stats, indent=1))
    subprocess.run(["rsync", "-a", "--delete", f"{ds}/", f"donv:unsloth-studio/assets/datasets/uploads/{a.name}/"], check=True)
    print("synced to donv uploads/", a.name)


if __name__ == "__main__":
    main()
