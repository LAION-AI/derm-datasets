"""Per-disease LoRA training set from the permissive reference pool (CC BY / CC0; no text annotations; 79 / 82).

1. Candidates: source diagnosis label matches --label, or the caption matches --include (and not --exclude) while the
   label is unspecific ("no definitive diagnosis", dermatitis, eczema ...).
2. Verification + structured caption with the Splash vision LLM (incoai/Qwen3.8-27B-Splash): it receives the image
   AND the source caption / label and must not diagnose freely; it states whether the image is consistent with the
   disease (inconsistent images are dropped) and describes subtype / stage, body site, age group, sex, Fitzpatrick type
   and morphology. Training caption: "<disease>, <stage>, <body site>, <age group> <sex>, Fitzpatrick <type>. <morphology>".
3. Images (optionally Real-ESRGAN x4 upscaled when small; long side <= 1024 px, JPEG; Fitzpatrick V-VI optionally
   oversampled by copies) + <stem>.txt captions (skin colour in words + Fitzpatrick type) + attribution.jsonl (CC BY attribution, modification:
   resized) are written locally and rsynced to the Unsloth Studio computer (LORA_REMOTE/<slug>).
Usage: python 91_lora_dataset.py --disease "atopic dermatitis" --label "atopic" \
         --include "atopic (dermatitis|eczema)|neurodermatitis|\\beczema\\b" --exclude "contact|seborrh|dyshidro|nummular|herpeticum|stasis"
"""
import argparse
import base64
import io
import json
import os
import re
import shutil
import subprocess
import urllib.request
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from PIL import Image

from common import WORK

BASE = os.environ.get("LLM_BASE_URL", "http://localhost:4242/v1")
MODEL = os.environ.get("LLM_MODEL", "incoai/Qwen3.8-27B-Splash")
REMOTE = os.environ.get("LORA_REMOTE", "trainhost:/data/synthderm/lora_data")
SCHEMA = {"type": "object", "properties": {
    "consistent_with_disease": {"type": "boolean"}, "reason": {"type": "string"},
    "subtype_or_stage": {"type": "string"}, "body_site": {"type": "string"},
    "age_group": {"type": "string", "enum": ["infant", "child", "adolescent", "adult", "elderly", "unknown"]},
    "sex": {"type": "string", "enum": ["female", "male", "unknown"]},
    "fitzpatrick": {"type": "string", "enum": ["I", "II", "III", "IV", "V", "VI", "unknown"]},
    "morphology": {"type": "string"}},
    "required": ["consistent_with_disease", "reason", "subtype_or_stage", "body_site", "age_group", "sex", "fitzpatrick", "morphology"]}


SKIN_WORDS = {"I": "very fair, pale skin", "II": "fair skin", "III": "light brown skin", "IV": "moderate brown skin",
              "V": "dark brown skin", "VI": "very dark brown to black skin"}
_SR = {}


def upscale(im):
    """Real-ESRGAN x4 (realesr-general-x4v3, BSD-3; weights loaded with torch.load(weights_only=True) via spandrel)."""
    import numpy as np
    import torch
    from spandrel import ModelLoader
    if not _SR:
        sd = torch.load(WORK.parent / "external/models/realesrgan/realesr-general-x4v3.pth", map_location="cpu", weights_only=True)
        dev = "mps" if torch.backends.mps.is_available() else "cpu"
        _SR.update(m=ModelLoader().load_from_state_dict(sd).to(dev).eval(), dev=dev)
    x = torch.from_numpy(np.asarray(im, dtype=np.float32) / 255).permute(2, 0, 1)[None].to(_SR["dev"])
    with torch.no_grad():
        y = _SR["m"](x).clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()
    return Image.fromarray((y * 255).round().astype("uint8"))


def img_url(im):
    im = im.copy(); im.thumbnail((768, 768))
    b = io.BytesIO(); im.save(b, "JPEG", quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode()


def describe(disease, im, label, caption, lifecycle_hint):
    text = (f"This clinical photograph comes from an open-access article. Article diagnosis label: '{label}'. Source caption: "
            f"'{str(caption)[:600]}'. The training target is the disease '{disease}'.\n"
            f"Known variants / stages of {disease}: {lifecycle_hint}\n"
            "Do not diagnose freely. Decide only whether the image is consistent with the stated disease given the label and "
            "caption (false if the caption names a different condition, if no skin lesion is visible, or if it is histology / "
            "a diagram). Then describe what is visible: subtype or stage (use the known stages where they fit, e.g. 'acute, "
            "oozing' / 'chronic, lichenified'), precise body site, age group, sex if visible, Fitzpatrick type of the skin, and "
            "the morphology in precise dermatological terms (one sentence).")
    body = {"model": MODEL, "reasoning_effort": "low", "temperature": 0, "max_tokens": 3000,
            "messages": [{"role": "user", "content": [{"type": "text", "text": text}, {"type": "image_url", "image_url": {"url": img_url(im)}}]}],
            "response_format": {"type": "json_schema", "json_schema": {"name": "lesion", "schema": SCHEMA}}}
    req = urllib.request.Request(BASE + "/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    for _ in range(3):
        try:
            with urllib.request.urlopen(req, timeout=900) as r:
                txt = json.load(r)["choices"][0]["message"]["content"]
            return json.loads(re.sub(r"<think>.*?</think>", "", txt or "", flags=re.S).strip())
        except (OSError, ValueError, TypeError) as e:
            err = e
    return {"error": repr(err)[:200]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--disease", required=True)
    ap.add_argument("--label", required=True, help="regex on the source diagnosis label")
    ap.add_argument("--include", default="", help="regex on label + caption for unspecific labels")
    ap.add_argument("--exclude", default="^$")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--min-side", type=int, default=192, help="drop images whose shorter side is smaller (journal figures are small)")
    ap.add_argument("--no-describe", action="store_true", help="only use cached descriptions (e.g. while the LLM server is down)")
    ap.add_argument("--upscale", action="store_true", help="Real-ESRGAN x4 (realesr-general-x4v3) for images with shorter side < 640 px")
    ap.add_argument("--oversample-dark", type=int, default=1, help="repeat Fitzpatrick V-VI images this many times")
    ap.add_argument("--crop-margin", type=float, default=0.0, help="crop this fraction from every edge (removes corner panel letters)")
    ap.add_argument("--tag", default="", help="suffix of the output dataset name (e.g. _v2)")
    a = ap.parse_args()
    slug = re.sub(r"[^a-z0-9]+", "_", a.disease.lower()).strip("_")
    out = WORK / f"lora_data/{slug}{a.tag}"
    desc_dir = WORK / f"lora_data/{slug}"   # descriptions are shared between dataset versions
    out.mkdir(parents=True, exist_ok=True)
    R = pd.read_parquet(WORK / "refs/permissive_refs.parquet")
    u = R[R.usable & R.lesion_photo & ~R.has_text].drop_duplicates("sha256")
    lab = u.diagnosis.fillna("")
    txt = (lab + " || " + u.caption.fillna("")).str.lower()
    by_label = lab.str.contains(a.label, case=False, regex=True)
    unspecific = lab.str.contains(r"no definitive|eczema|dermatitis|^$", case=False, regex=True)
    by_caption = (txt.str.contains(a.include, regex=True) if a.include else False) & unspecific
    cand = u[(by_label | by_caption) & ~txt.str.contains(a.exclude, regex=True)]
    print(a.disease, "candidates", len(cand), flush=True)
    life = json.load(open(WORK.parent / "dataset/synthetic/plan/lesion_lifecycle.json")) if (WORK.parent / "dataset/synthetic/plan/lesion_lifecycle.json").exists() else {}
    entry = next((v for k, v in life.items() if k.lower() == a.disease.lower()), {})
    hint = "; ".join(f"{v.get('name')}: " + ", ".join(s.get("stage", "") for s in v.get("stages", [])) for v in entry.get("variants", [])) or "n/a"
    desc_dir.mkdir(parents=True, exist_ok=True)
    cache_f = desc_dir / "_descriptions.json"
    cache = json.load(open(cache_f)) if cache_f.exists() else {}

    def run(r):
        if r.sha256 in cache:
            return r.sha256, cache[r.sha256]
        im = Image.open(r.path); im.draft("RGB", (2048, 2048)); im = im.convert("RGB")
        return r.sha256, describe(a.disease, im, r.diagnosis, r.caption, hint)
    todo = [] if a.no_describe else [r for r in cand.itertuples() if r.sha256 not in cache or "error" in cache[r.sha256]]
    for r in todo:
        cache.pop(r.sha256, None)
    with ThreadPoolExecutor(a.workers) as ex:
        for i, (h, d) in enumerate(ex.map(run, todo)):
            cache[h] = d
            if (i + 1) % 10 == 0:
                json.dump(cache, open(cache_f, "w"), indent=1)
                print(i + 1, "/", len(todo), flush=True)
    json.dump(cache, open(cache_f, "w"), indent=1)

    for f in out.glob("*.jpg"):
        f.unlink()
    for f in out.glob("*.txt"):
        f.unlink()
    att, kept = [], 0
    for r in cand.itertuples():
        d = cache.get(r.sha256, {})
        if not d.get("consistent_with_disease"):
            continue
        im = Image.open(r.path); im.draft("RGB", (2048, 2048)); im = im.convert("RGB")
        if min(im.size) < a.min_side:
            continue
        mod = "resized (long side <= 1024 px)"
        if a.crop_margin > 0:
            w, h = im.size; m = a.crop_margin
            im = im.crop((int(w * m), int(h * m), int(w * (1 - m)), int(h * (1 - m))))
            mod = f"cropped {int(m * 100)}% margins; " + mod
            if min(im.size) < a.min_side:
                continue
        if a.upscale and min(im.size) < 640:
            im = upscale(im); mod = "upscaled x4 with Real-ESRGAN realesr-general-x4v3, then resized (long side 1024 px)"
        im.thumbnail((1024, 1024))
        stem = r.sha256[:16]
        im.save(out / f"{stem}.jpg", quality=93)
        who = " ".join(x for x in [d["age_group"], d["sex"]] if x and x != "unknown")
        fst = f", {SKIN_WORDS[d['fitzpatrick']]} (Fitzpatrick {d['fitzpatrick']})" if d["fitzpatrick"] != "unknown" else ""
        cap = f"{a.disease}, {d['subtype_or_stage']}, {d['body_site']}" + (f", {who}" if who else "") + f"{fst}. {d['morphology']}"
        (out / f"{stem}.txt").write_text(cap.strip())
        att.append({"file": f"{stem}.jpg", "sha256": r.sha256, "source_license": r.license_raw, "attribution": r.attribution,
                    "modification": mod + "; caption written by Qwen3.8-27B-Splash", "caption": cap})
        if d["fitzpatrick"] in ("V", "VI"):
            for k in range(1, a.oversample_dark):   # oversampling of dark skin (no per-image repeat option in the trainer)
                shutil.copy(out / f"{stem}.jpg", out / f"{stem}_r{k}.jpg"); shutil.copy(out / f"{stem}.txt", out / f"{stem}_r{k}.txt")
        kept += 1
    # text check on the final images (Apple Vision OCR): drop images that still show letters / labels
    tool = Path(__file__).parent / "tools/vision_ocr"
    if tool.exists():
        files = sorted(out.glob("*.jpg"))
        res = subprocess.run([str(tool), *map(str, files)], capture_output=True, text=True).stdout.splitlines()
        bad = {Path(json.loads(l)["file"]).name for l in res if any(w.get("conf", 0) >= 0.3 for w in json.loads(l).get("words", []))}
        for b in bad:
            stem = b[:-4]
            (out / b).unlink(missing_ok=True); (out / f"{stem}.txt").unlink(missing_ok=True)
        att = [x for x in att if x["file"] not in bad]
        kept = len({x["sha256"] for x in att})
        print("removed after OCR (text visible):", len(bad), flush=True)
    with open(out / "attribution.jsonl", "w") as f:
        for x in att:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    print(a.disease, "kept", kept, "of", len(cand), flush=True)
    subprocess.run(["rsync", "-a", "--delete", "--exclude", "_descriptions.json", f"{out}/", f"{REMOTE}/{slug}{a.tag}/"], check=True)
    print("synced to", f"{REMOTE}/{slug}{a.tag}/")


if __name__ == "__main__":
    main()
