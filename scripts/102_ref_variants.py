"""Reference-based SynthDerm variants: real, openly licensed lesion photographs (CC BY / CC0 only; no CC BY-SA, so the
derivatives stay permissive) are re-rendered by FLUX.2-klein-4B image editing on other Fitzpatrick skin types. The
lesion's size, shape, number, distribution, surface, body part and framing stay those of the real photograph, so the
morphology is realistic rather than the exaggerated text-to-image look (expert feedback 2026-10-16); only the
constitutive skin colour and the lesion's colour on that skin change. Main purpose: close the Fitzpatrick V-VI gap.

Selection: references described by 101_lora_multi.py (Splash vision; diagnosis named by the article label or caption,
clinical skin photograph), licence cc_by / cc0, open-tier site (94.site_ok), minors only on face / arms / hands
(94.minor_ok), CLIP nudity screen passed. Targets: Fitzpatrick V and VI from Fitzpatrick I-IV sources; pigmentary diagnoses are skipped; a Splash vision
pre-check drops multi-panel figures, privacy masks, overlay graphics and non-dermatological images.
Outputs: <out>/img/<synth_id>.jpg + manifest.parquet with caption, edit prompt, source attribution (CC BY) and the
modification note; ref_cos = CLIP cosine to the source photograph (derivatives are expected to stay close).
Usage: python 102_ref_variants.py --n-refs 60 --out ../work/synthetic/refvar_pilot
"""
import argparse
import base64
import importlib
import io
import json
import re
import sys
import time
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from common import WORK

sys.path.insert(0, str(Path(__file__).parent))
m94 = importlib.import_module("94_lifecycle_prompts")
API, MODEL = "http://127.0.0.1:8188", "flux2-klein-4b"
ROMAN = {1: "I", 2: "II", 3: "III", 4: "IV", 5: "V", 6: "VI"}
RN = {v: k for k, v in ROMAN.items()}
LOOK_DARK = ("On this dark skin, redness and inflammation appear violaceous, dark brown or grey-brown instead of bright "
             "red, scale can look ashy grey, and post-inflammatory hyperpigmentation is common.")
LOOK_LIGHT = "On this light skin, inflammation appears pink to red, and brown or black pigment keeps its colour."
KEEP = ("Keep the lesion's size, shape, number, distribution and surface exactly as in the photograph, and keep the "
        "body part, framing, camera angle and lighting the same. Depigmented (white) or pigmented lesions keep their "
        "character. Realistic clinical photograph, sharp focus, natural skin texture with pores; subtle and true to "
        "life, not exaggerated; no text, no labels, no arrows.")


def targets(src):
    """Darker targets only (pilot 2026-10-20: lightening dark-skinned sources made lesions neon-pink / changed the
    disease, while V-VI variants of light-skinned sources kept realistic morphology)."""
    return [5, 6] if src <= 4 else []


# the pigment IS the disease: no skin-colour shift
PIGMENTARY = re.compile(r"pigment|lentig|melasma|vitiligo|nev|naev|melanom|acanthosis nigricans|ochronos|freckl|ephelid|"
                        r"caf[eé] au lait|mongolian|of ota|of ito|versicolor|leukoderm|albinis|piebald|poikiloderm|"
                        r"argyri|chloasma|tattoo", re.I)
PRECHECK = WORK / "refs/ref_precheck.json"
PC_SCHEMA = {"type": "object", "properties": {
    "single_photograph": {"type": "boolean"}, "privacy_mask": {"type": "boolean"}, "overlay_graphics": {"type": "boolean"},
    "skin_condition_visible": {"type": "boolean"}, "notes": {"type": "string"}},
    "required": ["single_photograph", "privacy_mask", "overlay_graphics", "skin_condition_visible", "notes"]}
LLM = "http://localhost:4242/v1"


def precheck_one(path):
    im = Image.open(path).convert("RGB"); im.thumbnail((768, 768)); b = io.BytesIO(); im.save(b, "JPEG", quality=88)
    q = ("Check this image from a dermatology article. single_photograph: true only if it is ONE photograph (not a "
         "multi-panel figure, collage, grid, or photo plus histology / dermoscopy inset). privacy_mask: true if any black "
         "bar, blur, pixelation or patch hides the eyes, face or another body part. overlay_graphics: true if drawn "
         "graphics are overlaid on the image (circles, arrows, letters, numbers, scale bars, rulers); pen marks drawn "
         "directly on the skin by clinicians do NOT count. skin_condition_visible: true if a skin, hair or nail condition "
         "is visible (false for surgical wounds, orthopaedics, radiology, intra-operative views).")
    body = {"model": "incoai/Qwen3.8-27B-Splash", "reasoning_effort": "low", "temperature": 0, "max_tokens": 2000,
            "messages": [{"role": "user", "content": [{"type": "text", "text": q}, {"type": "image_url", "image_url": {
                "url": "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode()}}]}],
            "response_format": {"type": "json_schema", "json_schema": {"name": "pc", "schema": PC_SCHEMA}}}
    req = urllib.request.Request(LLM + "/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    for _ in range(3):
        try:
            with urllib.request.urlopen(req, timeout=900) as r:
                txt = json.load(r)["choices"][0]["message"]["content"]
            return json.loads(re.sub(r"<think>.*?</think>", "", txt or "", flags=re.S).strip())
        except (OSError, ValueError, TypeError) as e:
            err = e
    return {"error": repr(err)[:200]}


def precheck(C, workers=2):
    cache = json.load(open(PRECHECK)) if PRECHECK.exists() else {}
    cache = {k: v for k, v in cache.items() if "error" not in v}
    todo = [r for r in C.itertuples() if r.ref_sha256 not in cache]
    print(len(todo), "references to pre-check", flush=True)
    with ThreadPoolExecutor(workers) as ex:
        for i, (k, v) in enumerate(ex.map(lambda r: (r.ref_sha256, precheck_one(r.ref_path)), todo)):
            cache[k] = v
            if (i + 1) % 25 == 0:
                print(i + 1, "pre-checked", flush=True); json.dump(cache, open(PRECHECK, "w"), indent=1)
    json.dump(cache, open(PRECHECK, "w"), indent=1)
    ok = C.ref_sha256.map(lambda k: (lambda v: "error" not in v and v["single_photograph"] and not v["privacy_mask"]
                                     and not v["overlay_graphics"] and v["skin_condition_visible"])(cache.get(k, {"error": 1})))
    print("pre-check passed:", int(ok.sum()), "of", len(C), flush=True)
    return C[ok].reset_index(drop=True)


def edit_prompt(target):
    return (f"Change only the patient's natural skin colour to {m94.SKIN[target]} (Fitzpatrick skin type {ROMAN[target]}), "
            f"as in a person of {m94.ANCESTRY[target][0]} descent. " + (LOOK_DARK if target >= 5 else LOOK_LIGHT) + " " + KEEP)


def out_size(path):
    w, h = Image.open(path).size
    s = 1024 / max(w, h)
    return f"{max(512, int(w * s) // 64 * 64)}x{max(512, int(h * s) // 64 * 64)}"


def post_edit(img_path, prompt, size, seed, api=None, model=None):
    api, model = api or API, model or MODEL
    b = uuid.uuid4().hex
    im = Image.open(img_path).convert("RGB"); buf = io.BytesIO(); im.save(buf, "PNG")
    fields = {"model": model, "prompt": prompt, "size": size, "response_format": "b64_json", "seed": str(seed)}
    parts = [f'--{b}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode() for k, v in fields.items()]
    parts.append(f'--{b}\r\nContent-Disposition: form-data; name="image"; filename="ref.png"\r\nContent-Type: image/png\r\n\r\n'.encode()
                 + buf.getvalue() + b"\r\n")
    parts.append(f"--{b}--\r\n".encode())
    req = urllib.request.Request(api + "/v1/images/edits", data=b"".join(parts),
                                 headers={"Content-Type": f"multipart/form-data; boundary={b}"})
    for t in range(30):   # rides out server restarts (~25 min)
        try:
            with urllib.request.urlopen(req, timeout=900) as r:
                return Image.open(io.BytesIO(base64.b64decode(json.load(r)["data"][0]["b64_json"]))).convert("RGB")
        except (OSError, ValueError, KeyError) as e:
            err = e; time.sleep(min(60, 5 * (t + 1)))
    raise RuntimeError(repr(err)[:300])


def candidates():
    R = pd.read_parquet(WORK / "refs/permissive_refs.parquet").set_index("sha256")
    c = json.load(open(WORK / "lora_data/multi/_descriptions.json"))
    rows = []
    for sha, d in c.items():
        if "error" in d or d.get("diagnosis_source") == "unknown" or not d.get("is_clinical_skin_photo"):
            continue
        if d["fitzpatrick"] not in RN or sha not in R.index:
            continue
        r = R.loc[sha]
        if r.license_class not in ("cc_by", "cc0") or not m94.site_ok(d["body_site"]):
            continue
        if d["age_group"] in m94.MINOR and not m94.minor_ok(d["body_site"]):
            continue
        rows.append(dict(ref_sha256=sha, ref_path=r.path, ref_license=r.license_class, ref_attribution=r.attribution,
                         ref_source=r.doi or r.pmcid, diagnosis=d["diagnosis"], variant=d["variant"], stage=d["stage"],
                         site=d["body_site"], age_group=d["age_group"], sex=d["sex"], src_fst=RN[d["fitzpatrick"]],
                         dx_source=d["diagnosis_source"],
                         morphology=d["morphology"]))
    C = pd.DataFrame(rows)
    pe, _, thr = importlib.import_module("99_batch_review").screen(C.ref_path)
    return C[pe < thr].reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-refs", type=int, default=60)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--precheck-only", action="store_true", help="only run the Splash vision pre-check (no GPU needed)")
    ap.add_argument("--api", default=API, help="edit endpoint, e.g. http://localhost:8080 (Splash klein)")
    ap.add_argument("--model", default=MODEL, help="flux2-klein-4b (donv) or klein (Splash)")
    a = ap.parse_args()
    out = Path(a.out); (out / "img").mkdir(parents=True, exist_ok=True)
    C = candidates()
    C = C[(C.src_fst <= 4) & ~C.diagnosis.str.contains(PIGMENTARY) & ~C.variant.fillna("").str.contains(PIGMENTARY)]
    print(len(C), "eligible references (Fitzpatrick I-IV, non-pigmentary)", flush=True)
    C["dx_key"] = C.diagnosis.str.lower().map(m94.norm)
    # spread over diagnoses first (one per diagnosis), then fill up; --n-refs 0 = all eligible references
    # diagnoses named by the article label first (caption-recovered diagnoses are noisier), then shuffled
    C = C.sample(frac=1, random_state=5).sort_values("dx_source", key=lambda x: x.ne("label"), kind="stable")
    first = C.drop_duplicates("dx_key")
    pick = pd.concat([first, C.drop(first.index)])
    pick = precheck(pick if a.n_refs == 0 else pick.head(int(a.n_refs * 1.6)))   # vision pre-check, cached
    pick = pick if a.n_refs == 0 else pick.head(a.n_refs)
    if a.precheck_only:
        print("pre-check only: done"); return
    tab = pd.read_csv(m94.PLAN / "conditioning_table.csv")
    vocab = {m94.norm(n): int(c) for c, n in zip(tab.concept_id, tab.concept_name)}
    rows = []
    for r in pick.itertuples():
        for t in targets(r.src_fst):
            sid = f"rv_{r.ref_sha256[:12]}_f{t}"
            who = {"female": "woman", "male": "man"}.get(r.sex, "patient") if r.age_group in ("adult", "elderly") else r.age_group
            caption = (f"{r.diagnosis}, {r.variant}, {r.stage} stage, {r.site}, {r.age_group} {r.sex}, "
                       f"Fitzpatrick {ROMAN[t]} ({m94.SKIN[t]}).")
            rows.append(dict(r._asdict(), synth_id=sid, fst_target=t, prompt=edit_prompt(t), caption=caption,
                             lifecycle_key=r.dx_key, concept_id=vocab.get(r.dx_key), seed=int(r.ref_sha256[:7], 16) + t,
                             path=str((out / "img" / f"{sid}.jpg").resolve()), generator=f"klein-4b edit of CC reference ({a.model})",
                             modification=f"AI edit (FLUX.2-klein-4B): skin colour changed to Fitzpatrick {ROMAN[t]}, re-rendered at higher resolution",
                             who=who))
    M = pd.DataFrame(rows).drop(columns=["Index"], errors="ignore")
    M["concept_id"] = M.concept_id.astype("Int64")
    todo = M[[not Path(p).exists() for p in M.path]]
    print(len(M), "variants,", len(todo), "to generate,", M.diagnosis.nunique(), "diagnoses", flush=True)
    t0 = time.time()

    def run(r):
        post_edit(r.ref_path, r.prompt, out_size(r.ref_path), r.seed, a.api, a.model).save(r.path, quality=93)
    list(ThreadPoolExecutor(a.workers).map(run, todo.itertuples()))
    print(f"{(time.time() - t0) / max(1, len(todo)):.1f} s/variant", flush=True)
    M = M[[Path(p).exists() for p in M.path]]
    # closeness to the source photograph (CLIP ViT-B/16, same space as the gate)
    ev = importlib.import_module("60_eval_clip")
    import types
    be = ev.load_backend(types.SimpleNamespace(backend="openclip", pretrained="openai", model="ViT-B-16-quickgelu"),
                         importlib.import_module("common").device())
    A = be.encode_images(M.path.tolist()); B = be.encode_images(M.ref_path.tolist())
    M["ref_cos"] = (np.asarray(A) * np.asarray(B)).sum(1)
    M.to_parquet(out / "manifest.parquet", index=False)
    print("done", len(M), "| ref_cos median", round(float(M.ref_cos.median()), 3))


if __name__ == "__main__":
    main()
