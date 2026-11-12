"""Hybrid generation test for dark skin (batch 1: gate pass falls from 49% on Fitzpatrick I to 32% on VI; the gate itself
passes real V-VI photographs at 80-87%). Arms, on the same plan rows (Fitzpatrick V-VI, one per diagnosis):
  direct       - batch 1 image (klein text-to-image with the full disease prompt)
  klein_edit   - klein healthy base person -> klein edit adds the lesion ("keep everything else unchanged")
  zimage_edit  - Z-Image-Turbo healthy base person (most realistic skin, Apache-2.0) -> klein edit adds the lesion
Steps:
  prep        select rows, write work/synthetic/hybrid/{rows.parquet, zimage_rows.jsonl}
  klein-base  healthy bases via the donv klein server (tunnel :8188)
  edit        --base klein_base|zimage_base: lesion edits via the donv klein server
  manifests   gate-ready manifests (same synth_id across arms) for 99_batch_review.py
Z-Image bases run on donv: 97_gen_diffusers.py --pipeline zimage --base Tongyi-MAI/Z-Image-Turbo --steps 9 --guidance 0.0
"""
import argparse
import base64
import io
import json
import time
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "work/synthetic/hybrid"
B1 = ROOT / "work/synthetic/batch1"
API, MODEL = "http://127.0.0.1:8188", "flux2-klein-4b"
HEALTHY = " Healthy, normal-looking skin of this body part, without any rash, lesion, scar or discolouration."
KEEP = (" Keep the person, body part, framing, skin tone, skin texture, lighting and background exactly unchanged; "
        "only add the lesion. Realistic clinical photograph.")


def split_prompt(p):
    """plan prompt -> (style + subject, lesion part after 'Diagnosis: ...')."""
    head, tail = p.split(" Diagnosis: ", 1)
    return head, "Diagnosis: " + tail


def post(url, body=None, files=None, timeout=900):
    if files is None:
        req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    else:
        b = uuid.uuid4().hex
        parts = [f'--{b}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode() for k, v in body.items()]
        for k, (name, data) in files.items():
            parts.append(f'--{b}\r\nContent-Disposition: form-data; name="{k}"; filename="{name}"\r\nContent-Type: image/jpeg\r\n\r\n'.encode()
                         + data + b"\r\n")
        parts.append(f"--{b}--\r\n".encode())
        req = urllib.request.Request(url, data=b"".join(parts), headers={"Content-Type": f"multipart/form-data; boundary={b}"})
    for t in range(4):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                d = json.load(r)["data"][0]
            return Image.open(io.BytesIO(base64.b64decode(d["b64_json"]))).convert("RGB")
        except (OSError, ValueError, KeyError) as e:
            err = e; time.sleep(5 * (t + 1))
    raise RuntimeError(repr(err)[:300])


def prep(n):
    M = pd.read_parquet(B1 / "manifest.parquet")
    E = pd.read_parquet(B1 / "eval/per_image.parquet")[["synth_id"]]
    R = M[M.synth_id.isin(E.synth_id) & M.fst_target.isin([5, 6]) & M.ab_pair.isna() & M.path.map(lambda p: Path(p).exists())]
    R = R.sample(frac=1, random_state=3).drop_duplicates("lifecycle_key").head(n).reset_index(drop=True)
    R["base_prompt"] = R.prompt.map(lambda p: split_prompt(p)[0] + HEALTHY)
    R["edit_prompt"] = R.prompt.map(lambda p: "Add this skin condition to the photograph. " + split_prompt(p)[1] + KEEP)
    OUT.mkdir(parents=True, exist_ok=True)
    R.to_parquet(OUT / "rows.parquet", index=False)
    R[["synth_id", "seed"]].assign(prompt=R.base_prompt).to_json(OUT / "zimage_rows.jsonl", orient="records", lines=True)
    print(len(R), "rows,", R.lifecycle_key.nunique(), "diagnoses; fst", R.fst_target.value_counts().to_dict())


def klein_base():
    R = pd.read_parquet(OUT / "rows.parquet"); d = OUT / "klein_base"; d.mkdir(exist_ok=True)

    def run(r):
        f = d / f"{r.synth_id}.jpg"
        if not f.exists():
            post(API + "/v1/images/generations", {"model": MODEL, "prompt": r.base_prompt, "size": "1024x1024", "n": 1,
                                                  "seed": int(r.seed) % 2**31, "response_format": "b64_json"}).save(f, quality=93)
    list(ThreadPoolExecutor(2).map(run, R.itertuples())); print("klein bases:", len(list(d.glob("*.jpg"))))


def edit(base):
    R = pd.read_parquet(OUT / "rows.parquet"); src = OUT / base; d = OUT / base.replace("_base", "_edit"); d.mkdir(exist_ok=True)

    def run(r):
        f, b = d / f"{r.synth_id}.jpg", src / f"{r.synth_id}.jpg"
        if b.exists() and not f.exists():
            post(API + "/v1/images/edits", {"model": MODEL, "prompt": r.edit_prompt, "size": "1024x1024",
                                            "response_format": "b64_json", "seed": str(int(r.seed) % 2**31)},
                 files={"image": (b.name, b.read_bytes())}).save(f, quality=93)
    t = time.time(); list(ThreadPoolExecutor(2).map(run, R.itertuples()))
    print(base, "edits:", len(list(d.glob("*.jpg"))), f"{(time.time() - t) / max(1, len(R)):.1f} s/row")


def manifests():
    R = pd.read_parquet(OUT / "rows.parquet")
    for arm in ["klein_edit", "zimage_edit", "direct"]:
        d = OUT / arm; d.mkdir(exist_ok=True)
        m = R.copy() if arm == "direct" else R.assign(path=[str(d / f"{s}.jpg") for s in R.synth_id], generator=arm)
        m = m[m.path.map(lambda p: Path(p).exists())].drop(columns=["base_prompt"]).assign(prompt=lambda x: x.edit_prompt if arm != "direct" else x.prompt)
        m.to_parquet(d / "manifest.parquet", index=False); print(arm, len(m))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["prep", "klein-base", "edit", "manifests"])
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--base", default="klein_base")
    a = ap.parse_args()
    {"prep": lambda: prep(a.n), "klein-base": klein_base, "edit": lambda: edit(a.base), "manifests": manifests}[a.step]()
