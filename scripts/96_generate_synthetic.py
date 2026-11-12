"""Generate SynthDerm images for rows of a prompt plan (94_lifecycle_prompts.py) through an OpenAI-style image API:
the donv klein server (FLUX.2-klein-4B, INT8, via `ssh -N -L 8188:127.0.0.1:8188 donv`) or the Splash image server
(model "klein"). Resumable: existing images are skipped. Writes <out>/img/<synth_id>.jpg and <out>/manifest.parquet
(plan columns + path + generator), the input format of 86_eval_synthetic.py and annotator/import_items.py.

Usage: python 96_generate_synthetic.py --plan ../dataset/synthetic/plan/lifecycle_pilot.parquet --out ../work/synthetic/pilot_klein4
       [--api http://127.0.0.1:8188 --model flux2-klein-4b] [--size 1024] [--workers 1]
"""
import argparse
import base64
import io
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from PIL import Image


def gen(api, model, prompt, seed, size):
    body = {"model": model, "prompt": prompt, "size": f"{size}x{size}", "n": 1, "seed": int(seed) % 2**31,
            "response_format": "b64_json"}
    req = urllib.request.Request(api + "/v1/images/generations", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    for t in range(30):   # rides out server restarts / short GPU windows (~25 min in total)
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                return Image.open(io.BytesIO(base64.b64decode(json.load(r)["data"][0]["b64_json"]))).convert("RGB")
        except (OSError, ValueError, KeyError) as e:
            err = e; time.sleep(min(60, 5 * (t + 1)))
    raise RuntimeError(repr(err)[:300])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--api", default="http://127.0.0.1:8188")
    ap.add_argument("--model", default="flux2-klein-4b")
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args()
    P = pd.read_parquet(a.plan)
    out = Path(a.out); (out / "img").mkdir(parents=True, exist_ok=True)
    P["path"] = [str((out / "img" / f"{s}.jpg").resolve()) for s in P.synth_id]
    todo = P[[not Path(p).exists() for p in P.path]]
    print(len(todo), "of", len(P), "to generate", flush=True)
    t0 = time.time()

    def run(r):
        im = gen(a.api, a.model, r.prompt, r.seed, a.size)
        im.save(r.path, quality=93)
        return r.synth_id
    with ThreadPoolExecutor(a.workers) as ex:
        for i, sid in enumerate(ex.map(run, todo.itertuples())):
            if (i + 1) % 10 == 0:
                print(i + 1, sid, f"{(time.time() - t0) / (i + 1):.1f} s/img", flush=True)
    P = P[[Path(p).exists() for p in P.path]].assign(generator=f"{a.model} @ {a.api.split('//')[-1]}", size=a.size)
    P.to_parquet(out / "manifest.parquet", index=False)
    print("done", len(P), "images,", round((time.time() - t0) / 60, 1), "min")


if __name__ == "__main__":
    main()
