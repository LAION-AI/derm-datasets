"""Name the reference-pool clusters (83_ref_clusters.py) with the Splash vision LLM (incoai/Qwen3.8-27B-Splash):
for each view (image / text / joint) and cluster, up to 30 member captions + diagnosis labels and 4 sample images
(384 px) are sent; the model returns a short cluster name and a one-sentence description of what unites the images
(diagnosis, stage, body site, image style). Resumable cache: work/refs/clusters/cluster_names.json.
"""
import base64
import io
import json
import os
import re
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from PIL import Image

from common import WORK

OUT = WORK / "refs/clusters"
BASE = os.environ.get("LLM_BASE_URL", "http://localhost:4242/v1")
MODEL = os.environ.get("LLM_MODEL", "incoai/Qwen3.8-27B-Splash")
SCHEMA = {"type": "object", "properties": {"name": {"type": "string"}, "description": {"type": "string"},
                                           "dominant_diagnoses": {"type": "array", "items": {"type": "string"}},
                                           "body_sites": {"type": "array", "items": {"type": "string"}},
                                           "stages_or_variants": {"type": "array", "items": {"type": "string"}}},
          "required": ["name", "description", "dominant_diagnoses", "body_sites", "stages_or_variants"]}


def img_url(path):
    im = Image.open(path); im.draft("RGB", (768, 768)); im = im.convert("RGB"); im.thumbnail((384, 384))
    b = io.BytesIO(); im.save(b, "JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode()


def ask(view, c, g):
    caps = [f"- [{d}] {str(t)[:220]}" for d, t in zip(g.diagnosis.fillna("?"), g.caption.fillna(""))][:30]
    content = [{"type": "text", "text": (
        f"These are {len(g)} clinical photographs from open-access dermatology articles that were grouped together by "
        f"{'image similarity' if view == 'image' else 'caption similarity' if view == 'text' else 'combined image and caption similarity'}. "
        "Below are up to 30 of their source captions with the article's diagnosis label in brackets, and 4 sample images. "
        "Give the cluster a short specific name (2-6 words, e.g. 'Facial acne, inflammatory papules' or 'Plantar psoriasis plaques') "
        "and describe in one sentence what unites the images (diagnosis, subtype / stage, body site, imaging style). "
        "Rely on the source labels for the diagnosis; do not invent diagnoses.\n\n" + "\n".join(caps))}]
    for p in g.sample(min(4, len(g)), random_state=0).path:
        try:
            content.append({"type": "image_url", "image_url": {"url": img_url(p)}})
        except Exception:
            pass
    body = {"model": MODEL, "reasoning_effort": "low", "temperature": 0, "max_tokens": 3000,
            "messages": [{"role": "user", "content": content}],
            "response_format": {"type": "json_schema", "json_schema": {"name": "cluster", "schema": SCHEMA}}}
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
    C = pd.read_parquet(OUT / "clusters.parquet")
    cache_f = OUT / "cluster_names.json"
    cache = json.load(open(cache_f)) if cache_f.exists() else {}
    cache = {k: v for k, v in cache.items() if "error" not in v and v.get("name")}   # retry failed / empty entries
    todo = [(v, int(c)) for v in ["joint", "image", "text"] for c in sorted(C[f"c_{v}"].unique()) if c >= 0
            and f"{v}:{c}" not in cache]
    print(len(todo), "clusters to name", flush=True)

    def run(vc):
        v, c = vc
        g = C[C[f"c_{v}"] == c]
        return f"{v}:{c}", {**ask(v, c, g), "size": len(g), "model": MODEL}
    with ThreadPoolExecutor(2) as ex:
        for i, (k, r) in enumerate(ex.map(run, todo)):
            cache[k] = r
            print(i + 1, k, r.get("name"), flush=True)
            if (i + 1) % 5 == 0:
                json.dump(cache, open(cache_f, "w"), indent=1, ensure_ascii=False)
    json.dump(cache, open(cache_f, "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
