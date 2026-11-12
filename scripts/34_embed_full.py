"""Embed complete image lists with a frozen backbone (OpenCLIP / MedSigLIP-448 / EmbeddingGemma-2),
sharded across GPUs and resumable (chunks of 4,000 images).

Lists: ours    = every unique image of dataset/metadata/images.parquet
       derm1m  = every unique Derm1M image (work/embed_derm1m.parquet)
       bench   = every benchmark image (work/embed_bench.parquet)
Usage (one process per GPU):
  CUDA_VISIBLE_DEVICES=0 python 34_embed_full.py --name medsiglip_448 --backend medsiglip \
      --pretrained google/medsiglip-448 --list derm1m --shard 0 --nshards 2
  ... then: python 34_embed_full.py --name medsiglip_448 --list derm1m --merge --nshards 2
Writes work/backbone/<name>/<list>_img.npy (float16, row order = <list>_index.parquet).
"""
import argparse
import importlib
import os
import sys
import types

import numpy as np
import pandas as pd

from common import DATASET, WORK, device

CHUNK = 4000


def image_list(name):
    if name == "ours":
        I = pd.read_parquet(DATASET / "metadata/images.parquet")
        return pd.DataFrame({"sha256": I.sha256, "path": [str(DATASET / f) for f in I.file]})
    if name == "derm1m":
        return pd.read_parquet(WORK / "embed_derm1m.parquet")[["sha256", "path"]]
    if name == "bench":
        return pd.read_parquet(WORK / "embed_bench.parquet")[["path"]]
    raise ValueError(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--backend", choices=["openclip", "medsiglip", "embeddinggemma"])
    ap.add_argument("--pretrained")
    ap.add_argument("--model", default="ViT-B-16-quickgelu")
    ap.add_argument("--list", required=True, choices=["ours", "derm1m", "bench"])
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--merge", action="store_true")
    a = ap.parse_args()
    out = WORK / "backbone" / a.name
    out.mkdir(parents=True, exist_ok=True)
    L = image_list(a.list).reset_index(drop=True)
    chunks = list(range(0, len(L), CHUNK))
    part = lambda c: out / f"{a.list}_img.part{c:04d}.npy"
    if a.merge:
        missing = [c for c in range(len(chunks)) if not part(c).exists()]
        assert not missing, f"missing chunks {missing[:10]}"
        np.save(out / f"{a.list}_img.npy", np.concatenate([np.load(part(c)) for c in range(len(chunks))]))
        L.to_parquet(out / f"{a.list}_index.parquet", index=False)
        for c in range(len(chunks)):
            os.remove(part(c))
        print("merged", a.list, len(L))
        return
    sys.argv = [sys.argv[0]]
    ev = importlib.import_module("60_eval_clip")
    be = ev.load_backend(types.SimpleNamespace(backend=a.backend, pretrained=a.pretrained, model=a.model), device())
    for c, s in enumerate(chunks):
        if c % a.nshards != a.shard or part(c).exists():
            continue
        np.save(part(c), be.encode_images(L.path.tolist()[s:s + CHUNK]).astype(np.float16))
        print(a.list, c, "/", len(chunks), flush=True)


if __name__ == "__main__":
    main()
