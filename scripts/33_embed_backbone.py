"""Assemble adapter-training data for one frozen backbone: image embeddings of the CLIP-experiment
training pairs (work/pairs/{ours,derm1m}.parquet, 69,536 each, taken from the full embeddings of
34_embed_full.py) plus their text embeddings, the zero-shot class prompts and SkinCAP captions.

Usage: python 33_embed_backbone.py --name medsiglip_448 --backend medsiglip --pretrained google/medsiglip-448
Writes work/backbone/<name>/{ours,derm1m}_pairs_{img,txt}.npy, prompts.npz, skincap_txt.npy, skincap_index.parquet
"""
import argparse
import importlib
import json
import sys
import types

import numpy as np
import pandas as pd

from common import DATASET, WORK, device


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--backend", required=True, choices=["openclip", "medsiglip", "embeddinggemma"])
    ap.add_argument("--pretrained", required=True)
    ap.add_argument("--model", default="ViT-B-16-quickgelu")
    a = ap.parse_args()
    sys.argv = [sys.argv[0]]
    ev = importlib.import_module("60_eval_clip")
    be = ev.load_backend(types.SimpleNamespace(backend=a.backend, pretrained=a.pretrained, model=a.model), device())
    out = WORK / "backbone" / a.name

    II = pd.read_parquet(WORK / "image_info.parquet", columns=["src_path", "sha256"])
    DI = pd.read_parquet(DATASET / "metadata/images.parquet", columns=["file", "sha256"])
    for ds, lst in [("ours", "ours"), ("derm1m", "derm1m")]:
        P = pd.read_parquet(WORK / f"pairs/{ds}.parquet")
        idx = pd.read_parquet(out / f"{lst}_index.parquet")
        E = np.load(out / f"{lst}_img.npy")
        # match by content hash: the full-embedding index keeps one path per unique image, while pairs may
        # reference a byte-identical copy under another path (0.6% of Derm1M pairs)
        rel = lambda p: p.split("/wbs/", 1)[-1]   # same files on Mac and donv
        sha_of = {**dict(zip(II.src_path.map(rel), II.sha256)), **dict(zip(DI.file.map(lambda f: "dataset/" + f), DI.sha256))}
        if "sha256" in idx:
            pos = dict(zip(idx.sha256, range(len(idx))))
            rows = [pos[sha_of[rel(p)]] for p in P.path]
        else:
            pos = dict(zip(idx.path.map(rel), range(len(idx))))
            rows = [pos[rel(p)] for p in P.path]
        np.save(out / f"{ds}_pairs_img.npy", E[rows])
        np.save(out / f"{ds}_pairs_txt.npy", be.encode_texts(P.text.tolist()).astype(np.float16))
        print(ds, len(rows), flush=True)

    prom = {}
    for bench, classes in ev.CLASSES.items():
        prom[bench + "_raw"] = np.stack([be.encode_texts([t.format(c=c) for t in ev.TEMPLATES]) for c in classes])
    np.savez(out / "prompts.npz", **prom)
    B = pd.read_parquet(WORK / "bench.parquet")
    sc = B[B.bench == "SkinCAP"].dropna(subset=["caption"])
    np.save(out / "skincap_txt.npy", be.encode_texts(sc.caption.tolist()))
    sc[["path"]].to_parquet(out / "skincap_index.parquet", index=False)
    json.dump(vars(a), open(out / "meta.json", "w"))
    print("done", a.name)


if __name__ == "__main__":
    main()
