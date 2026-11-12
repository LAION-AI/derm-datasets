"""Compute L2-normalised CLIP image embeddings for a list of images.

Usage:
  python 30_embed.py <paths.parquet> <out_prefix> [--model ViT-B-16 --pretrained openai]
The parquet needs a column `path`. Writes <out_prefix>.npy (float16, N x D, row order
= parquet order; rows for unreadable images are zero) and <out_prefix>.ok.npy (bool).
Resumable: work is done in chunks of 20k rows saved as <out_prefix>.partNNN.npy.
"""
from common import device
import argparse
import os
import warnings

import numpy as np
import open_clip
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

Image.MAX_IMAGE_PIXELS = None
warnings.filterwarnings("ignore")
CHUNK = 20000


class DS(Dataset):
    def __init__(self, paths, tf):
        self.paths, self.tf = paths, tf

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        try:
            with Image.open(self.paths[i]) as im:
                im.seek(0)
                x = self.tf(im.convert("RGB"))
            return x, True
        except Exception:
            return torch.zeros(3, 224, 224), False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths")
    ap.add_argument("out")
    ap.add_argument("--model", default="ViT-B-16-quickgelu")
    ap.add_argument("--pretrained", default="openai")
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()

    paths = pd.read_parquet(a.paths).path.tolist()
    dev = device()
    model, _, tf = open_clip.create_model_and_transforms(a.model, pretrained=a.pretrained)
    model = model.to(dev).eval().half() if dev != "cpu" else model.eval()

    parts = []
    for c, start in enumerate(range(0, len(paths), CHUNK)):
        pf = f"{a.out}.part{c:03d}.npz"
        parts.append(pf)
        if os.path.exists(pf):
            continue
        sub = paths[start:start + CHUNK]
        dl = DataLoader(DS(sub, tf), batch_size=a.bs, num_workers=a.workers)
        embs, oks = [], []
        with torch.no_grad():
            for x, ok in tqdm(dl, desc=f"chunk {c}", mininterval=60):
                x = x.to(dev).half() if dev != "cpu" else x
                e = model.encode_image(x)
                e = e / e.norm(dim=-1, keepdim=True)
                embs.append(e.float().cpu().numpy().astype(np.float16))
                oks.append(ok.numpy())
        np.savez(pf, e=np.concatenate(embs), ok=np.concatenate(oks))
    E = [np.load(p) for p in parts]
    np.save(f"{a.out}.npy", np.concatenate([e["e"] for e in E]))
    np.save(f"{a.out}.ok.npy", np.concatenate([e["ok"] for e in E]))
    for p in parts:
        os.remove(p)
    print("done", len(paths))


if __name__ == "__main__":
    main()
