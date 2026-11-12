"""Exact top-k cosine neighbours between two embedding sets (blockwise on MPS).

Usage:
  python 21_neighbors.py <query.npy> <base.npy> <out.parquet> [--k 5] [--min-sim 0.85] [--self]
--self: query == base; the trivial self-match is removed.
Rows whose embedding is all-zero (unreadable image) never match.
Output columns: q (query row), b (base row), sim.
"""
from common import device
import argparse

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("base")
    ap.add_argument("out")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--min-sim", type=float, default=0.85)
    ap.add_argument("--self", action="store_true")
    ap.add_argument("--block", type=int, default=4096)
    a = ap.parse_args()

    dev = device()
    Q = np.load(a.query)
    B = torch.from_numpy(np.load(a.base)).to(dev).half()
    qs, bs, ss = [], [], []
    k = a.k + 1 if a.self else a.k
    for s in tqdm(range(0, len(Q), a.block), mininterval=30):
        q = torch.from_numpy(Q[s:s + a.block]).to(dev).half()
        sim = q @ B.T
        val, idx = sim.topk(k, dim=1)
        val, idx = val.float().cpu().numpy(), idx.cpu().numpy()
        rows = np.arange(s, s + len(q))[:, None].repeat(k, 1)
        m = val >= a.min_sim
        if a.self:
            m &= idx != rows
        qs.append(rows[m]); bs.append(idx[m]); ss.append(val[m])
    df = pd.DataFrame({"q": np.concatenate(qs), "b": np.concatenate(bs), "sim": np.concatenate(ss)})
    df.to_parquet(a.out, index=False)
    print(len(df), "pairs >=", a.min_sim)


if __name__ == "__main__":
    main()
