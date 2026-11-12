"""Same-photo detection beyond near-duplicates: crops, zooms and re-hosted copies of one photograph.

22_dedup links images at CLIP cos >= 0.98 (or >= 0.95 with pHash distance <= 10) and mutual top-3 neighbours
at >= 0.96. Crops / zooms of the same photograph (e.g. an atlas re-hosting a cropped DermIS image) often have
cos 0.93-0.97 and very different pHashes, so they escaped and could end up in different splits.

Most of these pairs are not crops but different shots of the same patient (other angle / distance, same
session), which neither CLIP similarity nor keypoint matching separates from different patients with the same
disease. Linking rules, calibrated on 72 hand-judged pairs (work/crop_pairs_labelsample.parquet; 12 per stratum):
  R1 different source, same atlas diagnosis concept, cos >= 0.95   -> 9/12 same photo or patient (re-hosted
     atlas images, e.g. dermaamin copies of DermIS / Atlas Dermatologico photos with other crop / watermark)
  R2 same source, same atlas diagnosis concept, cos >= 0.93        -> about 50% same patient (conservative)
  R3 any other pair with cos >= 0.95 whose SIFT match proves the same photograph: Lowe ratio 0.75, RANSAC
     similarity transform (reprojection 5 px), >= 50 inliers and >= 25% of the matches (below ~30 inliers,
     atlas watermarks / logos produce false matches)
  not linked: different / unknown diagnosis without SIFT proof (1/12 same; mostly journal histology).
Linked pairs are merged into the case groups by 40_build.py, so both images end up in the same split.

Writes work/crop_candidates.parquet (all candidates + rule flags) and work/crop_dups.parquet (linked pairs).
Usage: python 23_crop_dups.py [--min-cos 0.93] [--k 10] [--workers 10]
"""
import argparse
from multiprocessing import Pool

import numpy as np
import pandas as pd

from common import DATASET, WORK, load_local_embeddings

SIDE = 640


def sift(path):
    import cv2
    im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if im is None:
        return None, None
    s = SIDE / max(im.shape)
    if s < 1:
        im = cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    kp, des = cv2.SIFT_create(nfeatures=1500).detectAndCompute(im, None)
    return (np.float32([k.pt for k in kp]) if kp else None), des


def verify(pair):
    import cv2
    pa, pb = pair
    ka, da = sift(pa)
    kb, db = sift(pb)
    if da is None or db is None or len(da) < 8 or len(db) < 8:
        return 0, 0
    m = cv2.BFMatcher(cv2.NORM_L2).knnMatch(da, db, k=2)
    good = [x[0] for x in m if len(x) == 2 and x[0].distance < 0.75 * x[1].distance]
    if len(good) < 8:
        return len(good), 0
    A = ka[[g.queryIdx for g in good]]
    B = kb[[g.trainIdx for g in good]]
    _, inl = cv2.estimateAffinePartial2D(A, B, method=cv2.RANSAC, ransacReprojThreshold=5.0)
    return len(good), int(inl.sum()) if inl is not None else 0


def candidates(min_cos, k):
    import torch
    P, E = load_local_embeddings()
    I = pd.read_parquet(DATASET / "metadata/images.parquet", columns=["sha256", "case_group", "split", "file"])
    P = P.reset_index().merge(I, on="sha256").drop_duplicates("sha256")
    X = torch.from_numpy(E[P["index"].values].astype(np.float32))
    cg = P.case_group.values
    out = []
    for s in range(0, len(X), 4096):
        sim = X[s:s + 4096] @ X.T
        v, j = sim.topk(k + 1, dim=1)
        v, j = v.numpy(), j.numpy()
        for r in range(len(v)):
            q = s + r
            for vv, jj in zip(v[r], j[r]):
                if jj > q and vv >= min_cos and cg[jj] != cg[q]:
                    out.append((q, jj, float(vv)))
    C = pd.DataFrame(out, columns=["a", "b", "cos"]).drop_duplicates(["a", "b"])
    for x in "ab":
        C[f"sha256_{x}"] = P.sha256.values[C[x]]
        C[f"split_{x}"] = P.split.values[C[x]]
        C[f"path_{x}"] = [str(DATASET / f) for f in P.file.values[C[x]]]
    return C.drop(columns=["a", "b"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-cos", type=float, default=0.93)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--min-inliers", type=int, default=50)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0, help="verify only a random sample (calibration)")
    ap.add_argument("--out", default=str(WORK / "crop_dups.parquet"))
    a = ap.parse_args()
    C = candidates(a.min_cos, a.k)
    print("candidate pairs (different case groups):", len(C), "| cross-split:", int((C.split_a != C.split_b).sum()))
    # atlas diagnosis concepts per image (from the atlas label, 53_diagnosis_link) + source
    S = pd.read_parquet(DATASET / "metadata/samples.parquet", columns=["sha256", "src_path", "source", "source_type"])
    dx = pd.read_parquet(WORK / "dx_samples_ours.parquet").explode("concepts").dropna(subset=["concepts"])
    dx = dx.merge(S[S.source_type == "atlas"][["src_path", "sha256"]], on="src_path")
    con = dx.groupby("sha256").concepts.agg(lambda x: set(int(c) for c in x))
    src = S.groupby("sha256").source.agg(set)
    ca, cb = C.sha256_a.map(con), C.sha256_b.map(con)
    C["same_dx"] = [isinstance(x, set) and isinstance(y, set) and bool(x & y) for x, y in zip(ca, cb)]
    C["same_source"] = [bool(x & y) for x, y in zip(C.sha256_a.map(src), C.sha256_b.map(src))]
    C["R1"] = ~C.same_source & C.same_dx & (C.cos >= 0.95)
    C["R2"] = C.same_source & C.same_dx & (C.cos >= 0.93)
    sub = C[~C.R1 & ~C.R2 & (C.cos >= 0.95)]
    if a.limit:
        sub = sub.sample(min(a.limit, len(sub)), random_state=0)
    print("SIFT verification of", len(sub), "pairs")
    with Pool(a.workers) as pool:
        r = pool.map(verify, list(zip(sub.path_a, sub.path_b)), chunksize=16)
    C["matches"], C["inliers"] = -1, -1
    C.loc[sub.index, "matches"] = [x[0] for x in r]
    C.loc[sub.index, "inliers"] = [x[1] for x in r]
    C["R3"] = (C.inliers >= a.min_inliers) & (C.inliers >= 0.25 * C.matches)
    C["link"] = C.R1 | C.R2 | C.R3
    C.to_parquet(WORK / "crop_candidates.parquet", index=False)
    L = C[C.link]
    L.to_parquet(a.out, index=False)
    print({k: int(C[k].sum()) for k in ["R1", "R2", "R3", "link"]},
          "| cross-split linked:", int((L.split_a != L.split_b).sum()))

if __name__ == "__main__":
    main()
