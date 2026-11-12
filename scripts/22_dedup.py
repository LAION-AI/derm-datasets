"""Near-duplicate clusters, case groups and benchmark-overlap flags for all local images.

Thresholds were calibrated by visual inspection of CLIP-cosine bands (see
dataset/docs/PROVENANCE.md):
  near_dup  : cos >= 0.98, or cos >= 0.95 and pHash Hamming distance <= 10
              (same image re-encoded / re-watermarked / slightly cropped)
  case group: near-dup links + photos that are mutual top-3 neighbours at cos >= 0.96
              (same patient/lesion photographed twice); used for leakage-free splits.
Benchmark overlap: any local image with cos >= 0.95 to a benchmark image
(PAD-UFES-20, HAM10000, Fitzpatrick17k, Daffodil, SkinCAP) is flagged.

Writes work/dedup.parquet (sha256, near_dup_cluster, case_group, bench_overlap, bench_overlap_test).
"""
import subprocess
import sys

import imagehash
import numpy as np
import pandas as pd

from common import WORK, load_local_embeddings

PY = sys.executable


class UF:
    def __init__(self, n):
        self.p = np.arange(n)

    def find(self, a):
        p = self.p
        while p[a] != a:
            p[a] = p[p[a]]
            a = p[a]
        return a

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def main():
    P, E = load_local_embeddings()
    np.save(WORK / "emb/local_all.npy", E)
    P.to_parquet(WORK / "embed_local_all.parquet", index=False)

    subprocess.run([PY, "21_neighbors.py", str(WORK / "emb/local_all.npy"), str(WORK / "emb/local_all.npy"),
                    str(WORK / "nn_local_all.parquet"), "--self", "--k", "10", "--min-sim", "0.95"], check=True)
    nn = pd.read_parquet(WORK / "nn_local_all.parquet")
    nn = nn[nn.q < nn.b]
    info = pd.read_parquet(WORK / "image_info.parquet").drop_duplicates("sha256").set_index("sha256")
    ph = [imagehash.hex_to_hash(h) for h in info.loc[P.sha256, "phash"].values]
    nn = nn.assign(ham=[ph[a] - ph[b] for a, b in zip(nn.q.values, nn.b.values)])

    n = len(P)
    nd, cg = UF(n), UF(n)
    for a, b, s, h in zip(nn.q.values, nn.b.values, nn.sim.values, nn.ham.values):
        if s >= 0.98 or h <= 10:
            nd.union(a, b)
            cg.union(a, b)
    # case groups: additionally link photos that are mutual top-3 neighbours at cos >= 0.96.
    # (plain transitive linking at 0.95 chained 22k look-alike clinical photos / 7k H&E slides
    # into single components; this rule keeps the largest group at ~100 images)
    heads = pd.read_parquet(WORK / "heads_pred.parquet")[["sha256", "image_type"]].dropna().drop_duplicates("sha256")
    t = P.merge(heads, on="sha256", how="left").image_type.values
    photo = np.isin(t, ["clinical", "dermoscopy", "other_photo"])
    allnn = pd.read_parquet(WORK / "nn_local_all.parquet").sort_values(["q", "sim"], ascending=[True, False])
    allnn["rank"] = allnn.groupby("q").cumcount()
    e = allnn[(allnn.sim >= 0.96) & (allnn["rank"] < 3)]
    key = set(zip(e.q.values, e.b.values))
    for a, b in key:
        if a < b and (b, a) in key and photo[a] and photo[b]:
            cg.union(a, b)
    out = pd.DataFrame({"sha256": P.sha256,
                        "near_dup_cluster": [nd.find(i) for i in range(n)],
                        "case_group": [cg.find(i) for i in range(n)]})

    # benchmark overlap (query = benchmark images, base = local images)
    subprocess.run([PY, "21_neighbors.py", str(WORK / "emb/bench.npy"), str(WORK / "emb/local_all.npy"),
                    str(WORK / "nn_bench_local.parquet"), "--k", "10", "--min-sim", "0.95"], check=True)
    bn = pd.read_parquet(WORK / "nn_bench_local.parquet")
    B = pd.read_parquet(WORK / "embed_bench.parquet").merge(
        pd.read_parquet(WORK / "bench.parquet").drop_duplicates("path"), on="path", how="left")
    # an image is "test" for a benchmark if it is in its ZS test set or LP test split
    test_paths = set(pd.read_parquet(WORK / "bench.parquet").query("split == 'test'").path)
    bn["bench"] = B.bench.values[bn.q.values]
    bn["is_test"] = B.path.isin(test_paths).values[bn.q.values]
    agg = bn.groupby("b").agg(bench_overlap=("bench", lambda x: ",".join(sorted(set(x)))),
                              bench_overlap_test=("is_test", "max"))
    out = out.join(agg, how="left")
    out["bench_overlap_test"] = out.bench_overlap_test.fillna(False).astype(bool)
    out.to_parquet(WORK / "dedup.parquet", index=False)
    print("images", n, "| near-dup clusters", out.near_dup_cluster.nunique(),
          "| case groups", out.case_group.nunique(),
          "| largest case group", out.case_group.value_counts().iloc[0])
    print("bench overlap:", out.bench_overlap.value_counts().to_dict(),
          "| overlapping a test image:", int(out.bench_overlap_test.sum()))


if __name__ == "__main__":
    main()
