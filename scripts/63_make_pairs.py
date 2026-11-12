"""Training pairs for the equal-compute CLIP comparison (ours vs Derm1M).

Both sides are filtered identically:
  * dermatology images only (image_type in clinical / dermoscopy / histopathology, same head)
  * one image per near-duplicate cluster
  * no near-duplicate (CLIP cos >= 0.95) of ANY benchmark image (PAD, HAM, F17K, Daffodil, SkinCAP)
  * a text is required
  * ours: split == train;  Derm1M: its "train" (pretrain csv) rows
Then both are subsampled to the same number of pairs N (= size of the smaller pool).

Writes work/pairs/{ours,ours_caption_only,derm1m}.parquet (path, text) and work/pairs/summary.json.
"""
import json
import subprocess
import sys

import joblib
import numpy as np
import pandas as pd

from common import DATASET, WORK

OUT = WORK / "pairs"
OUT.mkdir(exist_ok=True)
CORE = {"clinical", "dermoscopy", "histopathology"}


def main():
    rng = np.random.default_rng(0)
    # ---------------- ours
    S = pd.read_parquet(DATASET / "metadata/samples.parquet")
    I = pd.read_parquet(DATASET / "metadata/images.parquet").set_index("image_id")
    S = S[(S.split == "train") & S.subset.eq("derm_core") & S.text.notna() & S.bench_overlap.isna()]
    S = S[S.image_id.map(I.is_canonical)]
    S = S.drop_duplicates("image_id")       # one text per image (first occurrence)
    S = S.assign(path=[str(DATASET / f) for f in S.file])
    ours = S[["path", "text", "source", "text_origin", "language"]]

    # ---------------- Derm1M
    d1 = pd.read_parquet(WORK / "derm1m.parquet")
    d1 = d1[d1.split == "train"]
    DP = pd.read_parquet(WORK / "embed_derm1m.parquet")
    DE = np.load(WORK / "emb/derm1m.npy").astype(np.float32)
    heads = joblib.load(WORK / "heads_models.joblib")
    DP["image_type"] = heads["type"].predict(DE)
    out = WORK / "nn_bench_derm1m.parquet"
    if not out.exists():
        subprocess.run([sys.executable, "21_neighbors.py", str(WORK / "emb/bench.npy"), str(WORK / "emb/derm1m.npy"),
                        str(out), "--k", "10", "--min-sim", "0.95"], check=True)
    contaminated = set(DP.sha256.values[pd.read_parquet(out).b.unique()])
    nd = pd.read_parquet(WORK / "compare/derm1m_near_dup.parquet") if (WORK / "compare/derm1m_near_dup.parquet").exists() else None
    ii = pd.read_parquet(WORK / "image_info.parquet", columns=["src_path", "sha256"])
    d1 = d1.merge(ii, on="src_path").merge(DP[["sha256", "image_type"]], on="sha256")
    d1 = d1[d1.image_type.isin(CORE) & ~d1.sha256.isin(contaminated) & d1.truncated_caption.notna()]
    if nd is not None:
        d1 = d1.merge(nd, on="sha256").drop_duplicates("near_dup_cluster")
    d1 = d1.drop_duplicates("sha256")
    derm = d1.rename(columns={"src_path": "path", "truncated_caption": "text"})[["path", "text", "source"]]

    n = min(len(ours), len(derm))
    summary = {"pool_ours": len(ours), "pool_derm1m": len(derm), "N": n,
               "derm1m_contaminated_images_removed": len(contaminated),
               "derm1m_near_dup_dedup_applied": nd is not None}
    ours.sample(n, random_state=0).to_parquet(OUT / "ours.parquet", index=False)
    derm.sample(n, random_state=0).to_parquet(OUT / "derm1m.parquet", index=False)
    oc = ours[ours.text_origin == "original_caption"]
    summary["pool_ours_caption_only"] = len(oc)
    oc.sample(min(n, len(oc)), random_state=0).to_parquet(OUT / "ours_caption_only.parquet", index=False)
    json.dump(summary, open(OUT / "summary.json", "w"), indent=1)
    print(summary)


if __name__ == "__main__":
    main()
