"""Concept map: diagnosis concepts of our dataset vs Derm1M (PaCMAP layout, HDBSCAN clusters).

Concepts are the SapBERT concepts of 53_diagnosis_link.py (shared vocabulary of our atlas labels, Derm1M
disease labels and phrases found in English journal captions). Count = unique images per concept and dataset
(ours: dataset images; Derm1M: all images). The map shows concepts with >= MIN_IMAGES images in at least one
dataset. Layout: PaCMAP on the SapBERT embedding of the concept name (cosine-like: unit vectors); clusters:
HDBSCAN on the 2-D layout (so cluster shading matches what is seen). Cluster names: 57_cluster_names.py.

Writes work/concepts/concept_map.json (points, clusters, totals for all thresholds).
"""
import json

import hdbscan
import numpy as np
import pacmap
import pandas as pd

from common import DATASET, WORK

MIN_IMAGES = 2
OUT = WORK / "concepts"


def counts():
    II = pd.read_parquet(WORK / "image_info.parquet", columns=["src_path", "sha256"])
    S = pd.read_parquet(DATASET / "metadata/samples.parquet", columns=["src_path", "sha256"]).drop_duplicates("src_path")
    cnt = {}
    for n in ["ours", "derm1m"]:
        d = pd.read_parquet(WORK / f"dx_samples_{n}.parquet").explode("concepts").dropna(subset=["concepts"])
        if n == "ours":
            d = d.merge(S, on="src_path")
        else:
            d = d.merge(II, on="src_path", how="left")
            d["sha256"] = d.sha256.fillna(d.src_path)
        cnt[n] = d.groupby("concepts").sha256.nunique()
    C = pd.concat(cnt, axis=1).fillna(0).astype(int)
    C.index = C.index.astype(int)
    return C


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    V = pd.read_parquet(WORK / "dx_vocab.parquet")
    E = np.load(WORK / "dx_vocab_sapbert.npy")
    C = counts()
    totals = {str(t): {"ours": int((C.ours >= t).sum()), "derm1m": int((C.derm1m >= t).sum()),
                       "both": int(((C.ours >= t) & (C.derm1m >= t)).sum()),
                       "union": int(((C.ours >= t) | (C.derm1m >= t)).sum())} for t in [1, 2, 3, 5, 10, 20]}
    C = C[(C.ours >= MIN_IMAGES) | (C.derm1m >= MIN_IMAGES)]

    # concept vector = SapBERT embedding of the leader term (the concept name)
    V = V.assign(row=np.arange(len(V)))
    lead = V[V.term == V.concept_name].drop_duplicates("concept_id").set_index("concept_id")
    C = C[C.index.isin(lead.index)]
    X = E[lead.loc[C.index, "row"].values]
    syn = (V.sort_values("freq", ascending=False).groupby("concept_id")
           .term.agg(lambda t: [x for x in t][:6]))

    # FP_ratio 8 (more far pairs) separates concept families into islands; chosen over the defaults on static
    # comparisons (n_neighbors 10/15/30, FP_ratio 2/4/8): clusters do not overlap and none dominates the centre
    Y = pacmap.PaCMAP(n_components=2, n_neighbors=15, FP_ratio=8.0, random_state=0).fit_transform(X, init="pca")
    lab = hdbscan.HDBSCAN(min_cluster_size=25, min_samples=8, cluster_selection_method="eom").fit_predict(Y)
    print("concepts", len(C), "clusters", lab.max() + 1, "noise", float((lab < 0).mean()))

    pts = []
    for (cid, r), (x, y), l in zip(C.iterrows(), Y, lab):
        pts.append({"id": int(cid), "n": lead.concept_name[cid], "o": int(r.ours), "d": int(r.derm1m),
                    "x": round(float(x), 3), "y": round(float(y), 3), "c": int(l),
                    "s": [t for t in syn.get(cid, []) if t != lead.concept_name[cid]][:4],
                    "on": lead.concept_ontology_node[cid] if isinstance(lead.concept_ontology_node[cid], str) else None})
    P = pd.DataFrame(pts)
    clusters = []
    for l, g in P[P.c >= 0].groupby("c"):
        g = g.assign(t=g.o + g.d).sort_values("t", ascending=False)
        clusters.append({"c": int(l), "size": len(g), "top": g.n.head(25).tolist(),
                         "ours": int((g.o >= MIN_IMAGES).sum()), "derm1m": int((g.d >= MIN_IMAGES).sum())})
    json.dump({"min_images": MIN_IMAGES, "totals": totals, "points": pts, "clusters": clusters},
              open(OUT / "concept_map.json", "w"), ensure_ascii=False)
    print(json.dumps(totals))


if __name__ == "__main__":
    main()
