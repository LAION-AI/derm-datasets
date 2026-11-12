"""Image-type head on OpenAI CLIP ViT-L/14 embeddings, the embedding space of the Re-LAION-2B-en kNN index
(laion/relaion2B-en-research-safe-index), so that 107_laion_index_score.py can classify every indexed image.

The ViT-B/16 heads (32_heads.py, 71 stage4) cannot be applied to ViT-L/14 vectors, so the head is re-trained:
  corpus images  confident ViT-B/16 head labels (p >= 0.9; at most 4,000 per class) + the 383 hand-labelled gold images
                 (weight 5);
  web_other      random Re-LAION images without a dermatology caption term (controls of 71 and of 105 fetch), the
                 explicit open-world negative class that made round 2 of 71 work, plus 200,000 uniformly random index
                 entries reconstructed from their PQ codes (107 sample; quantised like the vectors that are scored;
                 ~0.03% expected dermatology contamination).
5-fold CV: gold accuracy, derm-core precision / recall on gold (exact and PQ-quantised embeddings), and the out-of-fold
flag rate on web controls and on random index entries.
Writes work/laion5b/head_vitl14.joblib + head_vitl14.json.
Usage: python 106_laion_index_head.py
"""
import json
import subprocess
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from common import WORK

OUT = WORK / "laion5b"
CORE = ["clinical", "dermoscopy", "histopathology"]
GOLD_TO_HEAD = {"chart": "figure_nonphoto", "diagram": "figure_nonphoto", "table_text": "figure_nonphoto",
                "molecular": "figure_nonphoto"}


def trainset():
    h = pd.read_parquet(WORK / "heads_pred.parquet")
    pcols = [c for c in h if c.startswith("type_p_")]
    h["pmax"] = h[pcols].max(axis=1)
    h = h[h.pmax >= 0.9].sample(frac=1, random_state=0)
    corpus = (h[h.groupby("image_type").cumcount() < 4000][["path", "image_type"]]
              .rename(columns={"image_type": "y"}).assign(w=1.0, src="corpus"))
    g = pd.read_csv(WORK / "gold_types.csv")
    gold = pd.DataFrame({"path": g.path, "y": g.label.map(lambda l: GOLD_TO_HEAD.get(l, l)), "w": 5.0, "src": "gold"})
    corpus = corpus[~corpus.path.isin(set(gold.path))]
    web = [pd.read_parquet(WORK / "laion/stage4_scored.parquet").query("group == 'control'")[["path"]]]
    if (OUT / "sample.parquet").exists():
        s = pd.read_parquet(OUT / "sample.parquet")
        web.append(s[(s.group == "control") & (s.status == "ok") & (s.subset == "en")][["path"]])
    web = pd.concat(web).assign(y="web_other", w=1.0, src="web")
    return pd.concat([corpus, gold, web], ignore_index=True).drop_duplicates("path")


def main():
    T = trainset()
    T[["path"]].to_parquet(OUT / "head_vitl14_paths.parquet", index=False)
    pref = OUT / "head_vitl14_emb"
    subprocess.run([sys.executable, "30_embed.py", str(OUT / "head_vitl14_paths.parquet"), str(pref),
                    "--model", "ViT-L-14-quickgelu", "--pretrained", "openai"], check=True)
    E = np.load(f"{pref}.npy").astype(np.float32)
    ok = np.load(f"{pref}.ok.npy")
    T, E = T[ok].reset_index(drop=True), E[ok]
    z = np.load(OUT / "index_random.npz")
    T = pd.concat([T, pd.DataFrame({"path": [f"index:{i}" for i in z["ids"]], "y": "web_other", "w": 1.0,
                                    "src": "index"})], ignore_index=True)
    E = np.vstack([E, z["X"].astype(np.float32)])
    gi = np.where(T.src == "gold")[0]
    np.save(OUT / "gold_vitl14.npy", E[gi])
    subprocess.run([sys.executable, "107_laion_index_score.py", "quantize", str(OUT / "gold_vitl14.npy"),
                    str(OUT / "gold_vitl14_pq.npy")], check=True)
    Eq = np.load(OUT / "gold_vitl14_pq.npy")
    fit = lambda X, y, w: LogisticRegression(max_iter=4000, C=2.0, class_weight="balanced").fit(X, y, sample_weight=w)
    oof, pcore = np.empty(len(T), dtype=object), np.zeros(len(T))
    oofq, pcoreq = np.empty(len(gi), dtype=object), np.zeros(len(gi))
    for k, (tr, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=0).split(E, T.y)):
        m = fit(E[tr], T.y.values[tr], T.w.values[tr])
        core = np.isin(m.classes_, CORE)
        oof[te], pcore[te] = m.predict(E[te]), m.predict_proba(E[te])[:, core].sum(1)
        sel = np.isin(gi, te)
        oofq[sel], pcoreq[sel] = m.predict(Eq[sel]), m.predict_proba(Eq[sel])[:, core].sum(1)
        print("fold", k, flush=True)
    T["oof"], T["p_core_oof"] = oof, pcore
    g, web, ix = T[T.src == "gold"], T[T.src == "web"], T[T.src == "index"]
    core_t, core_p = g.y.isin(CORE), g.oof.isin(CORE)
    thr = {str(t): {"gold_core_recall_pq": round(float((pcoreq[core_t.values] >= t).mean()), 3),
                    "index_random_rate_pct": round(100 * float((ix.p_core_oof >= t).mean()), 4)}
           for t in [0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99]}
    rep = {"train_counts": T.y.value_counts().to_dict(),
           "gold_n": len(g), "gold_acc_cv": round(float((g.y == g.oof).mean()), 3),
           "gold_derm_core_precision_cv": round(float((core_t & core_p).sum() / max(1, core_p.sum())), 3),
           "gold_derm_core_recall_cv": round(float((core_t & core_p).sum() / max(1, core_t.sum())), 3),
           "gold_derm_core_recall_pq_cv": round(float((core_t.values & np.isin(oofq, CORE)).sum() / max(1, core_t.sum())), 3),
           "web_control_n": len(web), "web_control_flag_rate_pct_cv": round(100 * float(web.oof.isin(CORE).mean()), 3),
           "index_random_n": len(ix), "index_random_flag_rate_pct_cv": round(100 * float(ix.oof.isin(CORE).mean()), 3),
           "by_threshold_p_core": thr}
    m = fit(E, T.y.values, T.w.values)
    joblib.dump(m, OUT / "head_vitl14.joblib")
    T.to_parquet(OUT / "head_vitl14_train.parquet", index=False)
    json.dump(rep, open(OUT / "head_vitl14.json", "w"), indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
