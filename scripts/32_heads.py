"""Train linear heads on CLIP ViT-B/16 image embeddings and predict for all images.

1) Image type (10 classes) -> binary flags is_clinical / is_dermoscopy /
   is_histopathology / is_radiology ... (Aim 2 of the proposal: binary medical
   image classifiers). Training labels:
     dermoscopy      HAM10000 (benchmark LP train+val, dermoscopy by construction)
     clinical        Fitzpatrick17k + PAD-UFES-20 (LP train+val) + SkinCAP
     histopathology  DermNet pathology-topic images with zero-shot histology p>0.5,
                     plus local images with zero-shot histology p>0.95
     other classes   local journal/textbook images with zero-shot p>0.95 for that class
   Validation: hand-labelled gold set (work/gold_types.csv, 383 images), 5-fold CV; the final head
   is trained on weak labels + all gold labels (weight 5).
2) Fitzpatrick skin type (1-6): ridge regression on F17K expert labels (fitzpatrick_scale),
   5-fold CV grouped by near-duplicate cluster; predictions rounded and clipped.

Usage: python 32_heads.py
Inputs: work/emb/{local_b1,local_b2,bench}.npy, work/embed_*.parquet, work/zs_*.parquet,
        work/bench.parquet, work/atlases.parquet
Output: work/heads_pred.parquet (sha256 -> type probs, image_type, fst_pred, fst_pred_raw),
        work/heads_report.json
"""
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import GroupKFold
from sklearn.metrics import classification_report, confusion_matrix

from common import WORK, load_local_embeddings

TYPES = ["clinical", "dermoscopy", "histopathology", "microscopy_other", "radiology",
         "other_photo", "figure_nonphoto"]
# zero-shot classes -> head classes (charts/diagrams/tables/blots are merged: the distinction that
# matters for a dermatology corpus is photo type vs. non-photographic figure)
ZS_TO_HEAD = {"chart": "figure_nonphoto", "diagram": "figure_nonphoto", "table_text": "figure_nonphoto",
              "radiology": "radiology", "microscopy_other": "microscopy_other", "other_photo": "other_photo"}
GOLD_TO_HEAD = {"chart": "figure_nonphoto", "diagram": "figure_nonphoto", "table_text": "figure_nonphoto",
                "molecular": "figure_nonphoto"}


def main():
    rep = {}
    P, E, Z = load_local_embeddings(with_zeroshot=True)
    E = E.astype(np.float32)
    B = pd.read_parquet(WORK / "bench.parquet").drop_duplicates("path").reset_index(drop=True)
    BP = pd.read_parquet(WORK / "embed_bench.parquet")
    BE = np.load(WORK / "emb/bench.npy").astype(np.float32)
    B = BP.merge(B, on="path", how="left")
    assert len(B) == len(BE)

    # ---------------- image type training set
    X, y, src = [], [], []
    def add(mask_idx, emb, label, tag):
        X.append(emb[mask_idx]); y.extend([label] * len(mask_idx)); src.extend([tag] * len(mask_idx))
    lp = B.task.isin(["LP", "RET"]) & B.split.isin(["train", "val", "test"])
    add(np.where((B.bench == "HAM") & lp)[0], BE, "dermoscopy", "HAM")
    add(np.where(B.bench.isin(["F17K", "PAD", "SkinCAP"]) & lp)[0], BE, "clinical", "F17K/PAD/SkinCAP")
    atl = pd.read_parquet(WORK / "atlases.parquet", columns=["src_path", "extra", "source"])
    patho_paths = set(atl[atl.extra.fillna("").str.contains('"is_pathology_topic": true')].src_path)
    is_patho = P.path.isin(patho_paths).values
    add(np.where(is_patho & (Z.p_histopathology.values > 0.5))[0], E, "histopathology", "dermnet-pathology")
    for zt, ht in ZS_TO_HEAD.items():  # (molecular zero-shot labels were unreliable -> not used)
        thr = 0.90 if zt == "other_photo" else 0.95
        idx = np.where((Z[f"p_{zt}"].values > thr) & ~is_patho)[0]
        add(idx, E, ht, f"zeroshot>{zt}")
    X = np.concatenate(X); y = np.array(y); w = np.ones(len(y))
    rep["type_train_counts_weak"] = pd.Series(y).value_counts().to_dict()

    # hand-labelled gold set: 5-fold CV (weak labels + 4/5 gold, weight 5) for evaluation,
    # then the final head is trained on weak labels + all gold labels
    gold_f = WORK / "gold_types.csv"
    pos = {p: i for i, p in enumerate(P.path)}
    g = pd.read_csv(gold_f) if gold_f.exists() else pd.DataFrame(columns=["path", "label", "part"])
    g["label"] = g.label.map(lambda l: GOLD_TO_HEAD.get(l, l))
    g = g[g.path.isin(pos)].reset_index(drop=True)
    G = E[[pos[p] for p in g.path]]
    def fit(Xa, ya, wa):
        return LogisticRegression(max_iter=4000, C=2.0, class_weight="balanced").fit(Xa, ya, sample_weight=wa)
    if len(g):
        from sklearn.model_selection import StratifiedKFold
        cv_pred = np.empty(len(g), dtype=object)
        strat = g.label.where(g.label.map(g.label.value_counts()) >= 5, "rare")
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(G, strat):
            m = fit(np.concatenate([X, G[tr]]), np.concatenate([y, g.label.values[tr]]),
                    np.concatenate([w, np.full(len(tr), 5.0)]))
            cv_pred[te] = m.predict(G[te])
        g["head_cv"] = cv_pred
        zmap = dict(zip(P.path, Z.zs_type))
        g["zs"] = g.path.map(zmap).map(lambda l: GOLD_TO_HEAD.get(l, ZS_TO_HEAD.get(l, l)))
        for part in ["all", "random"]:
            gg = g if part == "all" else g[g.part == "random"]
            rep[f"gold_{part}_n"] = len(gg)
            rep[f"gold_{part}_acc_head_cv"] = float((gg.label == gg.head_cv).mean())
            rep[f"gold_{part}_acc_zeroshot"] = float((gg.label == gg.zs).mean())
        rep["gold_report_head_cv"] = classification_report(g.label, g.head_cv, output_dict=True, zero_division=0)
        labs = sorted(set(g.label) | set(g.head_cv))
        rep["gold_confusion_head_cv"] = {"labels": labs, "matrix": confusion_matrix(g.label, g.head_cv, labels=labs).tolist()}
        for flag in ["clinical", "dermoscopy", "histopathology", "radiology"]:
            t, p_ = g.label == flag, g.head_cv == flag
            tp = int((t & p_).sum())
            rep[f"gold_binary_{flag}"] = dict(n_pos=int(t.sum()), precision=tp / max(1, int(p_.sum())),
                                              recall=tp / max(1, int(t.sum())))
        core = g.label.isin(["clinical", "dermoscopy", "histopathology"])
        core_p = g.head_cv.isin(["clinical", "dermoscopy", "histopathology"])
        rep["gold_binary_derm_core"] = dict(precision=float((core & core_p).sum() / core_p.sum()),
                                            recall=float((core & core_p).sum() / core.sum()))
        print(classification_report(g.label, g.head_cv, zero_division=0))
        print({k: v for k, v in rep.items() if k.startswith("gold_") and not isinstance(v, dict)})
        X = np.concatenate([X, G]); y = np.concatenate([y, g.label.values]); w = np.concatenate([w, np.full(len(g), 5.0)])
    clf = fit(X, y, w)
    classes = list(clf.classes_)
    np.savez_compressed(WORK / "heads_trainset.npz", X=X.astype(np.float16), y=y, w=w)  # reused by 71 stage4

    allE = np.concatenate([E, BE])
    prob = clf.predict_proba(allE)
    pred = np.array(classes)[prob.argmax(1)]
    out = pd.DataFrame(prob, columns=[f"type_p_{c}" for c in classes])
    out["image_type"] = pred
    out["sha256"] = list(P.sha256) + [None] * len(BE)
    out["path"] = list(P.path) + list(B.path)

    # ---------------- Fitzpatrick head (F17K expert labels)
    f = B[(B.bench == "F17K") & B.fst.notna()].reset_index()
    Xf = BE[f["index"].values]
    yf = f.fst.values.astype(float)
    # group near-duplicates (cos >= 0.95) so CV folds do not share images
    S = Xf @ Xf.T
    parent = np.arange(len(f))
    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a
    ii, jj = np.where(np.triu(S >= 0.95, 1))
    for a, b in zip(ii, jj):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    groups = np.array([find(a) for a in range(len(f))])
    oof = np.zeros(len(f))
    for tr, te in GroupKFold(5).split(Xf, yf, groups):
        m = Ridge(alpha=1.0).fit(Xf[tr], yf[tr])
        oof[te] = m.predict(Xf[te])
    r = np.clip(np.rint(oof), 1, 6)
    rep["fst_cv"] = dict(n=len(f), mae=float(np.abs(oof - yf).mean()), exact=float((r == yf).mean()),
                         within1=float((np.abs(r - yf) <= 1).mean()),
                         spearman=float(pd.Series(oof).corr(pd.Series(yf), method="spearman")),
                         # binary light (I-III) vs dark (IV-VI); dark-skin recall matters for bias analysis
                         dark_recall=float(((r >= 4) & (yf >= 4)).sum() / max(1, (yf >= 4).sum())),
                         dark_precision=float(((r >= 4) & (yf >= 4)).sum() / max(1, (r >= 4).sum())),
                         label_dist=pd.Series(yf).value_counts().sort_index().to_dict())
    m = Ridge(alpha=1.0).fit(Xf, yf)
    raw = m.predict(allE)
    out["fst_pred_raw"] = raw
    out["fst_pred"] = np.clip(np.rint(raw), 1, 6).astype(int)
    print("FST CV:", rep["fst_cv"])

    import joblib
    joblib.dump({"type": clf, "fst": m}, WORK / "heads_models.joblib")
    out.to_parquet(WORK / "heads_pred.parquet", index=False)
    json.dump(rep, open(WORK / "heads_report.json", "w"), indent=1, default=str)
    print("type train counts (weak)", rep["type_train_counts_weak"])
    print(out[out.sha256.notna()].image_type.value_counts())


if __name__ == "__main__":
    main()
