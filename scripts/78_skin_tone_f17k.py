"""Skin-colour evaluation on Fitzpatrick17k (expert Fitzpatrick labels): models trained on our dataset vs on Derm1M.
(Planned on DDI; DDI access is restricted, so the same protocol runs on F17K, using the official F17K split of the Derm1M
benchmark package. Caveat: F17K comes from two of our atlases - exact / near-duplicates were removed from training, but
style is shared.)

Frozen image embeddings cached by 60_eval_clip.py (work/eval/feats_<model>.npz). Models: OpenAI CLIP ViT-B/16, CLIP
fine-tuned on ours / Derm1M (1k steps, 3 seeds each; 3k steps; union 3k; ours captions only), MedSigLIP-448,
EmbeddingGemma-2. Probes: standardised features, L2 logistic regression, C chosen on the validation split, evaluated on
the test split (one fixed split, as in the benchmark package).
 1. Skin-type prediction: Fitzpatrick group (I-II / III-IV / V-VI; fitzpatrick_scale 1-6, unknown excluded).
    Balanced accuracy, macro AUROC, recall per group.
 2. Fairness per Fitzpatrick group (test images of each group):
    a. malignant vs benign (F17K three_partition_label, neoplastic lesions only): probe AUROC per group,
    b. 114-disease accuracy per group (probe), next to the zero-shot accuracy per group (60_eval_clip.py, Table 3).
    Gap = I-II minus V-VI. 95% CIs by bootstrap over test images (1,000 resamples).
Writes work/eval/skin_tone_f17k.json.
"""
import json
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from common import WORK

GROUPS = ["I–II", "III–IV", "V–VI"]
GRID = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1]
MODELS = ["openai_vitb16"] + [f"clip_ft_{d}{s}" for d in ["ours", "derm1m"] for s in ["", "_s1", "_s2"]] + \
    ["clip_ft_ours_3k", "clip_ft_derm1m_3k", "clip_ft_union_3k", "clip_ft_ours_caption_only", "medsiglip_448", "embeddinggemma2"]
rel = lambda p: p.split("/wbs/", 1)[-1]


def data():
    B = pd.read_parquet(WORK / "bench.parquet")
    d = B[(B.bench == "F17K") & (B.task == "LP")].copy()
    F = pd.read_csv(WORK.parent / "external/fitzpatrick17k/fitzpatrick17k.csv")
    d["md5"] = d.path.map(lambda p: os.path.splitext(os.path.basename(p))[0])
    d = d.merge(F[["md5hash", "fitzpatrick_scale", "three_partition_label"]], left_on="md5", right_on="md5hash", how="left")
    d["fst"] = d.fitzpatrick_scale.where(d.fitzpatrick_scale.between(1, 6))
    d["group"] = pd.cut(d.fst, [0, 2, 4, 6], labels=GROUPS).astype(str).replace("nan", None)
    d["rp"] = d.path.map(rel)
    return d


def fit_select(Xtr, ytr, Xva, yva, score):
    best = None
    for C in GRID:
        m = LogisticRegression(C=C, max_iter=4000).fit(Xtr, ytr)
        s = score(m, Xva, yva)
        if best is None or s > best[0]:
            best = (s, C)
    return LogisticRegression(C=best[1], max_iter=4000).fit(np.vstack([Xtr]), ytr), best[1]


def boot(fn, idx_groups, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        out.append(fn({g: rng.choice(ix, len(ix)) for g, ix in idx_groups.items()}))
    return np.array(out)


def main():
    D = data()
    print("F17K LP", D.groupby(["split", "group"]).size().unstack().to_dict(orient="index"))
    res = {"n": D.groupby(["split", "group"]).size().unstack().fillna(0).astype(int).to_dict(orient="index"), "models": {}}
    zs = {}
    for name in MODELS:
        f = WORK / f"eval/feats_{name}.npz"
        if not f.exists():
            print("missing", name); continue
        z = np.load(f, allow_pickle=True)
        pos = {rel(p): i for i, p in enumerate(z["paths"])}
        if not D.rp.isin(pos).all():
            print("incomplete feats", name); continue
        X = z["F"].astype(np.float32)[[pos[p] for p in D.rp]]
        sp = {s: (D.split == s).values for s in ["train", "val", "test"]}
        sc = StandardScaler().fit(X[sp["train"]])
        Xs = sc.transform(X)
        r = {}
        # 1. skin-type group
        k = D.group.notna().values
        yg = pd.Categorical(D.group, categories=GROUPS).codes
        tr, va, te = sp["train"] & k, sp["val"] & k, sp["test"] & k
        m, C = fit_select(Xs[tr], yg[tr], Xs[va], yg[va], lambda m, X_, y_: balanced_accuracy_score(y_, m.predict(X_)))
        P = m.predict_proba(Xs[te]); pred = P.argmax(1); yt = yg[te]
        r["skin_type"] = {"C": C, "bal_acc": balanced_accuracy_score(yt, pred),
                          "auroc_macro": roc_auc_score(np.eye(3)[yt], P, multi_class="ovr", average="macro"),
                          **{f"recall_{g}": float((pred[yt == i] == i).mean()) for i, g in enumerate(GROUPS)}}
        # 2a. malignant vs benign (neoplastic only)
        neo = D.three_partition_label.isin(["malignant", "benign"]).values
        ym = (D.three_partition_label == "malignant").values.astype(int)
        tr, va, te = sp["train"] & neo, sp["val"] & neo, sp["test"] & neo & k
        m, C = fit_select(Xs[tr], ym[tr], Xs[va], ym[va], lambda m, X_, y_: roc_auc_score(y_, m.predict_proba(X_)[:, 1]))
        s = m.predict_proba(Xs[te])[:, 1]; yt = ym[te]; gt = D.group.values[te]
        ix = {g: np.where(gt == g)[0] for g in GROUPS}
        auc = lambda I: {g: roc_auc_score(yt[I[g]], s[I[g]]) for g in GROUPS}
        point = auc(ix)
        b = boot(lambda I: (lambda a: [a[g] for g in GROUPS] + [a["I–II"] - a["V–VI"]])(auc(I)), ix)
        r["malignancy_auroc"] = {"C": C, **point, "gap": point["I–II"] - point["V–VI"],
                                 "ci95": {k_: [float(np.percentile(b[:, j], 2.5)), float(np.percentile(b[:, j], 97.5))]
                                          for j, k_ in enumerate(GROUPS + ["gap"])},
                                 "n_test": {g: [int(len(ix[g])), int(yt[ix[g]].sum())] for g in GROUPS}}
        # 2b. 114-disease accuracy per group (probe)
        yd = D.label.values.astype(int)
        tr, va, te = sp["train"], sp["val"], sp["test"] & k
        m, C = fit_select(Xs[tr], yd[tr], Xs[va], yd[va], lambda m, X_, y_: (m.predict(X_) == y_).mean())
        hit = m.predict(Xs[te]) == yd[te]; gt = D.group.values[te]
        ix = {g: np.where(gt == g)[0] for g in GROUPS}
        acc = lambda I: {g: float(hit[I[g]].mean()) for g in GROUPS}
        point = acc(ix)
        b = boot(lambda I: (lambda a: [a[g] for g in GROUPS] + [a["I–II"] - a["V–VI"]])(acc(I)), ix)
        r["disease_acc"] = {"C": C, **point, "gap": point["I–II"] - point["V–VI"],
                            "ci95": {k_: [float(np.percentile(b[:, j], 2.5)), float(np.percentile(b[:, j], 97.5))]
                                     for j, k_ in enumerate(GROUPS + ["gap"])}}
        # zero-shot accuracy per group (already computed by 60_eval_clip.py)
        j = WORK / f"eval/{name}.json"
        if j.exists():
            jj = json.load(open(j))
            r["zero_shot_acc"] = {g: jj.get(f"zs_F17K_acc_fst_{g.replace('–', '-')}") for g in GROUPS}
        res["models"][name] = r
        print(f"{name:26s} skin-type bal.acc {r['skin_type']['bal_acc']:.3f} V-VI recall {r['skin_type']['recall_V–VI']:.3f} | "
              f"malig AUROC " + " ".join(f"{r['malignancy_auroc'][g]:.3f}" for g in GROUPS) + f" gap {r['malignancy_auroc']['gap']:+.3f} | "
              f"disease acc " + " ".join(f"{r['disease_acc'][g]:.3f}" for g in GROUPS) + f" gap {r['disease_acc']['gap']:+.3f}", flush=True)
    json.dump(res, open(WORK / "eval/skin_tone_f17k.json", "w"), indent=1, ensure_ascii=False, default=float)


if __name__ == "__main__":
    main()
