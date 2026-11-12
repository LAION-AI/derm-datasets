"""Skin-colour evaluation on DDI (Diverse Dermatology Images; Daneshjou et al., Sci Adv 2022): models trained on our
dataset vs on Derm1M.

DDI: 656 images with expert Fitzpatrick groups (skin_tone 12 / 34 / 56 = I-II / III-IV / V-VI) and a
malignant / benign label (dermatopathology-confirmed). Expected under external/ddi/ (ddi_metadata.csv + images,
as distributed by Stanford AIMI; the folder layout inside is searched).

Models (frozen image embeddings): OpenAI CLIP ViT-B/16; CLIP fine-tuned on ours and on Derm1M (1k steps, 3 seeds
each; 3k steps; union 3k); MedSigLIP-448 and EmbeddingGemma-2 as references.
Tasks:
 1. Skin-type prediction: multinomial logistic regression (standardised features, C by inner 3-fold CV) predicting the
    Fitzpatrick group; 5-fold stratified CV repeated 5x. Balanced accuracy, macro AUROC, recall per group.
 2. Fairness, malignant vs benign, AUROC per Fitzpatrick group:
    a. zero-shot (prompts "malignant skin lesion" / "benign skin lesion" with the Derm1M templates),
    b. linear probe (5-fold stratified CV repeated 5x, out-of-fold scores).
    Gap = AUROC(I-II) - AUROC(V-VI).
Writes work/eval/ddi.json (+ work/eval/ddi_feats_<model>.npz).
Usage (GPU; donv): python 77_eval_ddi.py [--ddi ../external/ddi]
"""
import argparse
import importlib
import json
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import WORK, device

GROUP = {12: "I–II", 34: "III–IV", 56: "V–VI"}
MODELS = [("openai_vitb16", "openclip", "openai")] + \
    [(f"clip_ft_{d}{s}", "openclip", f"models/clip_{d}{s}/model.pt") for d in ["ours", "derm1m"] for s in ["", "_s1", "_s2"]] + \
    [(f"clip_ft_{d}_3k", "openclip", f"models/clip_{d}_3k/model.pt") for d in ["ours", "derm1m", "union"]] + \
    [("medsiglip_448", "medsiglip", "/mnt/nvme/wbs/models/medsiglip-448"),
     ("embeddinggemma2", "embeddinggemma", "google/embeddinggemma-2")]


def load_ddi(root):
    csv = next(Path(root).rglob("ddi_metadata.csv"))
    M = pd.read_csv(csv)
    files = {p.name: p for p in Path(root).rglob("*") if p.suffix.lower() in (".png", ".jpg", ".jpeg")}
    M["path"] = M.DDI_file.map(lambda f: str(files[f]) if f in files else None)
    assert M.path.notna().all(), f"{M.path.isna().sum()} DDI images not found"
    M["group"] = M.skin_tone.astype(int).map(GROUP)
    M["malignant"] = M.malignant.astype(str).str.lower().isin(["true", "1"]).astype(int)
    return M


def skin_type_probe(X, g, seed=0):
    y = pd.Categorical(g, categories=list(GROUP.values())).codes
    oof = np.zeros((len(y), 3)); n = np.zeros(len(y))
    for tr, te in RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=seed).split(X, y):
        clf = make_pipeline(StandardScaler(), LogisticRegressionCV(Cs=[0.001, 0.01, 0.1, 1], cv=3, max_iter=3000))
        oof[te] += clf.fit(X[tr], y[tr]).predict_proba(X[te]); n[te] += 1
    oof /= n[:, None]
    pred = oof.argmax(1)
    return {"bal_acc": balanced_accuracy_score(y, pred),
            "auroc_macro": roc_auc_score(np.eye(3)[y], oof, multi_class="ovr", average="macro"),
            **{f"recall_{k}": float((pred[y == i] == i).mean()) for i, k in enumerate(GROUP.values())}}


def malignancy_probe(X, y, seed=0):
    s = np.zeros(len(y)); n = np.zeros(len(y))
    for tr, te in RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=seed).split(X, y):
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.01, max_iter=3000, class_weight="balanced"))
        s[te] += clf.fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]; n[te] += 1
    return s / n


def by_group(score, y, g):
    out = {"all": roc_auc_score(y, score)}
    for k in GROUP.values():
        m = g == k
        out[k] = roc_auc_score(y[m], score[m])
    out["gap_I-II_minus_V-VI"] = out["I–II"] - out["V–VI"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ddi", default=str(WORK.parent / "external/ddi"))
    a = ap.parse_args()
    D = load_ddi(a.ddi)
    print("DDI", len(D), D.group.value_counts().to_dict(), "malignant", int(D.malignant.sum()))
    sys.argv = [sys.argv[0]]
    ev = importlib.import_module("60_eval_clip")
    y, g = D.malignant.values, D.group.values
    res = {"n": len(D), "groups": D.group.value_counts().to_dict(), "malignant": int(y.sum()), "models": {}}
    for name, backend, pre in MODELS:
        pre = str(WORK / pre) if pre.startswith("models/") else pre
        if pre.endswith(".pt") and not Path(pre).exists():
            print("skip", name); continue
        f = WORK / f"eval/ddi_feats_{name}.npz"
        if f.exists():
            z = np.load(f); X, W = z["X"], z["W"]
        else:
            be = ev.load_backend(types.SimpleNamespace(backend=backend, pretrained=pre, model="ViT-B-16-quickgelu"), device())
            X = be.encode_images(D.path.tolist()).astype(np.float32)
            W = ev.zs_classifier(be, ["benign skin lesion", "malignant skin lesion"]).astype(np.float32)
            np.savez(f, X=X, W=W)
            del be
        zs = X @ W
        r = {"skin_type_probe": skin_type_probe(X, g),
             "malignancy_zero_shot_auroc": by_group(zs[:, 1] - zs[:, 0], y, g),
             "malignancy_probe_auroc": by_group(malignancy_probe(X, y), y, g)}
        res["models"][name] = r
        print(name, {k: round(v, 3) for k, v in r["skin_type_probe"].items()},
              "| ZS", {k: round(v, 3) for k, v in r["malignancy_zero_shot_auroc"].items()},
              "| LP", {k: round(v, 3) for k, v in r["malignancy_probe_auroc"].items()}, flush=True)
    json.dump(res, open(WORK / "eval/ddi.json", "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
