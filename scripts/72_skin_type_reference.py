"""Skin-type representation vs population reference.

Datasets (clinical photographs, canonical images):
  ours     Fitzpatrick head estimate (32_heads.py)          -> raw and bias-corrected
  Derm1M   same head (51_compare_derm1m.py)                  -> raw and bias-corrected
  F17K     expert labels (reference dataset)
Bias correction: the head under-detects dark skin. Its confusion matrix P(predicted | true) is re-estimated from
grouped 5-fold out-of-fold predictions on the Fitzpatrick17k expert labels (same procedure as 32_heads.py), and the
true distribution q is recovered from the observed one p by non-negative least squares (p = C q, sum q = 1).
Uncertainty: 200 bootstrap resamples of the F17K out-of-fold pairs (confusion matrix).
Assumption: the head's errors on our / Derm1M photographs resemble those on F17K (F17K comes from two of our atlases).

References (scripts/reference_prevalence.yaml): Germany (SUN-Study 2012, measured) and a modelled world scenario
(UN WPP 2024 populations x assumed regional skin-type shares; low-dark and high-dark scenario).
Writes work/bias/skin_reference.json and report/figures/fig5_skin_type_reference.{png,csv}.
"""
import json

import numpy as np
import pandas as pd
import yaml
from scipy.optimize import nnls
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

from common import DATASET, WORK

GROUPS = {"I–II": (1, 2), "III–IV": (3, 4), "V–VI": (5, 6)}


def f17k_oof():
    """Out-of-fold ridge predictions on F17K expert labels, grouped by near-duplicates (as in 32_heads.py)."""
    B = pd.read_parquet(WORK / "bench.parquet").drop_duplicates("path").reset_index(drop=True)
    BP = pd.read_parquet(WORK / "embed_bench.parquet")
    BE = np.load(WORK / "emb/bench.npy").astype(np.float32)
    B = BP.merge(B, on="path", how="left")
    f = B[(B.bench == "F17K") & B.fst.notna()].reset_index()
    X, y = BE[f["index"].values], f.fst.values.astype(int)
    S = X @ X.T
    parent = np.arange(len(f))
    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a
    for a, b in zip(*np.where(np.triu(S >= 0.95, 1))):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    groups = np.array([find(a) for a in range(len(f))])
    oof = np.zeros(len(f))
    for tr, te in GroupKFold(5).split(X, y, groups):
        oof[te] = Ridge(alpha=1.0).fit(X[tr], y[tr]).predict(X[te])
    return y, np.clip(np.rint(oof), 1, 6).astype(int)


def confusion(y, p):
    C = np.zeros((6, 6))
    for t, q in zip(y, p):
        C[q - 1, t - 1] += 1
    return C / C.sum(0, keepdims=True)          # column j: P(pred | true = j+1)


def correct(p_obs, C):
    """q >= 0 with C q ~ p_obs and sum q = 1 (constraint as a heavily weighted extra row)."""
    A = np.vstack([C, 100 * np.ones(6)])
    b = np.concatenate([p_obs, [100]])
    q, _ = nnls(A, b)
    return q / q.sum()


def grouped(d):
    d = np.asarray(d, float); d = d / d.sum()
    return {g: float(d[lo - 1:hi].sum()) for g, (lo, hi) in GROUPS.items()}


def main():
    ref = yaml.safe_load(open(WORK.parent / "scripts/reference_prevalence.yaml"))["skin_type"]
    y, p = f17k_oof()
    C = confusion(y, p)
    I = pd.read_parquet(DATASET / "metadata/images.parquet", columns=["image_type", "is_canonical", "fst_pred"])
    ours = I[(I.image_type == "clinical") & I.is_canonical].fst_pred.astype(int).value_counts().reindex(range(1, 7), fill_value=0).values
    dd = json.load(open(WORK / "compare/derm1m.json"))["A_derm1m"]["fst_dist_clinical"]
    derm = np.array([dd.get(str(k), dd.get(k, 0)) for k in range(1, 7)], float)
    f17 = np.bincount(y, minlength=7)[1:]

    rng = np.random.default_rng(0)
    res = {"confusion_P_pred_given_true": C.round(3).tolist(), "n_f17k_expert": int(len(y)),
           "head_dark_recall_V_VI": float(((p >= 5) & (y >= 5)).sum() / (y >= 5).sum())}
    for name, obs in [("ours", ours), ("derm1m", derm)]:
        po = obs / obs.sum()
        q = correct(po, C)
        boots = []
        for _ in range(200):
            i = rng.integers(0, len(y), len(y))
            Cb = confusion(y[i], p[i])
            if np.isnan(Cb).any():
                continue
            boots.append(grouped(correct(po, Cb))["V–VI"])
        res[name] = {"n": int(obs.sum()), "raw": grouped(po), "corrected": grouped(q),
                     "corrected_V_VI_ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]}
    res["f17k_expert"] = {"n": int(f17.sum()), "raw": grouped(f17)}

    g = ref["germany_sun_study_2012"]["groups"]
    res["germany_sun_2012"] = {"I–II": g["I-II"], "III–IV": g["III"] + g["IV"], "V–VI": g["V-VI_or_missing"],
                               "note": "V–VI value is an upper bound (includes missing answers)"}
    W = ref["world_scenario"]
    pop = W["population_bn"]
    for sc in ["low_dark", "high_dark"]:
        tot = np.zeros(3)
        for region, n in pop.items():
            tot += n * np.array(W[sc][region])
        res["world_" + sc] = dict(zip(GROUPS, (tot / tot.sum()).round(4).tolist()))

    # representation ratio of V–VI (dataset share / population share)
    for name in ["ours", "derm1m"]:
        for kind in ["raw", "corrected"]:
            s = res[name][kind]["V–VI"]
            res[name][f"ratio_V_VI_{kind}_vs_world"] = [s / res["world_high_dark"]["V–VI"], s / res["world_low_dark"]["V–VI"]]
    json.dump(res, open(WORK / "bias/skin_reference.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps({k: v for k, v in res.items() if k != "confusion_P_pred_given_true"}, indent=1, ensure_ascii=False))
    figure(res)


def figure(res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    C1, C2, C3, GREY, INK, MUTED, GRID, SURF = "#2a78d6", "#eb6834", "#1baf7a", "#a3a29e", "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": MUTED,
                         "xtick.color": MUTED, "ytick.color": MUTED, "figure.facecolor": SURF, "axes.facecolor": SURF,
                         "axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": 10,
                         "axes.titleweight": "bold", "axes.titlelocation": "left"})
    rows = [("Ours (estimated)", res["ours"]["raw"], C1, None), ("Ours (bias-corrected)", res["ours"]["corrected"], C1, "//"),
            ("Derm1M (estimated)", res["derm1m"]["raw"], C2, None), ("Derm1M (bias-corrected)", res["derm1m"]["corrected"], C2, "//"),
            ("Fitzpatrick17k (expert)", res["f17k_expert"]["raw"], C3, None),
            ("Germany, SUN-Study 2012", {k: res["germany_sun_2012"][k] for k in GROUPS}, GREY, None)]
    lo, hi = res["world_low_dark"], res["world_high_dark"]
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    x = np.arange(3); w = 0.12
    for i, (lab, d, col, hatch) in enumerate(rows):
        ax.bar(x + (i - 2.5) * w, [100 * d[g] for g in GROUPS], w, color=col if not hatch else SURF, edgecolor=col,
               hatch=hatch, linewidth=1.2 if hatch else 0.8, label=lab)
    for i, name in [(1, "ours"), (3, "derm1m")]:   # bootstrap 95% CI of the corrected V–VI share
        lo_, hi_ = res[name]["corrected_V_VI_ci95"]; m = res[name]["corrected"]["V–VI"]
        ax.errorbar(2 + (i - 2.5) * w, 100 * m, yerr=[[100 * (m - lo_)], [100 * (hi_ - m)]], color=INK, lw=0.8, capsize=2)
    for j, g in enumerate(GROUPS):   # world scenario: range band
        ax.fill_between([j - 0.42, j + 0.42], 100 * lo[g], 100 * hi[g], color=INK, alpha=0.10, linewidth=0, zorder=0)
    ax.fill_between([], [], [], color=INK, alpha=0.10, label="World, modelled range (UN WPP 2024 × assumptions)")
    ax.set_xticks(x, [f"Fitzpatrick {g}" for g in GROUPS]); ax.set_ylabel("share (%)")
    ax.grid(axis="y", color=GRID, linewidth=0.6); ax.set_axisbelow(True)
    ax.set_title("Skin types in the datasets vs population references")
    ax.legend(frameon=False, fontsize=7.5, ncol=2, loc="upper left", bbox_to_anchor=(0, -0.12))
    out = WORK.parent / "report/figures/fig5_skin_type_reference"
    fig.savefig(f"{out}.png", dpi=300, bbox_inches="tight")
    pd.DataFrame({lab: d for lab, d, _, _ in rows} | {"World low-dark": lo, "World high-dark": hi}).T.round(4).to_csv(f"{out}.csv")
    plt.close(fig)


if __name__ == "__main__":
    main()
