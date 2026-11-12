"""Disease representation vs known prevalence (GBD 2019), ours and Derm1M.

For the 15 skin conditions with GBD 2019 global prevalence (scripts/reference_prevalence.yaml):
  image share      = images of the condition / images of all 15 conditions (clinical + dermoscopy photographs;
                     one image per near-duplicate cluster: canonical images for ours, clusters for Derm1M)
  prevalence share = prevalent cases of the condition / prevalent cases of all 15 conditions
  representation ratio = image share / prevalence share  (1 = proportional)
Classes: over-represented ratio >= 2, under-represented ratio <= 0.5, proportional in between. Ranges use the
GBD 95% uncertainty interval of the condition's prevalence. Basal / squamous cell carcinoma are reported against
GBD 2019 incidence (no prevalence published), separately.

Matching: a concept belongs to a condition if its name or one of its synonyms matches an include pattern and that
string matches no exclude pattern, and the concept name itself matches no exclude pattern. Concepts come from atlas labels, Derm1M labels and journal caption phrases
(53_diagnosis_link.py). Matched concepts with counts: work/bias/disease_concepts.csv (for audit).
Writes work/bias/disease_prevalence.json, report/figures/fig6_disease_prevalence.{png,csv}.
"""
import json
import re

import joblib
import numpy as np
import pandas as pd
import yaml

from common import DATASET, WORK

PHOTO = {"clinical", "dermoscopy"}


def image_concepts():
    """{dataset: DataFrame(unit, concepts)} with one unit per near-duplicate cluster of a clinical/dermoscopic photo."""
    S = pd.read_parquet(DATASET / "metadata/samples.parquet", columns=["src_path", "sha256"]).drop_duplicates("src_path")
    I = pd.read_parquet(DATASET / "metadata/images.parquet", columns=["sha256", "image_type", "near_dup_cluster"])
    d = pd.read_parquet(WORK / "dx_samples_ours.parquet").explode("concepts").dropna(subset=["concepts"])
    d = d.merge(S, on="src_path").merge(I, on="sha256")
    ours = d[d.image_type.isin(PHOTO)].assign(unit=lambda x: x.near_dup_cluster.astype(str))[["unit", "concepts"]]

    II = pd.read_parquet(WORK / "image_info.parquet", columns=["src_path", "sha256"])
    DP = pd.read_parquet(WORK / "embed_derm1m.parquet")
    DP["image_type"] = joblib.load(WORK / "heads_models.joblib")["type"].predict(np.load(WORK / "emb/derm1m.npy").astype(np.float32))
    ND = pd.read_parquet(WORK / "compare/derm1m_near_dup.parquet")
    e = pd.read_parquet(WORK / "dx_samples_derm1m.parquet").explode("concepts").dropna(subset=["concepts"])
    e = e.merge(II, on="src_path").merge(DP[["sha256", "image_type"]], on="sha256").merge(ND, on="sha256")
    derm = e[e.image_type.isin(PHOTO)].assign(unit=lambda x: x.near_dup_cluster.astype(str))[["unit", "concepts"]]
    return {"ours": ours.drop_duplicates(), "derm1m": derm.drop_duplicates()}


def match(spec, terms_of):
    inc = [re.compile(p, re.I) for p in spec["include"]]
    exc = [re.compile(p, re.I) for p in spec.get("exclude", [])]
    hit = set()
    for cid, terms in terms_of.items():
        if any(r.search(terms[0]) for r in exc):   # the concept name itself must not be excluded (synonyms can be looser)
            continue
        for t in terms:
            if any(r.search(t) for r in inc) and not any(r.search(t) for r in exc):
                hit.add(cid)
                break
    return hit


def main():
    ref = yaml.safe_load(open(WORK.parent / "scripts/reference_prevalence.yaml"))["diseases"]
    V = pd.read_parquet(WORK / "dx_vocab.parquet")
    terms_of = V.groupby("concept_id").apply(lambda g: [g.concept_name.iloc[0]] + g.term.tolist(), include_groups=False).to_dict()
    names = V.drop_duplicates("concept_id").set_index("concept_id").concept_name
    D = image_concepts()

    rows, audit = [], []
    for table, kind in [("gbd2019_prevalence_mn", "prevalence"), ("gbd2019_incidence_mn", "incidence")]:
        for cond, spec in ref[table].items():
            cids = match(spec, terms_of)
            r = {"condition": cond, "measure": kind, "ref": spec["value"][0], "ref_lo": spec["value"][1],
                 "ref_hi": spec["value"][2], "n_concepts": len(cids)}
            for ds, d in D.items():
                m = d[d.concepts.astype(int).isin(cids)]
                r[f"img_{ds}"] = int(m.unit.nunique())
                for cid, n in m.groupby("concepts").unit.nunique().sort_values(ascending=False).items():
                    audit.append({"condition": cond, "dataset": ds, "concept_id": int(cid), "concept": names.get(int(cid)), "images": int(n)})
            rows.append(r)
    T = pd.DataFrame(rows)
    P = T[T.measure == "prevalence"].copy()
    tot_prev = P.ref.sum()
    for ds in D:
        tot_img = P[f"img_{ds}"].sum()
        P[f"share_img_{ds}"] = P[f"img_{ds}"] / tot_img
        P[f"ratio_{ds}"] = P[f"share_img_{ds}"] / (P.ref / tot_prev)
        # prevalence UI -> ratio range (other conditions held at their point estimates)
        P[f"ratio_{ds}_lo"] = P[f"share_img_{ds}"] / (P.ref_hi / (tot_prev - P.ref + P.ref_hi))
        P[f"ratio_{ds}_hi"] = P[f"share_img_{ds}"] / (P.ref_lo / (tot_prev - P.ref + P.ref_lo))
        P[f"class_{ds}"] = np.where(P[f"ratio_{ds}"] >= 2, "over-represented",
                                    np.where(P[f"ratio_{ds}"] <= 0.5, "under-represented", "proportional"))
    P["share_prev"] = P.ref / tot_prev
    K = T[T.measure == "incidence"].copy()   # keratinocyte carcinoma vs incidence (prevalence is higher)
    for ds in D:
        K[f"ratio_{ds}_vs_incidence"] = (K[f"img_{ds}"] / (P[f"img_{ds}"].sum() + K[f"img_{ds}"].sum())) / \
            (K.ref / (tot_prev + K.ref.sum()))
    out = {"conditions": P.round(5).to_dict(orient="records"), "keratinocyte_carcinoma": K.round(5).to_dict(orient="records"),
           "total_images_15": {ds: int(P[f"img_{ds}"].sum()) for ds in D}, "total_prevalence_15_mn": float(tot_prev)}
    json.dump(out, open(WORK / "bias/disease_prevalence.json", "w"), indent=1)
    pd.DataFrame(audit).to_csv(WORK / "bias/disease_concepts.csv", index=False)
    cols = ["condition", "ref", "share_prev", "img_ours", "share_img_ours", "ratio_ours", "class_ours",
            "img_derm1m", "share_img_derm1m", "ratio_derm1m", "class_derm1m"]
    print(P[cols].sort_values("ratio_ours").round(3).to_string(index=False))
    print(K[["condition", "ref", "img_ours", "ratio_ours_vs_incidence", "img_derm1m", "ratio_derm1m_vs_incidence"]].round(2).to_string(index=False))
    figure(P, K)


def figure(P, K):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    C1, C2, INK, MUTED, GRID, SURF = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": MUTED,
                         "xtick.color": MUTED, "ytick.color": MUTED, "figure.facecolor": SURF, "axes.facecolor": SURF,
                         "axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": 10,
                         "axes.titleweight": "bold", "axes.titlelocation": "left"})
    P = P.sort_values("ratio_ours")
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    y = np.arange(len(P))
    ax.axvspan(0.5, 2, color=INK, alpha=0.06, lw=0)
    ax.axvline(1, color=MUTED, lw=0.8)
    for ds, col, mk, dy in [("ours", C1, "o", 0.15), ("derm1m", C2, "^", -0.15)]:
        r = P[f"ratio_{ds}"].clip(lower=1 / 256)
        ax.hlines(y + dy, P[f"ratio_{ds}_lo"].clip(lower=1 / 256), P[f"ratio_{ds}_hi"].clip(lower=1 / 256), color=col, lw=1.2, alpha=0.6)
        ax.scatter(r, y + dy, s=34, marker=mk, color=col, edgecolor=SURF, linewidth=1, zorder=3,
                   label={"ours": "Ours", "derm1m": "Derm1M"}[ds])
    ax.set_xscale("log", base=2)
    ax.set_xticks([1 / 64, 1 / 16, 1 / 4, 1 / 2, 1, 2, 4, 16, 64], ["1/64", "1/16", "1/4", "1/2", "1", "2", "4", "16", "64"])
    ax.set_yticks(y, [f"{c}  ({v:,.0f} M)" for c, v in zip(P.condition, P.ref)], fontsize=8)
    ax.set_xlabel("representation ratio = share of images / share of global prevalence (GBD 2019)")
    ax.text(0.5 ** 0.5 * 2 ** 0.5, len(P) - 0.3, "proportional\n(0.5–2)", ha="center", va="bottom", fontsize=7, color=MUTED)
    ax.text(1 / 40, len(P) - 0.3, "under-represented", ha="center", va="bottom", fontsize=7, color=MUTED)
    ax.text(24, len(P) - 0.3, "over-represented", ha="center", va="bottom", fontsize=7, color=MUTED)
    ax.set_ylim(-0.7, len(P) + 0.6)
    ax.grid(axis="x", color=GRID, linewidth=0.6); ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="lower right", fontsize=8)
    ax.set_title("Disease representation vs global prevalence")
    out = WORK.parent / "report/figures/fig6_disease_prevalence"
    fig.savefig(f"{out}.png", dpi=300, bbox_inches="tight")
    P.to_csv(f"{out}.csv", index=False)
    plt.close(fig)


if __name__ == "__main__":
    main()
