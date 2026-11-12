"""Patient sex in our dataset vs Derm1M, overall and per condition, against population references.

Patient sex per photograph (clinical / dermoscopic images only; one unit per near-duplicate cluster, majority vote,
conflicting clusters dropped):
  ours    structured atlas field (incl. French Homme / Femme; "non identifie" or both = unknown); otherwise the
          caption if it names exactly one sex (woman/girl/female/Patientin/Frau ... vs man/boy/male/Patient/Mann ...)
          and contains no sex-ratio phrase, no "male/female pattern" disease name and no animal context.
          (Stricter than 70_bias.py, which counted captions naming both sexes as female.)
  Derm1M  its `gender` field ("male, female" = unknown).
References: world population about 49.7% female (UN WPP 2024: sex ratio about 101 males per 100 females);
per condition, the female share of GBD 2019 prevalent cases (Szeto et al. 2024, Table 1; scripts/reference_prevalence.yaml).
Tests: two-sided exact binomial test of the dataset's female share against the GBD share, Holm-corrected per dataset;
"skewed" = significant and |difference| >= 10 percentage points.
Writes work/bias/sex_distribution.json, work/bias/sex_caption_audit.csv, report/figures/fig7_sex.{png,csv}.
"""
import json
import re

import joblib
import numpy as np
import pandas as pd
import yaml
from scipy.stats import binomtest
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportion_confint

from common import DATASET, WORK

PHOTO = {"clinical", "dermoscopy"}
# language-specific (German "man" is a pronoun: "kann man ... erkennen")
FEM = {"en": re.compile(r"\b(woman|women|female|girl|girls|lady)\b", re.I),
       "de": re.compile(r"\b(patientin|frau|mädchen|weiblich\w*)\b", re.I),
       "fr": re.compile(r"\b(femme|fille|patiente)\b", re.I)}
MAL = {"en": re.compile(r"\b(man|men|male|boy|boys|gentleman)\b", re.I),
       "de": re.compile(r"\b(mann|junge|männlich\w*)\b|\b\d{1,3}-jährige[rn]? patient(en)?\b", re.I),
       "fr": re.compile(r"\b(homme|garçon|patient)\b", re.I)}
NOT_PATIENT = re.compile(r"(fe)?male[- ]pattern|male[- ]to[- ]female|female[- ]to[- ]male|sex ratio|\bratio\b.*\b(fe)?males?\b|"
                         r"\b(mice|mouse|murine|rats?|dogs?|animals?|pigs?|rabbits?|zebrafish)\b", re.I)
WORLD_FEMALE = 0.497


def field_sex(v):
    if not isinstance(v, str):
        return None
    v = v.strip().lower()
    f = v in ("female", "f", "femme", "weiblich", "woman", "w")
    m = v in ("male", "m", "homme", "männlich", "man")
    return "female" if f and not m else "male" if m and not f else None


def caption_sex(c, lang):
    if not isinstance(c, str) or lang not in FEM or NOT_PATIENT.search(c):
        return None
    f, m = bool(FEM[lang].search(c)), bool(MAL[lang].search(c))
    return "female" if f and not m else "male" if m and not f else None


def cluster_vote(df, unit):
    """One sex per unit (majority; ties = unknown)."""
    v = df.dropna(subset=["sex"]).groupby(unit).sex.agg(lambda s: s.mode().iloc[0] if s.value_counts().is_unique or len(s.unique()) == 1 else None)
    return v.dropna()


def ours():
    S = pd.read_parquet(DATASET / "metadata/samples.parquet", columns=["src_path", "sha256", "source_type", "caption", "sex", "language"])
    I = pd.read_parquet(DATASET / "metadata/images.parquet", columns=["sha256", "image_type", "near_dup_cluster"])
    S = S.merge(I, on="sha256")
    S = S[S.image_type.isin(PHOTO)].copy()
    fs = S.sex.map(field_sex)
    cs = pd.Series([caption_sex(c, l) for c, l in zip(S.caption, S.language)], index=S.index)
    S["sex"] = fs.fillna(cs)
    S["sex_from"] = np.where(fs.notna(), "field", np.where(cs.notna(), "caption", None))
    S["unit"] = S.near_dup_cluster.astype(str)
    return S


def derm1m():
    d = pd.read_parquet(WORK / "derm1m.parquet", columns=["src_path", "gender", "source"])
    II = pd.read_parquet(WORK / "image_info.parquet", columns=["src_path", "sha256"])
    DP = pd.read_parquet(WORK / "embed_derm1m.parquet")
    DP["image_type"] = joblib.load(WORK / "heads_models.joblib")["type"].predict(np.load(WORK / "emb/derm1m.npy").astype(np.float32))
    ND = pd.read_parquet(WORK / "compare/derm1m_near_dup.parquet")
    d = d.merge(II, on="src_path").merge(DP[["sha256", "image_type"]], on="sha256").merge(ND, on="sha256")
    d = d[d.image_type.isin(PHOTO)].copy()
    g = d.gender.astype(str).str.strip().str.lower()
    d["sex"] = np.where(g == "female", "female", np.where(g == "male", "male", None))
    d["unit"] = d.near_dup_cluster.astype(str)
    return d


def share(units):
    n, k = len(units), int((units == "female").sum())
    lo, hi = proportion_confint(k, n, method="wilson") if n else (np.nan, np.nan)
    return {"n": n, "female": k, "female_share": k / n if n else None, "ci95": [lo, hi]}


def main():
    ref = yaml.safe_load(open(WORK.parent / "scripts/reference_prevalence.yaml"))["diseases"]["gbd2019_prevalence_mn"]
    gbd_f = yaml.safe_load(open(WORK.parent / "scripts/reference_prevalence.yaml"))["diseases"].get("gbd2019_female_share", {})
    O, D = ours(), derm1m()
    vo, vd = cluster_vote(O, "unit"), cluster_vote(D, "unit")
    res = {"world_female_share": WORLD_FEMALE,
           "ours": {"photo_units": int(O.unit.nunique()), "with_sex": share(vo),
                    "coverage": len(vo) / O.unit.nunique(),
                    "by_source_type": {t: share(cluster_vote(O[O.source_type == t], "unit")) for t in ["atlas", "journal", "textbook"]},
                    "sex_from": O.dropna(subset=["sex"]).drop_duplicates("unit").sex_from.value_counts().to_dict()},
           "derm1m": {"photo_units": int(D.unit.nunique()), "with_sex": share(vd), "coverage": len(vd) / D.unit.nunique(),
                      "by_source": {s: share(cluster_vote(D[D.source == s], "unit")) for s in sorted(D.source.unique())}}}

    # caption audit sample (precision check by reading)
    a = O[O.sex_from == "caption"].drop_duplicates("unit").sample(40, random_state=0)
    a[["unit", "sex", "caption"]].to_csv(WORK / "bias/sex_caption_audit.csv", index=False)

    # per condition (concept matching of 73_disease_prevalence.py)
    dp = importlib_73()
    V = pd.read_parquet(WORK / "dx_vocab.parquet")
    terms_of = V.groupby("concept_id").apply(lambda g: [g.concept_name.iloc[0]] + g.term.tolist(), include_groups=False).to_dict()
    dxo = pd.read_parquet(WORK / "dx_samples_ours.parquet").explode("concepts").dropna(subset=["concepts"])
    dxd = pd.read_parquet(WORK / "dx_samples_derm1m.parquet").explode("concepts").dropna(subset=["concepts"])
    uo = O.drop_duplicates("src_path").set_index("src_path").unit
    ud = D.drop_duplicates("src_path").set_index("src_path").unit
    dxo["unit"], dxd["unit"] = dxo.src_path.map(uo), dxd.src_path.map(ud)
    rows = []
    for cond, spec in ref.items():
        cids = dp.match(spec, terms_of)
        r = {"condition": cond, "gbd_female_share": gbd_f.get(cond)}
        for ds, dx, votes in [("ours", dxo, vo), ("derm1m", dxd, vd)]:
            units = dx[dx.concepts.astype(int).isin(cids)].unit.dropna().unique()
            s = votes.reindex(units).dropna()
            r[ds] = share(s)
            if r["gbd_female_share"] is not None and len(s) >= 10:
                r[ds]["p_vs_gbd"] = binomtest(r[ds]["female"], len(s), r["gbd_female_share"]).pvalue
        rows.append(r)
    for ds in ["ours", "derm1m"]:
        idx = [i for i, r in enumerate(rows) if "p_vs_gbd" in r[ds]]
        if idx:
            adj = multipletests([rows[i][ds]["p_vs_gbd"] for i in idx], method="holm")[1]
            for i, p in zip(idx, adj):
                d = rows[i][ds]["female_share"] - rows[i]["gbd_female_share"]
                rows[i][ds]["p_holm"] = float(p)
                rows[i][ds]["diff_pp"] = 100 * d
                rows[i][ds]["class"] = ("female-skewed" if d >= 0.10 else "male-skewed" if d <= -0.10 else "consistent") \
                    if p < 0.05 else "consistent"
    res["conditions"] = rows
    json.dump(res, open(WORK / "bias/sex_distribution.json", "w"), indent=1, default=float)
    print(json.dumps({k: v for k, v in res.items() if k != "conditions"}, indent=1, default=float))
    for r in rows:
        print(f"{r['condition'][:38]:38s} GBD {100 * r['gbd_female_share']:.1f}% | ours "
              f"{r['ours']['n']:5d} {100 * (r['ours']['female_share'] or 0):5.1f}% {r['ours'].get('class', '-'):14s} | Derm1M "
              f"{r['derm1m']['n']:5d} {100 * (r['derm1m']['female_share'] or 0):5.1f}% {r['derm1m'].get('class', '-')}")
    figure(res)


def importlib_73():
    import importlib
    return importlib.import_module("73_disease_prevalence")


def figure(res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    C1, C2, INK, MUTED, GRID, SURF = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": MUTED,
                         "xtick.color": MUTED, "ytick.color": MUTED, "figure.facecolor": SURF, "axes.facecolor": SURF,
                         "axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": 10,
                         "axes.titleweight": "bold", "axes.titlelocation": "left"})
    rows = [{"condition": "All photographs with known sex", "gbd_female_share": res["world_female_share"],
             "ours": res["ours"]["with_sex"], "derm1m": res["derm1m"]["with_sex"]}] + \
        sorted(res["conditions"], key=lambda r: r["gbd_female_share"])
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    y = np.arange(len(rows))[::-1]
    for yi, r in zip(y, rows):
        ax.scatter([100 * r["gbd_female_share"]], [yi], marker="|", s=220, color=INK, linewidths=2, zorder=4,
                   label="Reference (GBD 2019 prevalence; top row: world population)" if yi == y[0] else None)
        for ds, col, mk, dy in [("ours", C1, "o", 0.17), ("derm1m", C2, "^", -0.17)]:
            s = r[ds]
            if not s["n"]:
                continue
            ax.hlines(yi + dy, 100 * s["ci95"][0], 100 * s["ci95"][1], color=col, lw=1.2, alpha=0.6)
            ax.scatter([100 * s["female_share"]], [yi + dy], marker=mk, s=30, color=col, edgecolor=SURF, linewidth=0.8, zorder=3,
                       label={"ours": "Ours", "derm1m": "Derm1M"}[ds] if yi == y[0] else None)
    ax.axvline(50, color=GRID, lw=1)
    ax.set_yticks(y, [f"{r['condition']}  (n = {r['ours']['n']:,} / {r['derm1m']['n']:,})" for r in rows], fontsize=7.5)
    ax.set_xlim(0, 100); ax.set_xlabel("female share of photographs with known patient sex (%), 95% CI")
    ax.grid(axis="x", color=GRID, linewidth=0.6); ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=7.5, loc="upper left", bbox_to_anchor=(0, -0.09), ncol=2)
    ax.set_title("Patient sex: datasets vs reference")
    out = WORK.parent / "report/figures/fig7_sex"
    fig.savefig(f"{out}.png", dpi=300, bbox_inches="tight")
    pd.DataFrame([{"condition": r["condition"], "reference_female": r["gbd_female_share"],
                   "ours_n": r["ours"]["n"], "ours_female": r["ours"]["female_share"],
                   "derm1m_n": r["derm1m"]["n"], "derm1m_female": r["derm1m"]["female_share"]} for r in rows]).to_csv(f"{out}.csv", index=False)
    plt.close(fig)


if __name__ == "__main__":
    main()
