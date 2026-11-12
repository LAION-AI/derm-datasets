"""Aim 3: distribution and balance analysis (skin type, sex, age, disease x skin type).

* Fitzpatrick distribution of clinical photos (CLIP head estimate; canonical images), overall,
  per source, and compared with Derm1M and with the expert-labelled Fitzpatrick17k.
  Balance metrics: share FST V-VI, normalised Shannon entropy over 6 types (1 = uniform),
  imbalance ratio (largest / smallest type).
* Age / sex: structured fields (atlases) plus regular-expression extraction from case-description
  captions in English and German ("a 45-year-old woman", "62-jährige Patientin").
* Disease x skin type: share of FST V-VI images per frequent disease concept.

Writes work/bias/bias.json, work/bias/*.parquet.
"""
import json
import re

import numpy as np
import pandas as pd

from common import DATASET, WORK

OUT = WORK / "bias"
OUT.mkdir(exist_ok=True)

AGE_EN = re.compile(r"\b(\d{1,3})[- ](?:year|yr)s?[- ]old\b", re.I)
AGE_DE = re.compile(r"\b(\d{1,3})[- ]?(?:jährige[nmrs]?|jähriger|jährigen)\b", re.I)
FEM = re.compile(r"\b(woman|women|female|girl|lady|patientin|frau|mädchen|weiblich)\b", re.I)
# English "patient" carries no sex; German "-jähriger Patient" is masculine
MAL = re.compile(r"\b(man|men|male|boy|gentleman|mann|junge|männlich)\b|\b\d{1,3}-jährige[rn]? patient\b", re.I)


def balance(counts):
    c = pd.Series(counts).reindex([1, 2, 3, 4, 5, 6]).fillna(0).astype(float)
    p = c / c.sum()
    nz = p[p > 0]
    return {"n": int(c.sum()), "dist_pct": (100 * p).round(1).to_dict(),
            "V_VI_pct": round(100 * float(p[[5, 6]].sum()), 1),
            "entropy_norm": round(float(-(nz * np.log(nz)).sum() / np.log(6)), 3),
            "imbalance_ratio": round(float(c.max() / max(1, c.min())), 1)}


def norm_sex(v):
    if not isinstance(v, str):
        return None
    v = v.strip().lower()
    if v in ("female", "f", "femme", "weiblich", "woman", "w") or "female" in v or "femme" in v:
        return "female"
    if v in ("male", "m", "homme", "männlich", "man") or "male" in v or "homme" in v:
        return "male"
    return None


def main():
    R = {}
    S = pd.read_parquet(DATASET / "metadata/samples.parquet")
    I = pd.read_parquet(DATASET / "metadata/images.parquet")
    clin = I[(I.image_type == "clinical") & I.is_canonical]
    R["fst_ours_clinical"] = balance(clin.fst_pred.astype(int).value_counts().to_dict())
    src = S.drop_duplicates("image_id").set_index("image_id").source
    clin = clin.assign(source=clin.image_id.map(src))
    R["fst_by_source"] = {s: balance(g.fst_pred.astype(int).value_counts().to_dict())
                          for s, g in clin.groupby("source") if len(g) >= 200}
    gold = I.dropna(subset=["fst_gold_f17k"])
    R["fst_gold_f17k_subset"] = balance(gold.fst_gold_f17k.astype(int).value_counts().to_dict())
    B = pd.read_parquet(WORK / "bench.parquet")
    R["fst_fitzpatrick17k_benchmark"] = balance(B[(B.bench == "F17K") & B.fst.notna()].drop_duplicates("path")
                                                .fst.astype(int).value_counts().to_dict())
    cmp = WORK / "compare/derm1m.json"
    if cmp.exists():
        d = json.load(open(cmp))["A_derm1m"]["fst_dist_clinical"]
        R["fst_derm1m_clinical"] = balance({int(k): v for k, v in d.items()})
    # LLM (Claude 3 Opus) estimates vs expert labels where both exist
    both = S.dropna(subset=["fst_llm", "fst_gold_f17k"]).drop_duplicates("image_id")
    if len(both):
        e = both.fst_llm - both.fst_gold_f17k
        R["fst_llm_vs_gold"] = {"n": len(both), "exact": round(float((e == 0).mean()), 3),
                                "within1": round(float((e.abs() <= 1).mean()), 3), "mae": round(float(e.abs().mean()), 3),
                                "mean_bias": round(float(e.mean()), 3)}
    # CLIP head vs expert labels on our images (note: F17K images were in the head's training data)
    hb = S.dropna(subset=["fst_pred", "fst_gold_f17k"]).drop_duplicates("image_id")
    e = hb.fst_pred - hb.fst_gold_f17k
    R["fst_head_vs_gold_in_sample"] = {"n": len(hb), "exact": round(float((e == 0).mean()), 3),
                                       "within1": round(float((e.abs() <= 1).mean()), 3),
                                       "note": "in-sample (F17K images were training data); see heads_report.json for CV"}

    # ---------------- age / sex
    T = S.drop_duplicates("image_id").copy()
    cap = T.caption.fillna("")
    age_txt = cap.str.extract(AGE_EN)[0].fillna(cap.str.extract(AGE_DE)[0]).astype(float)
    T["age_years"] = age_txt.where((age_txt > 0) & (age_txt < 110))
    sex_txt = np.where(cap.str.contains(FEM), "female", np.where(cap.str.contains(MAL), "male", None))
    T["sex_norm"] = [norm_sex(v) or s for v, s in zip(T.sex, sex_txt)]
    has_case = T.age_years.notna() | T.sex_norm.notna()
    R["demographics_images_with_age"] = int(T.age_years.notna().sum())
    R["demographics_images_with_sex"] = int(T.sex_norm.notna().sum())
    R["sex_dist"] = T.sex_norm.value_counts().to_dict()
    bins = [0, 2, 12, 18, 40, 65, 120]
    labels = ["0-1", "2-11", "12-17", "18-39", "40-64", "65+"]
    R["age_dist"] = pd.cut(T.age_years, bins=bins, labels=labels, right=False).value_counts().reindex(labels).to_dict()
    R["age_median"] = float(T.age_years.median())
    R["age_sex_by_source_type"] = T[has_case].groupby("source_type").agg(
        with_age=("age_years", "count"), female=("sex_norm", lambda x: int((x == "female").sum())),
        male=("sex_norm", lambda x: int((x == "male").sum()))).to_dict(orient="index")
    T[["image_id", "source", "age_years", "sex_norm"]].to_parquet(OUT / "demographics.parquet", index=False)

    # ---------------- disease x skin type
    dx = pd.read_parquet(WORK / "dx_samples_ours.parquet").explode("concepts").dropna(subset=["concepts"])
    V = pd.read_parquet(WORK / "dx_vocab.parquet").drop_duplicates("concept_id").set_index("concept_id").concept_name
    sha = pd.read_parquet(WORK / "image_info.parquet").set_index("src_path").sha256
    dx["sha256"] = dx.src_path.map(sha)
    dx = dx.merge(clin[["sha256", "fst_pred"]], on="sha256").drop_duplicates(["sha256", "concepts"])
    g = dx.groupby("concepts").agg(n=("sha256", "size"), dark=("fst_pred", lambda x: (x >= 5).mean()))
    g = g[g.n >= 100].sort_values("n", ascending=False)
    g["name"] = g.index.map(V)
    R["disease_fst_top"] = g.head(40)[["name", "n", "dark"]].round(3).to_dict(orient="records")
    R["diseases_ge100_with_zero_dark"] = g[g.dark == 0].name.tolist()
    g.to_parquet(OUT / "disease_by_fst.parquet")
    json.dump(R, open(OUT / "bias.json", "w"), indent=1, default=str)
    print(json.dumps({k: v for k, v in R.items() if k not in ("disease_fst_top", "fst_by_source")}, indent=1, default=str))


if __name__ == "__main__":
    main()
