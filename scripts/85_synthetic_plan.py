"""Generation plan for the synthetic dermatology image dataset (last step before the report).

Builds the conditioning metadata a text-to-image / image-editing pipeline needs, aimed at the gaps found in
the bias analysis (few Fitzpatrick V-VI images, rare diagnoses):

1. Real reference counts per diagnosis x skin-type group, from atlas-labelled images only (the atlas
   diagnosis is a curated label; caption-mined concepts also contain morphology terms). Images: canonical,
   clinical photographs, train split. Counts are near-duplicate clusters (effective number of photographs).
   Skin type = CLIP Fitzpatrick head (fst_pred; under-detects dark skin, recall 0.63).
2. Allocation: every diagnosis with >= MIN_REF clusters gets each skin-type group topped up to
   T = clip(n/3, 30, 200). Cells without any real image in that group are flagged (pure extrapolation,
   100% expert review).
3. One row per image to generate: diagnosis, target Fitzpatrick type, body site (sampled from the real
   body-site distribution of that diagnosis), prompt, negative prompt, seed, and candidate reference images
   for the editing track (train split only; never validation/test, so downstream tests stay clean).

Writes dataset/synthetic/plan/{conditioning_table.csv, prompts.parquet, prompts.jsonl, references.parquet,
plan_summary.json}. Protocol: dataset/docs/SYNTHETIC_PROTOCOL.md; quality gate: 86_eval_synthetic.py.
"""
import json
import re

import numpy as np
import pandas as pd

from common import DATASET, WORK

OUT = DATASET / "synthetic/plan"
MIN_REF, T_MIN, T_MAX = 10, 30, 200
GROUPS = {"I-II": (1, 2), "III-IV": (3, 4), "V-VI": (5, 6)}
FST_WORDS = {1: "very fair, pale white", 2: "fair, white", 3: "light brown", 4: "moderate brown",
             5: "dark brown", 6: "deeply pigmented dark brown to black"}
ROMAN = {1: "I", 2: "II", 3: "III", 4: "IV", 5: "V", 6: "VI"}
# generic or descriptive labels that are not diagnoses (in addition to 53_diagnosis_link.GENERIC)
NOT_DX = {"proliferations", "nail disease", "skin diseases caused by warts", "skin cancer", "skin infection",
          "systemic disease", "vascular", "blister", "papules", "plaques", "erythematous", "trunk", "exanthem",
          "pigmentation", "hyperpigmentation", "hypopigmentation", "pruritus", "purpura", "alopecia", "eczema",
          "nevus", "warts", "cyst", "tumor", "benign tumor", "drug eruption", "genodermatosis", "infection"}
# coarse body-site vocabulary (first matching keyword wins; order matters)
SITES = [("scalp", ["scalp", "hairy", "hair"]), ("nail", ["nail", "ungu", "onych"]),
         ("lip", ["lip"]), ("oral mucosa", ["oral mucosa", "oral cavity", "tongue", "gingiv", "palate", "buccal", "mucosa"]),
         ("eyelid", ["eyelid", "periorbital", "canth", "eyelash"]), ("ear", ["ear"]), ("nose", ["nose", "nasal"]),
         ("face", ["face", "facial", "cheek", "forehead", "chin", "temple", "perioral", "jaw", "eye", "mouth", "brow"]),
         ("neck", ["neck", "nuchal"]), ("palm", ["palm"]), ("sole", ["sole", "plantar"]),
         ("finger", ["finger", "thumb", "digit"]), ("toe", ["toe"]), ("hand", ["hand", "wrist"]),
         ("foot", ["foot", "feet", "ankle", "heel"]), ("axilla", ["axill", "armpit"]),
         ("genital area", ["genital", "penis", "scrot", "vulva", "labia", "pubic", "perian", "anal"]),
         ("groin", ["groin", "inguinal"]), ("buttocks", ["butt", "gluteal"]),
         ("arm", ["arm", "elbow", "forearm", "shoulder", "extremit"]),
         ("leg", ["leg", "thigh", "knee", "shin", "calf", "lower limb"]),
         ("back", ["back", "lumbar", "sacr"]), ("chest", ["chest", "breast", "thorax", "sternal"]),
         ("abdomen", ["abdom", "umbilic", "flank", "belly"]), ("trunk", ["trunk", "torso"])]


def site_of(text):
    t = text.lower()
    for name, keys in SITES:
        if any(re.search(r"\b" + k, t) for k in keys):
            return name
    return None


def sites_for_sample(body_site, llm_parts):
    out = []
    if isinstance(body_site, str):
        s = site_of(body_site.replace("-", " "))
        if s:
            out.append(s)
    if not out and isinstance(llm_parts, str):
        try:
            parts = json.loads(llm_parts)
        except ValueError:
            parts = []
        out = [s for s in dict.fromkeys(site_of(p) for p in parts if isinstance(p, str)) if s]
    return out


def main():
    gen = importlib_generic()
    S = pd.read_parquet(DATASET / "metadata/samples.parquet",
                        columns=["src_path", "sha256", "source", "source_type", "body_site", "llm_vision_body_parts_seen",
                                 "license", "release_tier", "terms_restrict_ai"])
    I = pd.read_parquet(DATASET / "metadata/images.parquet",
                        columns=["sha256", "image_id", "file", "image_type", "is_canonical", "split", "fst_pred",
                                 "near_dup_cluster", "subset", "bench_overlap"])
    dx = pd.read_parquet(WORK / "dx_samples_ours.parquet").explode("concepts").dropna(subset=["concepts"])
    V = pd.read_parquet(WORK / "dx_vocab.parquet").drop_duplicates("concept_id").set_index("concept_id")
    dx = dx.merge(S[S.source_type == "atlas"], on=["src_path", "source"])
    dx = dx.merge(I, on="sha256")
    dx = dx[(dx.image_type == "clinical") & dx.is_canonical & (dx.split == "train") & dx.bench_overlap.isna()]
    dx["concept_name"] = dx.concepts.map(V.concept_name)
    dx = dx[~dx.concept_name.isin(NOT_DX | gen)]
    dx["fst_group"] = pd.cut(dx.fst_pred, [0, 2, 4, 6], labels=list(GROUPS)).astype(str)
    dx["sites"] = [sites_for_sample(b, l) for b, l in zip(dx.body_site, dx.llm_vision_body_parts_seen)]
    print("atlas-labelled clinical train images:", dx.sha256.nunique(), "concepts:", dx.concepts.nunique())

    # 1. conditioning table (effective n = distinct near-duplicate clusters)
    eff = dx.drop_duplicates(["concepts", "near_dup_cluster"])
    tab = eff.groupby(["concepts", "fst_group"]).size().unstack(fill_value=0).reindex(columns=list(GROUPS), fill_value=0)
    tab["n"] = tab.sum(axis=1)
    tab["concept_name"] = tab.index.map(V.concept_name)
    tab["ontology_node"] = tab.index.map(V.concept_ontology_node)
    tab["n_sources"] = eff.groupby("concepts").source.nunique()
    tab = tab[tab.n >= MIN_REF].sort_values("n", ascending=False)
    tab["target_per_group"] = np.clip(np.ceil(tab.n / 3), T_MIN, T_MAX).astype(int)
    for g in GROUPS:
        tab[f"gen_{g}"] = (tab.target_per_group - tab[g]).clip(lower=0)
    tab["rare"] = tab.n < 30
    tab["no_real_V-VI"] = tab["V-VI"] == 0
    site_dist = (dx.explode("sites").dropna(subset=["sites"]).drop_duplicates(["concepts", "near_dup_cluster", "sites"])
                 .groupby("concepts").sites.value_counts(normalize=True))
    tab["top_sites"] = [", ".join(f"{s} {p:.0%}" for s, p in site_dist.get(c, pd.Series(dtype=float)).head(4).items())
                        for c in tab.index]

    # 2. one row per image to generate
    rng = np.random.default_rng(0)
    rows = []
    for c, r in tab.iterrows():
        sd = site_dist.get(c)
        for g, (lo, hi) in GROUPS.items():
            for k in range(int(r[f"gen_{g}"])):
                fst = lo + k % 2
                site = rng.choice(sd.index.values, p=sd.values / sd.values.sum()) if sd is not None and len(sd) else None
                where = f" on the {site}" if site else ""
                rows.append(dict(
                    concept_id=int(c), diagnosis=r.concept_name, fst_target=fst, fst_group=g, body_site=site,
                    prompt=(f"Clinical photograph of {r.concept_name}{where} of a patient with Fitzpatrick skin type "
                            f"{ROMAN[fst]} ({FST_WORDS[fst]} skin), dermatology atlas style, close-up, natural lighting, "
                            f"sharp focus"),
                    negative_prompt="text, watermark, logo, arrows, ruler, frame, collage, illustration, drawing, "
                                    "cartoon, deformed anatomy, extra fingers",
                    rare=bool(r.rare), extrapolation=bool(r[g] == 0), seed=int(rng.integers(2**31 - 1))))
    P = pd.DataFrame(rows)
    # generation order: 1 = common diagnosis (>= 30 clusters) with real references in the target group,
    # 2 = common diagnosis, no real image in the target group (extrapolation), 3 = rare diagnosis
    P["priority"] = np.where(P.rare, 3, np.where(P.extrapolation, 2, 1))
    P = P.sort_values(["priority", "fst_group", "concept_id"], ascending=[True, False, True], kind="stable")
    P.insert(0, "synth_id", [f"syn_{i:06d}" for i in range(len(P))])

    # 3. reference pool for the editing / reference-conditioned track (one image per near-dup cluster)
    ref = eff[eff.concepts.isin(tab.index)].copy()
    ref["derivative_publishable"] = ref.license.str.startswith("CC BY") & ~ref.license.str.contains("ND") & \
        ~ref.terms_restrict_ai.fillna(False).astype(bool)
    ref = ref[["concepts", "concept_name", "image_id", "sha256", "file", "source", "fst_pred", "fst_group", "sites",
               "license", "release_tier", "derivative_publishable"]].rename(columns={"concepts": "concept_id"})
    ref["sites"] = ref.sites.map(lambda x: ", ".join(x))

    OUT.mkdir(parents=True, exist_ok=True)
    tab.reset_index(names="concept_id").to_csv(OUT / "conditioning_table.csv", index=False)
    P.to_parquet(OUT / "prompts.parquet", index=False)
    P.to_json(OUT / "prompts.jsonl", orient="records", lines=True, force_ascii=False)
    ref.to_parquet(OUT / "references.parquet", index=False)
    summ = dict(
        diagnoses=len(tab), rare_diagnoses=int(tab.rare.sum()), diagnoses_without_real_V_VI=int(tab["no_real_V-VI"].sum()),
        real_clusters_by_group={g: int(tab[g].sum()) for g in GROUPS},
        images_to_generate=len(P), by_group=P.fst_group.value_counts().reindex(list(GROUPS)).to_dict(),
        extrapolation_images=int(P.extrapolation.sum()),
        by_priority=P.priority.value_counts().sort_index().to_dict(), with_body_site=float(P.body_site.notna().mean()),
        reference_images=len(ref), references_derivative_publishable=int(ref.derivative_publishable.sum()),
        params=dict(MIN_REF=MIN_REF, T_MIN=T_MIN, T_MAX=T_MAX, groups=GROUPS))
    json.dump(summ, open(OUT / "plan_summary.json", "w"), indent=1)
    print(json.dumps(summ, indent=1))
    print(tab.head(25)[["concept_name", "n", "I-II", "III-IV", "V-VI", "gen_V-VI", "top_sites"]].to_string())


def importlib_generic():
    """GENERIC label set of 53_diagnosis_link.py (module name starts with a digit)."""
    import ast
    from pathlib import Path
    tree = ast.parse((Path(__file__).parent / "53_diagnosis_link.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", None) == "GENERIC":
            return set(ast.literal_eval(node.value))
    return set()


if __name__ == "__main__":
    main()
