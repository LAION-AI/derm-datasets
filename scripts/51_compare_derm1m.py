"""Compare our dataset with Derm1M (HF redlessone/Derm1M v2, 423,016 rows incl. validation).

A  composition: images, pairs, near-duplicate redundancy, image types, skin types, metadata
   fill rates, caption length, resolution, languages
B  image overlap in both directions: exact (SHA-256) and near-duplicate (CLIP cos >= 0.98, or
   >= 0.95 with pHash distance <= 10 — the same rule as in 22_dedup.py)
C  diagnosis coverage via SapBERT concepts (53_diagnosis_link.py); concepts that only our data
   covers are re-checked against the full Derm1M caption text (not only its labels)
D  our "unique contribution": dermatology images with no near-duplicate in Derm1M

Writes work/compare/derm1m.json and work/compare/*.parquet.
"""
import json
import re
import subprocess
import sys

import imagehash
import joblib
import numpy as np
import pandas as pd

from common import DATASET, WORK, load_local_embeddings

OUT = WORK / "compare"
OUT.mkdir(exist_ok=True)
PY = sys.executable
CORE = {"clinical", "dermoscopy", "histopathology"}


def entropy_norm(counts):
    p = np.asarray(list(counts), float)
    p = p[p > 0] / p.sum()
    return float(-(p * np.log(p)).sum() / np.log(6))   # normalised over 6 Fitzpatrick types


def nn(query, base, out, k=5, min_sim=0.95, self_=False):
    args = [PY, "21_neighbors.py", str(query), str(base), str(out), "--k", str(k), "--min-sim", str(min_sim)]
    if self_:
        args.append("--self")
    if not out.exists():
        subprocess.run(args, check=True)
    return pd.read_parquet(out)


def main():
    R = {}
    info = pd.read_parquet(WORK / "image_info.parquet").drop_duplicates("sha256").set_index("sha256")
    heads = joblib.load(WORK / "heads_models.joblib")

    # ---------------- Derm1M side
    d1 = pd.read_parquet(WORK / "derm1m.parquet")
    ii = pd.read_parquet(WORK / "image_info.parquet", columns=["src_path", "sha256", "width", "height"])
    d1 = d1.merge(ii, on="src_path", how="left")
    DP = pd.read_parquet(WORK / "embed_derm1m.parquet")
    DE = np.load(WORK / "emb/derm1m.npy").astype(np.float32)
    dtype = heads["type"].predict(DE)
    dfst = np.clip(np.rint(heads["fst"].predict(DE)), 1, 6)
    dimg = DP.assign(image_type=dtype, fst_pred=dfst)
    d1 = d1.merge(dimg[["sha256", "image_type", "fst_pred"]], on="sha256", how="left")

    # ---------------- our side
    S = pd.read_parquet(DATASET / "metadata/samples.parquet")
    I = pd.read_parquet(DATASET / "metadata/images.parquet")
    LP, LE = load_local_embeddings()
    LE = LE.astype(np.float32)

    # ---------------- A: composition
    def no_info(s):
        return s.fillna("").str.startswith("No ") | s.isna()
    d1_dx = ~d1.disease_label.fillna("").str.lower().isin(["", "no definitive diagnosis"])
    nd_self = nn(WORK / "emb/derm1m.npy", WORK / "emb/derm1m.npy", OUT / "nn_derm1m_self.parquet", k=10, self_=True)
    ph = info.phash.to_dict()
    def near_dup_clusters(nnf, P):
        nnf = nnf[nnf.q < nnf.b]
        hp = [imagehash.hex_to_hash(ph[h]) for h in P.sha256]
        parent = np.arange(len(P))
        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]; a = parent[a]
            return a
        for a, b, s in zip(nnf.q.values, nnf.b.values, nnf.sim.values):
            if s >= 0.98 or (hp[a] - hp[b]) <= 10:
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[ra] = rb
        return np.array([find(i) for i in range(len(P))])
    d1_clusters = near_dup_clusters(nd_self, DP)
    dimg["near_dup_cluster"] = d1_clusters
    dimg[["sha256", "near_dup_cluster"]].to_parquet(OUT / "derm1m_near_dup.parquet", index=False)

    def comp(name, n_pairs, imgs, fst, caption_words, dx, age, sex, site, lang, long_side):
        t = pd.Series(imgs.image_type).value_counts()
        core = imgs.image_type.isin(CORE)
        fc = pd.Series(fst).value_counts().reindex([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).fillna(0)
        return {
            "dataset": name, "pairs": int(n_pairs), "unique_images": int(len(imgs)),
            "near_dup_clusters": int(imgs.near_dup_cluster.nunique()),
            "redundancy_pct": round(100 * (1 - imgs.near_dup_cluster.nunique() / len(imgs)), 1),
            "derm_core_images": int(core.sum()),
            "derm_core_clusters": int(imgs[core].near_dup_cluster.nunique()),
            "image_types": {k: int(v) for k, v in t.items()},
            "fst_dist_clinical": {int(k): int(v) for k, v in fc.items()},
            "fst_V_VI_pct": round(100 * fc[[5.0, 6.0]].sum() / fc.sum(), 1),
            "fst_entropy_norm": round(entropy_norm(fc.values), 3),
            "caption_words_mean": round(float(np.mean(caption_words)), 1),
            "caption_words_median": float(np.median(caption_words)),
            "pairs_with_diagnosis_pct": round(100 * float(np.mean(dx)), 1),
            "pairs_with_age_pct": round(100 * float(np.mean(age)), 1),
            "pairs_with_sex_pct": round(100 * float(np.mean(sex)), 1),
            "pairs_with_body_site_pct": round(100 * float(np.mean(site)), 1),
            "languages": lang, "median_long_side_px": int(np.median(long_side)),
        }
    clin_d = dimg.image_type == "clinical"
    R["A_derm1m"] = comp(
        "Derm1M v2", len(d1), dimg, dimg.fst_pred[clin_d], d1.caption.fillna("").str.split().str.len(),
        d1_dx, ~no_info(d1.age), ~no_info(d1.gender), ~no_info(d1.body_location),
        {"en (incl. translated zh)": len(d1)},
        np.maximum(d1.width.fillna(0), d1.height.fillna(0)))
    # "has a diagnosis": source label, or a recognised disease concept in the (English) caption —
    # the same notion as Derm1M's LLM-extracted disease_label != "no definitive diagnosis"
    so_ = pd.read_parquet(WORK / "dx_samples_ours.parquet")
    with_concept = set(so_[so_.concepts.str.len() > 0].src_path)
    ours_dx = S.diagnosis.notna() | S.src_path.isin(with_concept)
    oi = I
    clin_o = oi.image_type == "clinical"
    R["A_ours"] = comp(
        "ours", len(S), oi, oi.fst_pred[clin_o], S.text_words.fillna(0),
        ours_dx, S.age.notna(), S.sex.notna(), S.body_site.notna(),
        S.language.value_counts().to_dict(), np.maximum(oi.width, oi.height))
    R["A_derm1m"]["sources"] = d1.source.value_counts().to_dict()
    R["A_ours"]["sources"] = S.source_type.value_counts().to_dict()
    R["A_derm1m_disease_label_no_definitive_pct"] = round(
        100 * float((d1.disease_label.str.lower() == "no definitive diagnosis").mean()), 1)
    R["A_derm1m_top_labels"] = d1.disease_label.str.lower().value_counts().head(10).to_dict()
    # sanity check of FST head on Derm1M "public" clinical images is not possible (no labels); report only

    # ---------------- B: overlap
    o_sha, d_sha = set(I.sha256), set(DP.sha256)
    R["B_exact_sha_overlap_images"] = len(o_sha & d_sha)
    lp_pos = {h: i for i, h in enumerate(LP.sha256)}
    canon = I[I.is_canonical]
    q_idx = np.array([lp_pos[h] for h in canon.sha256])
    np.save(OUT / "ours_canonical.npy", LE[q_idx].astype(np.float16))
    o2d = nn(OUT / "ours_canonical.npy", WORK / "emb/derm1m.npy", OUT / "nn_ours_to_derm1m.parquet", k=3)
    d2o = nn(WORK / "emb/derm1m.npy", OUT / "ours_canonical.npy", OUT / "nn_derm1m_to_ours.parquet", k=3)

    def dup_rule(df, qsha, bsha):
        h = [imagehash.hex_to_hash(ph[qsha[a]]) - imagehash.hex_to_hash(ph[bsha[b]]) for a, b in zip(df.q, df.b)]
        return df.assign(ham=h, dup=(df.sim >= 0.98) | (np.array(h) <= 10))
    o2d = dup_rule(o2d, canon.sha256.values, DP.sha256.values)
    d2o = dup_rule(d2o, DP.sha256.values, canon.sha256.values)
    canon = canon.assign(in_derm1m=False, best_sim=0.0)
    hit = o2d[o2d.dup].q.unique()
    canon.loc[canon.index[hit], "in_derm1m"] = True
    bs = o2d.groupby("q").sim.max()
    canon.loc[canon.index[bs.index], "best_sim"] = bs.values
    src_of = S.drop_duplicates("sha256").set_index("sha256").source
    canon["source"] = canon.sha256.map(src_of)
    per_src = canon.groupby("source").agg(images=("sha256", "size"), in_derm1m=("in_derm1m", "mean"),
                                          sim095=("best_sim", lambda x: (x >= 0.95).mean()))
    R["B_ours_in_derm1m_pct"] = round(100 * float(canon.in_derm1m.mean()), 2)
    R["B_ours_in_derm1m_by_source_pct"] = (100 * per_src.in_derm1m).round(2).to_dict()
    dimg["in_ours"] = False
    dimg.loc[d2o[d2o.dup].q.unique(), "in_ours"] = True
    d1src = d1.drop_duplicates("sha256").set_index("sha256").source
    dimg["source"] = dimg.sha256.map(d1src)
    R["B_derm1m_in_ours_pct"] = round(100 * float(dimg.in_ours.mean()), 2)
    R["B_derm1m_in_ours_by_source_pct"] = (100 * dimg.groupby("source").in_ours.mean()).round(2).to_dict()
    canon[["sha256", "source", "image_type", "in_derm1m", "best_sim"]].to_parquet(OUT / "ours_vs_derm1m.parquet", index=False)

    # ---------------- C: diagnosis coverage
    V = pd.read_parquet(WORK / "dx_vocab.parquet")
    so = pd.read_parquet(WORK / "dx_samples_ours.parquet")
    sd = pd.read_parquet(WORK / "dx_samples_derm1m.parquet")
    sha_of = pd.read_parquet(WORK / "image_info.parquet").set_index("src_path").sha256
    def concept_images(df):
        df = df.assign(sha256=df.src_path.map(sha_of)).explode("concepts").dropna(subset=["concepts"])
        return df.groupby("concepts").sha256.nunique()
    co, cd = concept_images(so), concept_images(sd)
    cov = pd.DataFrame({"ours": co, "derm1m": cd}).fillna(0).astype(int)
    for k in (1, 10, 50):
        R[f"C_concepts_ge{k}"] = {"ours": int((cov.ours >= k).sum()), "derm1m": int((cov.derm1m >= k).sum()),
                                  "both": int(((cov.ours >= k) & (cov.derm1m >= k)).sum())}
    node_of = V.drop_duplicates("concept_id").set_index("concept_id").concept_ontology_node
    cov["onto"] = cov.index.map(node_of)
    on = cov.dropna(subset=["onto"]).groupby("onto")[["ours", "derm1m"]].sum()
    R["C_ontology_nodes_total"] = int(len([k for k in json.load(open(WORK.parent / "external/Derm1M-hf/ontology.json")) if k != "root"]))
    for k in (1, 10):
        R[f"C_ontology_nodes_ge{k}"] = {"ours": int((on.ours >= k).sum()), "derm1m": int((on.derm1m >= k).sum())}
    # concepts exclusive to ours (>=10 images, 0 Derm1M images) -> check Derm1M caption text
    name = V.drop_duplicates("concept_id").set_index("concept_id").concept_name
    excl = cov[(cov.ours >= 10) & (cov.derm1m == 0)].copy()
    terms = V[V.concept_id.isin(excl.index)].groupby("concept_id").term.apply(list)
    allterms = {t: c for c, ts in terms.items() for t in ts if len(t) >= 5}
    maxw = max(len(t.split()) for t in allterms) if allterms else 1
    tok = re.compile(r"[a-z0-9'\-]+")
    found = set()
    for cap in d1.caption.dropna().str.lower():
        w = tok.findall(cap)
        for i in range(len(w)):
            for n in range(1, min(maxw, len(w) - i) + 1):
                p = " ".join(w[i:i + n])
                if p in allterms:
                    found.add(allterms[p])
    excl["name"] = excl.index.map(name)
    excl["mentioned_in_derm1m_captions"] = excl.index.isin(found)
    # near-synonym check against every term of concepts Derm1M does have (SapBERT cos >= 0.90 or
    # spelling ratio >= 88): catches synonyms/variants the strict concept clustering kept apart
    from rapidfuzz import fuzz
    VE = np.load(WORK / "dx_vocab_sapbert.npy")
    d1_terms = np.where(V.concept_id.isin(cov.index[cov.derm1m > 0]).values)[0]
    lead_idx = {c: V.index[(V.concept_id == c) & (V.term == name[c])][0] for c in excl.index}
    near, near_term = [], []
    for c in excl.index:
        sims = VE[d1_terms] @ VE[lead_idx[c]]
        top = d1_terms[np.argsort(-sims)[:20]]
        best = max(top, key=lambda t: max(float(VE[t] @ VE[lead_idx[c]]) / 0.90, fuzz.ratio(V.term[t], name[c]) / 88))
        ok = float(VE[best] @ VE[lead_idx[c]]) >= 0.90 or fuzz.ratio(V.term[best], name[c]) >= 88
        near.append(ok); near_term.append(V.term[best])
    excl["near_derm1m_term"] = near
    excl["closest_derm1m_term"] = near_term
    excl["exclusive_strict"] = ~excl.mentioned_in_derm1m_captions & ~excl.near_derm1m_term
    excl = excl.sort_values("ours", ascending=False)
    excl.to_parquet(OUT / "concepts_exclusive_ours.parquet")
    R["C_exclusive_ours_ge10"] = int(len(excl))
    R["C_exclusive_ours_ge10_not_in_captions"] = int((~excl.mentioned_in_derm1m_captions).sum())
    R["C_exclusive_ours_ge10_strict"] = int(excl.exclusive_strict.sum())
    R["C_exclusive_ours_strict_images"] = int(excl[excl.exclusive_strict].ours.sum())
    R["C_exclusive_ours_examples"] = excl[excl.exclusive_strict].head(60)[["name", "ours", "closest_derm1m_term"]].to_dict(orient="records")
    excl_d = cov[(cov.derm1m >= 10) & (cov.ours == 0)]
    R["C_exclusive_derm1m_ge10"] = int(len(excl_d))
    R["C_exclusive_derm1m_examples"] = [{"name": name[c], "derm1m": int(v)} for c, v in excl_d.derm1m.sort_values(ascending=False).head(25).items()]
    cov.assign(name=cov.index.map(name)).to_parquet(OUT / "concept_coverage.parquet")

    # ---------------- D: unique contribution
    core_c = canon.image_type.isin(CORE)
    R["D_unique_derm_core_canonical_not_in_derm1m"] = int((core_c & ~canon.in_derm1m).sum())
    R["D_unique_by_source"] = canon[core_c & ~canon.in_derm1m].source.value_counts().to_dict()

    # ---------------- E: Derm1M train/validation leakage (lower bound: k=10 neighbours per image)
    sp = d1.groupby("sha256").split.agg(lambda x: "both" if x.nunique() > 1 else x.iloc[0])
    split = DP.sha256.map(sp).values
    e = nd_self[(split[nd_self.q.values] == "validation") & (split[nd_self.b.values] == "train")]
    hd = [imagehash.hex_to_hash(ph[DP.sha256[x]]) - imagehash.hex_to_hash(ph[DP.sha256[y]]) for x, y in zip(e.q, e.b)]
    nd_val = e[(e.sim.values >= 0.98) | (np.array(hd) <= 10)].q.nunique()
    nval = int((split == "validation").sum())
    R["E_derm1m_val_images"] = nval
    R["E_derm1m_val_identical_in_train"] = int((sp == "both").sum())
    R["E_derm1m_val_neardup_in_train_pct"] = round(100 * nd_val / nval, 1)
    json.dump(R, open(OUT / "derm1m.json", "w"), indent=1, default=str)
    print(json.dumps({k: v for k, v in R.items() if "examples" not in k}, indent=1, default=str))


if __name__ == "__main__":
    main()
