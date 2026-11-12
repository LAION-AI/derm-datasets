"""Permissively licensed real reference images for lesion projection (SynthDerm-1M).

Sources:
  Derm1M PubMed figures (Derm1M itself is CC BY-NC as a compilation, but each figure keeps the licence of its source
    article): PMCID from the file name -> article licence from Europe PMC (REST search, resultType=core, 100 IDs per query;
    cached in external/pmc/europepmc_licenses.parquet).
  Our corpus: journal figures from articles detected as CC BY / CC0 (40_build licence field).
Kept: CC0 / public domain / CC BY (any version) -> `permissive`; CC BY-SA kept but flagged `share_alike` (derivatives
must be CC BY-SA). Excluded: NC / ND / unknown; figures whose caption credits a third party ("reproduced / reprinted /
adapted from", "with permission", "courtesy of", copyright sign), whose licence can differ from the article's;
near-duplicates of benchmark images (PAD, HAM, F17K, Daffodil, SkinCAP; evaluation must stay clean).
Per image: attribution (authors, title, journal, year, PMCID / DOI, licence), caption, diagnosis labels, body site (Derm1M),
image type and Fitzpatrick estimate (32_heads models), near-duplicate cluster.
Writes work/refs/permissive_refs.parquet and work/refs/summary.json.
"""
import json
import re
import time
import urllib.parse
import urllib.request

import joblib
import numpy as np
import pandas as pd

from common import DATASET, EXTERNAL, WORK

CACHE = EXTERNAL / "pmc/europepmc_licenses.parquet"
OUT = WORK / "refs"
THIRD_PARTY = re.compile(r"reproduced|reprinted|adapted from|with (kind )?permission|courtesy of|©|copyright", re.I)


def europepmc_licenses(pmcids):
    have = pd.read_parquet(CACHE) if CACHE.exists() else pd.DataFrame(columns=["pmcid", "license", "title", "authors", "journal", "year", "doi"])
    todo = sorted(set(pmcids) - set(have.pmcid))
    rows = []
    for i in range(0, len(todo), 100):
        q = " OR ".join(f"PMCID:{p}" for p in todo[i:i + 100])
        url = "https://www.ebi.ac.uk/europepmc/webservices/rest/search?" + urllib.parse.urlencode(
            {"query": q, "resultType": "core", "format": "json", "pageSize": 1000})
        for k in range(4):
            try:
                with urllib.request.urlopen(url, timeout=120) as r:
                    d = json.load(r)
                break
            except OSError:
                time.sleep(10 * (k + 1))
        else:
            raise RuntimeError("Europe PMC unreachable")
        found = set()
        for r in d["resultList"]["result"]:
            found.add(r.get("pmcid"))
            rows.append({"pmcid": r.get("pmcid"), "license": (r.get("license") or "").lower(), "title": r.get("title"),
                         "authors": r.get("authorString"), "journal": (r.get("journalInfo") or {}).get("journal", {}).get("title"),
                         "year": r.get("pubYear"), "doi": r.get("doi")})
        rows += [{"pmcid": p, "license": "", "title": None, "authors": None, "journal": None, "year": None, "doi": None}
                 for p in todo[i:i + 100] if p not in found]
        if (i // 100) % 20 == 0:
            print("Europe PMC", i + 100, "/", len(todo), flush=True)
            pd.concat([have, pd.DataFrame(rows)]).to_parquet(CACHE, index=False)
        time.sleep(0.2)
    have = pd.concat([have, pd.DataFrame(rows)]).drop_duplicates("pmcid")
    have.to_parquet(CACHE, index=False)
    return have


def classify(lic):
    l = (lic or "").lower().replace("-", " ").replace("_", " ")
    if not l:
        return "unknown"
    if "cc0" in l or "public domain" in l or l.strip() in ("pd",):
        return "cc0"
    if "nc" in l.split() or "non commercial" in l or "noncommercial" in l or " nd" in f" {l}" or "no deriv" in l:
        return "restricted"
    if "by" in l and "sa" in l.split():
        return "cc_by_sa"
    if l.startswith("cc by") or l == "cc by" or "attribution" in l:
        return "cc_by"
    return "restricted"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (EXTERNAL / "pmc").mkdir(parents=True, exist_ok=True)
    # ---------- Derm1M PubMed figures
    d = pd.read_parquet(WORK / "derm1m.parquet")
    d = d[d.source.str.startswith("pubmed")].copy()
    d["pmcid"] = d.filename.str.extract(r"(PMC\d+)")[0]
    L = europepmc_licenses(d.pmcid.dropna().unique())
    d = d.merge(L, on="pmcid", how="left")
    II = pd.read_parquet(WORK / "image_info.parquet", columns=["src_path", "sha256"])
    d = d.merge(II, on="src_path", how="left")
    dm = pd.DataFrame({
        "sha256": d.sha256, "path": d.src_path, "origin": "derm1m_pubmed", "license_raw": d.license, "pmcid": d.pmcid,
        "doi": d.doi, "attribution": [f"{a} {t} {j} ({y}). {p}. Licence: {l}." for a, t, j, y, p, l in
                                      zip(d.authors, d.title, d.journal, d.year, d.pmcid, d.license)],
        "caption": d.caption, "diagnosis": d.disease_label, "body_site": d.body_location})
    # ---------- our CC journal figures
    S = pd.read_parquet(DATASET / "metadata/samples.parquet",
                        columns=["sha256", "file", "license", "third_party_credit", "caption", "diagnosis", "body_site", "doi", "title", "year", "source"])
    S = S[S.license.fillna("").str.contains(r"CC0|public domain|CC BY", case=False, regex=True)]
    ours = pd.DataFrame({
        "sha256": S.sha256, "path": [str(DATASET / f) for f in S.file], "origin": "ours_" + S.source, "license_raw": S.license,
        "pmcid": None, "doi": S.doi, "attribution": [f"{t} ({y}). doi:{o}. Licence: {l}." for t, y, o, l in zip(S.title, S.year, S.doi, S.license)],
        "caption": S.caption, "diagnosis": S.diagnosis, "body_site": S.body_site,
        "third_party_flag": S.third_party_credit.fillna(False).astype(bool)})
    R = pd.concat([dm.assign(third_party_flag=False), ours], ignore_index=True)
    R["license_class"] = R.license_raw.map(classify)
    R["third_party_flag"] = R.third_party_flag | R.caption.fillna("").str.contains(THIRD_PARTY)
    R = R.dropna(subset=["sha256"]).drop_duplicates("sha256")

    # image type / Fitzpatrick estimate from the CLIP embeddings (Derm1M + local batches)
    heads = joblib.load(WORK / "heads_models.joblib")
    emb = {}
    DP = pd.read_parquet(WORK / "embed_derm1m.parquet"); DE = np.load(WORK / "emb/derm1m.npy")
    emb.update(dict(zip(DP.sha256, range(len(DP)))))
    X = np.zeros((len(R), DE.shape[1]), np.float32); have = np.zeros(len(R), bool)
    for i, h in enumerate(R.sha256):
        j = emb.get(h)
        if j is not None:
            X[i] = DE[j]; have[i] = True
    from common import load_local_embeddings
    P, E = load_local_embeddings()
    pos = dict(zip(P.sha256, range(len(P))))
    for i, h in enumerate(R.sha256):
        if not have[i] and h in pos:
            X[i] = E[pos[h]]; have[i] = True
    R["image_type"] = np.where(have, heads["type"].predict(X), None)
    R["fst_pred"] = np.where(have, np.clip(np.rint(heads["fst"].predict(X)), 1, 6), np.nan)
    # benchmark contamination (near-duplicates of benchmark images, as in 63_make_pairs.py)
    nnb = WORK / "nn_bench_derm1m.parquet"
    contaminated = set(DP.sha256.values[pd.read_parquet(nnb).b.unique()]) if nnb.exists() else set()
    I = pd.read_parquet(DATASET / "metadata/images.parquet", columns=["sha256", "bench_overlap"])
    contaminated |= set(I[I.bench_overlap.notna()].sha256)
    R["bench_overlap"] = R.sha256.isin(contaminated)
    R["usable"] = R.license_class.isin(["cc0", "cc_by", "cc_by_sa"]) & ~R.third_party_flag & ~R.bench_overlap
    R["lesion_photo"] = R.image_type.isin(["clinical", "dermoscopy"])
    R.to_parquet(OUT / "permissive_refs.parquet", index=False)
    u = R[R.usable]
    summ = {"candidates": len(R), "license_class": R.license_class.value_counts().to_dict(),
            "third_party_excluded": int(R.third_party_flag.sum()), "bench_overlap_excluded": int(R.bench_overlap.sum()),
            "usable": len(u), "usable_lesion_photos": int(u.lesion_photo.sum()),
            "usable_lesion_photos_by_licence": u[u.lesion_photo].license_class.value_counts().to_dict(),
            "usable_lesion_photos_by_origin": u[u.lesion_photo].origin.value_counts().head(10).to_dict(),
            "usable_lesion_photos_fst": u[u.lesion_photo].fst_pred.value_counts().sort_index().to_dict(),
            "usable_lesion_photos_with_diagnosis": int(u[u.lesion_photo].diagnosis.notna().sum())}
    json.dump(summ, open(OUT / "summary.json", "w"), indent=1, default=str)
    print(json.dumps(summ, indent=1, default=str))


if __name__ == "__main__":
    main()
