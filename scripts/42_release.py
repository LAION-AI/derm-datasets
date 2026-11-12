"""Build dataset/release_public/ — the only part of the dataset that may be published.

release_public/
  README.md                         public dataset card
  open/images/<source>/<id>.<ext>   figures from Creative Commons-licensed articles
  open/samples.parquet              their captions, context, licence, attribution
  open/ATTRIBUTION.csv              doi, title, authors, journal, year, licence
  metadata/samples.parquet          LAION-style pointer release for the `metadata` tier:
                                    identifiers + URLs + hashes + derived labels, no images,
                                    no verbatim caption text
`internal` rows (Springer textbooks, Thieme via medici) are not released at all.
"""
import ctypes
import os
import shutil

import pandas as pd
import yaml

from common import DATASET

LIBC = ctypes.CDLL("libc.dylib", use_errno=True)
OUT = DATASET / "release_public"
DERIVED = ["sample_id", "image_id", "sha256", "source", "source_type", "split", "release_tier", "license",
           "terms_restrict_ai", "language", "figure_label", "doi", "url", "image_url", "year",
           "diagnosis", "body_site", "age", "sex", "image_type", "subset", "fst_gold_f17k", "fst_llm",
           "fst_pred", "llm_vision_description", "llm_vision_most_likely_diagnosis",
           "near_dup_cluster", "case_group", "bench_overlap", "bench_overlap_test"]


def clone(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if not os.path.exists(dst) and LIBC.clonefile(src.encode(), dst.encode(), 0) != 0:
        shutil.copy2(src, dst)


def wayback(u):
    return f"https://web.archive.org/web/2020/{u}" if isinstance(u, str) and u else None


def main():
    reg = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), "sources.yaml")))
    S = pd.read_parquet(DATASET / "metadata/samples.parquet")
    I = pd.read_parquet(DATASET / "metadata/images.parquet").set_index("image_id")
    A = pd.read_parquet(DATASET / "metadata/articles.parquet").set_index("doi")
    S = S[S.release_tier.isin(["open", "metadata"])].copy()
    S = S.join(I[["width", "height", "phash"]], on="image_id")

    # URL policy per source: hijacked domains -> Wayback; publish_url false -> drop
    for src, cfg in reg.items():
        pol = cfg.get("publish_url", True)
        m = S.source == src
        if pol == "archive":
            S.loc[m, "url"] = S.loc[m, "url"].map(wayback)
            S.loc[m, "image_url"] = S.loc[m, "image_url"].map(wayback)
        elif pol is False:
            S.loc[m, ["url", "image_url"]] = None

    # ---------- open tier
    O = S[S.release_tier == "open"].copy()
    src_file = O.image_id.map(I.file)  # images/<canonical source>/<id>.<ext> inside dataset/
    O["file"] = ["images/" + f.split("/", 1)[1] for f in src_file]
    for f, src in zip(O.file, src_file):
        clone(str(DATASET / src), str(OUT / "open" / f))
    O["authors"] = O.doi.map(lambda d: "; ".join(A.authors.get(d, [])) if d in A.index else None)
    O["journal_name"] = O.doi.map(A.journal_name)
    keep = DERIVED + ["width", "height", "phash", "file", "caption", "context", "title", "authors", "journal_name"]
    O[keep].to_parquet(OUT / "open/samples.parquet", index=False)
    O.drop_duplicates("doi")[["doi", "title", "authors", "journal_name", "year", "license"]] \
        .to_csv(OUT / "open/ATTRIBUTION.csv", index=False)

    # ---------- metadata tier (pointer release)
    M = S[S.release_tier == "metadata"].copy()
    (OUT / "metadata").mkdir(parents=True, exist_ok=True)
    M[DERIVED + ["width", "height", "phash"]].to_parquet(OUT / "metadata/samples.parquet", index=False)

    n_open_img = O.image_id.nunique()
    print(f"open: {len(O)} samples / {n_open_img} images / {O.doi.nunique()} articles")
    print(O.license.value_counts().to_dict())
    print(f"metadata: {len(M)} samples; journal rows {int((M.source_type == 'journal').sum())}, "
          f"atlas rows {int((M.source_type == 'atlas').sum())}")
    print("caption columns in metadata release:", [c for c in ["caption", "text", "context"] if c in M[DERIVED].columns])


if __name__ == "__main__":
    main()
