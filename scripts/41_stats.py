"""Compute dataset statistics used in dataset/README.md and the report.

Writes dataset/docs/stats.json and dataset/docs/stats_tables.md.
"""
import json

import numpy as np
import pandas as pd

from common import DATASET

D = DATASET / "metadata"


def main():
    S = pd.read_parquet(D / "samples.parquet")
    I = pd.read_parquet(D / "images.parquet")
    A = pd.read_parquet(D / "articles.parquet")
    X = pd.read_parquet(D / "excluded.parquet")
    st = {}
    st["n_samples"] = len(S)
    st["n_images"] = len(I)
    st["n_canonical_images"] = int(I.is_canonical.sum())
    st["n_derm_core_images"] = int((I.subset == "derm_core").sum())
    st["n_derm_core_canonical"] = int(((I.subset == "derm_core") & I.is_canonical).sum())
    st["n_articles"] = len(A)
    st["n_excluded_files"] = len(X)
    st["excluded_reasons"] = X.reason.value_counts().to_dict()
    st["n_with_original_caption"] = int((S.text_origin == "original_caption").sum())
    st["n_with_text"] = int(S.text.notna().sum())
    st["words_original_caption"] = S.loc[S.text_origin == "original_caption", "text_words"].describe().round(1).to_dict()
    st["n_with_context"] = int(S.context.notna().sum())
    st["languages"] = S.language.value_counts().to_dict()
    st["release_tier_samples"] = S.release_tier.value_counts().to_dict()
    st["release_tier_images"] = I.release_tier.value_counts().to_dict()
    st["image_types"] = I.image_type.value_counts().to_dict()
    st["splits_images"] = I.split.value_counts().to_dict()
    st["bench_overlap_images"] = int(I.bench_overlap.notna().sum())
    st["bench_overlap_test_images"] = int(I.bench_overlap_test.sum())
    st["median_long_side_px"] = int(np.median(np.maximum(I.width, I.height)))
    st["terms_restrict_ai_samples"] = int(S.terms_restrict_ai.sum())
    yrs = S.year.dropna().astype(int)
    st["year_range"] = [int(yrs.min()), int(yrs.max())] if len(yrs) else None

    # per-source table
    g = S.groupby(["source_type", "source"])
    tab = pd.DataFrame({
        "samples": g.size(),
        "unique_images": g.sha256.nunique(),
        "original_caption": g.text_origin.agg(lambda x: (x == "original_caption").mean()),
        "median_words": g.text_words.median(),
        "derm_core": g.subset.agg(lambda x: (x == "derm_core").mean()),
        "language": g.language.agg(lambda x: ",".join(x.value_counts().index[:2])),
        "years": g.year.agg(lambda x: f"{int(x.min())}–{int(x.max())}" if x.notna().any() else ""),
        "open_tier": g.release_tier.agg(lambda x: (x == "open").sum()),
    }).reset_index()
    st["per_source"] = tab.to_dict(orient="records")

    # demographics / skin type (images with derm_core clinical photos)
    clin = S[S.image_type == "clinical"].drop_duplicates("sha256")
    st["fst_pred_clinical"] = clin.fst_pred.value_counts().sort_index().to_dict()
    st["fst_llm_clinical"] = clin.fst_llm.value_counts().sort_index().to_dict()
    st["fst_gold_f17k"] = S.drop_duplicates("sha256").fst_gold_f17k.value_counts().sort_index().to_dict()
    st["sex_raw_filled"] = int(S.sex.notna().sum())
    st["age_raw_filled"] = int(S.age.notna().sum())
    st["body_site_filled"] = int(S.body_site.notna().sum())
    st["diagnosis_filled"] = int(S.diagnosis.notna().sum())
    st["n_distinct_diagnosis_labels"] = int(S.diagnosis.str.lower().nunique())

    json.dump(st, open(DATASET / "docs/stats.json", "w"), indent=1, default=str)
    fmt = tab.copy()
    for c in ["original_caption", "derm_core"]:
        fmt[c] = (100 * fmt[c]).round(0).astype(int).astype(str) + "%"
    fmt["median_words"] = fmt.median_words.astype(int)
    open(DATASET / "docs/stats_tables.md", "w").write(fmt.to_markdown(index=False))
    print(json.dumps({k: v for k, v in st.items() if k != "per_source"}, indent=1, default=str))
    print(fmt.to_markdown(index=False))


if __name__ == "__main__":
    (DATASET / "docs").mkdir(exist_ok=True)
    main()
