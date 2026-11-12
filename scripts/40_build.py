"""Assemble the dataset in ./dataset from the work/ artefacts.

Layout
  dataset/images/<source>/<image_id>.<ext>          original image bytes (APFS clone of the source file)
  dataset/images_restored/<source>/<image_id>.jpg   watermark-inpainted variant (atlases only; INTERNAL)
  dataset/metadata/samples.parquet                  one row per image-text occurrence
  dataset/metadata/images.parquet                   one row per unique image (sha256)
  dataset/metadata/articles.parquet                 journal article metadata (one row per DOI)
  dataset/metadata/sources.csv                      source registry (provenance, licence, tier)
  dataset/metadata/excluded.parquet                 every dropped file with the reason
  dataset/text/*.parquet                            article / chapter full text (text corpus)
  dataset/embeddings/clip_vit_b16_openai.npy        float16, row order = images.parquet
  dataset/release_public/                           what may be published (built by 42_release.py)

Image files are written with APFS clonefile(2): independent copy-on-write files
that take no extra space; the source folders are never opened for writing.
"""
import ctypes
import hashlib
import json
import os
import re
import shutil

import numpy as np
import pandas as pd
import yaml

from common import DATASET, WORK, clean_text

LIBC = ctypes.CDLL("libc.dylib", use_errno=True)
EXT = {"JPEG": "jpg", "MPO": "jpg", "PNG": "png", "GIF": "gif", "WEBP": "webp", "TIFF": "tif", "BMP": "bmp"}
DROP_ROLES = {"thumbnail", "table", "visual_abstract", "graphical_abstract", "other"}
SUBSET = {"clinical": "derm_core", "dermoscopy": "derm_core", "histopathology": "derm_core",
          "microscopy_other": "other_medical", "radiology": "other_medical", "other_photo": "other_medical",
          "chart": "non_photo", "diagram": "non_photo", "table_text": "non_photo", "molecular": "non_photo"}
SOURCE_PRIORITY = ["dat", "jid", "jaad", "jddg", "die-dermatologie", "ajcd", "jama",  # journals (CC-rich first)
                   "braun-falco", "fritsch", "dermnet", "dermis", "altmeyers", "dermatlas", "iconotheque",
                   "hellenicdermatlas", "danderm", "atlasdermatologico", "uiowa", "dermatoweb", "meddean",
                   "mddk", "dermaamin", "viamedici"]


def clone(src, dst):
    if os.path.exists(dst):
        return
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if LIBC.clonefile(src.encode(), dst.encode(), 0) != 0:
        shutil.copy2(src, dst)


def split_of(group_key):
    h = int(hashlib.sha1(str(group_key).encode()).hexdigest()[:8], 16) % 100
    return "test" if h < 5 else "val" if h < 10 else "train"


def template_text(r):
    """Text for atlas images without a caption, from structured fields (marked text_origin=template)."""
    ex = json.loads(r.extra) if isinstance(r.extra, str) else {}
    parts = [r.diagnosis]
    if isinstance(r.body_site, str) and r.body_site:
        parts.append(str(r.body_site).replace("-", " "))
    morph = ex.get("primary_morphology") or ex.get("efflorescence") or ex.get("elementary_lesion") or ex.get("lesions")
    if morph:
        parts.append(str(morph).replace("-", " "))
    if ex.get("colors") or ex.get("color"):
        parts.append(str(ex.get("colors") or ex.get("color")))
    t = ", ".join(str(p) for p in parts if isinstance(p, str) and p.strip())
    return t or None


def main():
    reg = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), "sources.yaml")))
    S = pd.concat([pd.read_parquet(WORK / f"{t}.parquet") for t in ["atlases", "textbooks", "journals"]],
                  ignore_index=True)
    info = pd.read_parquet(WORK / "image_info.parquet")
    S = S.merge(info, on="src_path", how="left")

    # ---------------- exclusions
    ex_rows = []
    def drop(mask, reason):
        nonlocal S
        d = S[mask]
        ex_rows.append(pd.DataFrame({"src_path": d.src_path, "source": d.source, "reason": reason}))
        S = S[~mask]
    S["file_role"] = S.extra.map(lambda e: (json.loads(e) if isinstance(e, str) else {}).get("file_role", "figure"))
    drop(~S.ok.fillna(False).astype(bool), "unreadable or truncated image file (or saved HTML error page)")
    drop(S.file_role.isin(DROP_ROLES), "not a figure (table image / thumbnail duplicate / graphical or visual abstract)")
    drop((S.width.fillna(0) < 64) | (S.height.fillna(0) < 64), "image smaller than 64 px")

    # ---------------- image-level annotations
    heads = pd.read_parquet(WORK / "heads_pred.parquet")
    heads = heads[heads.sha256.notna()].drop_duplicates("sha256").drop(columns=["path"])
    dd = pd.read_parquet(WORK / "dedup.parquet")
    S = S.merge(heads, on="sha256", how="left").merge(dd, on="sha256", how="left")
    S["image_id"] = S.sha256.str[:16]
    S["subset"] = S.image_type.map(SUBSET)

    # ---------------- Fitzpatrick: gold (F17K expert, URL match), LLM, CLIP head
    f17 = pd.read_csv(WORK.parent / "external/fitzpatrick17k/fitzpatrick17k.csv")
    f17 = f17[f17.fitzpatrick_scale > 0]
    norm = lambda u: str(u).split("://")[-1].replace("www.", "").lower() if isinstance(u, str) else None
    fmap = dict(zip(f17.url.map(norm), f17.fitzpatrick_scale))
    S["fst_gold_f17k"] = S.image_url.map(lambda u: fmap.get(norm(u)))
    roman = {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6}
    S["fst_llm"] = S.llm_vision_fitz_patrick_scale.map(lambda v: roman.get(str(v).strip().upper()) if v else None)
    clinical = S.image_type.eq("clinical")
    S.loc[~clinical, "fst_pred"] = np.nan
    S.loc[~clinical, "fst_pred_raw"] = np.nan

    # ---------------- text for training
    S["text"] = S.caption
    S["text_origin"] = np.where(S.caption.notna(), "original_caption", None)
    need = S.text.isna() & S.source_type.eq("atlas")
    S.loc[need, "text"] = [template_text(r) for r in S[need].itertuples()]
    S.loc[need & S.text.notna(), "text_origin"] = "structured_template"
    S["context"] = S.extra.map(lambda e: (json.loads(e) if isinstance(e, str) else {}).get("context"))
    S["text_words"] = S.text.fillna("").str.split().str.len()

    # ---------------- licence / release tier
    # figures inside CC articles that credit a third party are not covered by the article licence
    third = re.compile(r"(?:reprinted|reproduced|adapted)\s+(?:with|by)\s+(?:kind\s+)?permission|with\s+(?:kind\s+)?"
                       r"permission|©|copyright|courtesy|mit\s+(?:freundlicher\s+)?(?:genehmigung|erlaubnis)|"
                       r"nachdruck|\baus:\s|\bfrom:\s|\bsource:\s|\bquelle:", re.I)
    S["third_party_credit"] = S.caption.fillna("").str.contains(third)
    def tier(r):
        if r.source_type == "journal":
            cc = isinstance(r.license, str) and r.license.startswith("CC ")
            return "open" if cc and not r.third_party_credit else "metadata"
        return reg[r.source]["release_tier"]
    S["release_tier"] = [tier(r) for r in S.itertuples()]
    nj = S.source_type != "journal"
    S.loc[nj, "license"] = S.loc[nj, "source"].map(lambda s: reg[s]["license"])
    ai = {k: v.get("ai_training_clause", "") for k, v in reg.items()}
    S["terms_restrict_ai"] = S.source.map(lambda s: bool(ai.get(s)) and not str(ai.get(s)).startswith(("not stated", "unverified")))

    # ---------------- canonical file per sha (exact duplicates across sources)
    pr = {s: i for i, s in enumerate(SOURCE_PRIORITY)}
    S["_prio"] = S.source.map(pr).fillna(99)
    S = S.sort_values(["_prio", "src_path"])
    canon = S.drop_duplicates("sha256")[["sha256", "source", "src_path", "format"]]
    canon = canon.assign(file=[f"images/{s}/{h[:16]}.{EXT.get(f, 'bin')}" for s, h, f in
                               zip(canon.source, canon.sha256, canon.format)])

    # ---------------- splits by case group (journal figures: also by article)
    # union-find over case groups, joined through articles and same-patient links (all figures of one
    # article, near-duplicates, same-case photos and re-hosted copies end up in the same split)
    parent = {}
    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for _, cgs in S[S.article_id.notna()].groupby("article_id").case_group:
        cgs = [int(c) for c in cgs.dropna().unique()]
        for c in cgs[1:]:
            ra, rb = find(cgs[0]), find(c)
            if ra != rb:
                parent[ra] = rb
    # same patient / same photograph across case groups (crops, other shots, re-hosted atlas copies; 23_crop_dups)
    # Links are applied most reliable first (R1 re-hosted copies / R3 SIFT-proven, then R2 by similarity) and a
    # merge is skipped when the group would exceed SPLIT_GROUP_CAP images (otherwise same-diagnosis links chain
    # thousands of atlas photos into one group); skipped links are counted in work/split_links_report.json.
    if (WORK / "crop_dups.parquet").exists():
        SPLIT_GROUP_CAP = 200
        U = S.dropna(subset=["case_group"]).drop_duplicates("sha256")
        cg_of = U.set_index("sha256").case_group
        size = {}
        for c, n in U.case_group.astype(int).value_counts().items():
            size[find(c)] = size.get(find(c), 0) + n
        L = pd.read_parquet(WORK / "crop_dups.parquet", columns=["sha256_a", "sha256_b", "cos", "R1", "R2", "R3"])
        L = L.assign(pri=(~(L.R1 | L.R3)).astype(int)).sort_values(["pri", "cos", "sha256_a", "sha256_b"],
                                                                     ascending=[True, False, True, True])
        skipped = 0
        for x, y in zip(L.sha256_a.map(cg_of), L.sha256_b.map(cg_of)):
            if pd.notna(x) and pd.notna(y):
                ra, rb = find(int(x)), find(int(y))
                if ra == rb:
                    continue
                if size.get(ra, 0) + size.get(rb, 0) > SPLIT_GROUP_CAP:
                    skipped += 1
                    continue
                parent[ra] = rb
                size[rb] = size.get(rb, 0) + size.pop(ra, 0)
        sp = {c: split_of(find(int(c))) for c in U.case_group.unique()}
        sa, sb = L.sha256_a.map(cg_of).map(sp), L.sha256_b.map(cg_of).map(sp)
        json.dump({"links": len(L), "skipped_by_cap": skipped, "cap": SPLIT_GROUP_CAP,
                   "residual_cross_split_links": int((sa != sb).sum())},
                  open(WORK / "split_links_report.json", "w"), indent=1)
    S["split_key"] = S.case_group.map(lambda c: find(int(c)) if pd.notna(c) else None)
    S["split"] = S.split_key.map(split_of)

    # ---------------- images table
    img_cols = ["sha256", "image_id", "width", "height", "format", "mode", "phash", "dhash",
                "image_type", "subset", *[c for c in S.columns if c.startswith("type_p_")],
                "fst_pred", "fst_pred_raw", "near_dup_cluster", "case_group", "bench_overlap",
                "bench_overlap_test", "split"]
    I = S.drop_duplicates("sha256")[img_cols].merge(canon[["sha256", "file", "src_path"]], on="sha256")
    g = S.groupby("sha256")
    I = I.merge(pd.DataFrame({
        "n_samples": g.size(), "sources": g.source.agg(lambda x: ",".join(sorted(set(x)))),
        "fst_gold_f17k": g.fst_gold_f17k.max(), "fst_llm": g.fst_llm.max(),
        "has_restored": g.restored_path.agg(lambda x: x.notna().any()),
        "release_tier": g.release_tier.agg(lambda x: "open" if "open" in set(x) else
                                           "metadata" if "metadata" in set(x) else "internal"),
    }).reset_index(), on="sha256")
    # canonical image per near-duplicate cluster: prefer open tier, captioned, largest
    cap = g.text.agg(lambda x: x.notna().any()).rename("_has_text")
    I = I.merge(cap.reset_index(), on="sha256")
    I["_area"] = I.width * I.height
    I["_open"] = I.release_tier.eq("open")
    I = I.sort_values(["_open", "_has_text", "_area"], ascending=False)
    I["is_canonical"] = ~I.near_dup_cluster.duplicated()
    I.loc[I.near_dup_cluster.isna(), "is_canonical"] = True
    I = I.drop(columns=["_area", "_open", "_has_text"]).sort_values("image_id").reset_index(drop=True)

    # ---------------- materialise image files
    DATASET.mkdir(exist_ok=True)
    for r in I.itertuples():
        clone(r.src_path, str(DATASET / r.file))
    rest = S[S.restored_path.notna()].drop_duplicates("sha256")
    rmap = {}
    for r in rest.itertuples():
        dst = f"images_restored/{r.source}/{r.image_id}.jpg"
        clone(r.restored_path, str(DATASET / dst))
        rmap[r.sha256] = dst
    S["file_restored"] = S.sha256.map(rmap)
    S = S.merge(canon[["sha256", "file"]], on="sha256", how="left")

    # ---------------- samples table
    S = S.sort_values(["source", "src_path"]).reset_index(drop=True)
    S["sample_id"] = [f"{s}-{i:06d}" for i, s in zip(S.groupby("source").cumcount(), S.source)]
    keep = ["sample_id", "image_id", "sha256", "file", "file_restored", "source", "source_type", "split",
            "release_tier", "license", "third_party_credit", "terms_restrict_ai", "language",
            "text", "text_origin", "text_words", "caption", "context", "figure_label", "title",
            "diagnosis", "body_site", "age", "sex", "extra",
            "doi", "article_id", "url", "image_url", "year",
            "image_type", "subset", "fst_gold_f17k", "fst_llm", "fst_pred",
            "llm_vision_description", "llm_vision_most_likely_diagnosis", "llm_vision_differential_diagnoses",
            "llm_vision_body_parts_seen", "llm_text_structured",
            "near_dup_cluster", "case_group", "bench_overlap", "bench_overlap_test", "src_path"]
    S = S[keep]
    out = DATASET / "metadata"
    out.mkdir(parents=True, exist_ok=True)
    S.to_parquet(out / "samples.parquet", index=False)
    I.drop(columns=["src_path"]).to_parquet(out / "images.parquet", index=False)

    # articles, sources, exclusions
    A = pd.read_parquet(WORK / "articles.parquet")
    A.drop(columns=["html"]).to_parquet(out / "articles.parquet", index=False)
    rows = []
    for k, v in reg.items():
        rows.append({"source": k, **{kk: (json.dumps(vv) if isinstance(vv, (list, dict)) else vv) for kk, vv in v.items()},
                     "n_samples": int((S.source == k).sum()), "n_images": int(S[S.source == k].sha256.nunique())})
    for j in sorted(S[S.source_type == "journal"].source.unique()):
        sub = S[S.source == j]
        rows.append({"source": j, "type": "journal", "license": "per article (see samples.license)",
                     "release_tier": "open (CC articles) / metadata", "n_samples": len(sub),
                     "n_images": sub.sha256.nunique(), "n_open": int((sub.release_tier == "open").sum())})
    pd.DataFrame(rows).to_csv(out / "sources.csv", index=False)
    X = pd.concat(ex_rows, ignore_index=True)
    X.to_parquet(out / "excluded.parquet", index=False)

    # text corpus
    tdir = DATASET / "text"
    tdir.mkdir(exist_ok=True)
    pd.read_parquet(WORK / "text_journals.parquet").to_parquet(tdir / "journal_articles.parquet", index=False)
    pd.read_parquet(WORK / "text_textbooks.parquet").drop(columns=["src_path"]).to_parquet(
        tdir / "textbook_chapters.parquet", index=False)

    # embeddings aligned with images.parquet
    P = pd.read_parquet(WORK / "embed_local_all.parquet")
    E = np.load(WORK / "emb/local_all.npy")
    pos = dict(zip(P.sha256, range(len(P))))
    edir = DATASET / "embeddings"
    edir.mkdir(exist_ok=True)
    np.save(edir / "clip_vit_b16_openai.npy", E[[pos[h] for h in I.sha256]])

    print("samples", len(S), "| images", len(I), "| excluded files", len(X))
    print(S.groupby(["source_type", "release_tier"]).size())
    print(I.image_type.value_counts())


if __name__ == "__main__":
    main()
