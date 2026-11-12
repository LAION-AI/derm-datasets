"""Aim 2: find dermatology images in a general web image-text dataset (Re-LAION-2B-en research-safe).

The proposal planned to filter LAION-5B; LAION-5B was withdrawn in Dec 2023 and replaced by
Re-LAION-5B (Aug 2024). One metadata shard (part-00000, 16.4 M pairs = 1/128 of 2.1 B) is used.

Stage 1  text prefilter: Aho-Corasick match of a dermatology lexicon (concept vocabulary terms that
         occur >= 5x as atlas labels + generic dermatology terms) on lower-cased captions.
Stage 2  download all text-hit candidates and a random control sample (20k) of
         non-matching rows; honour "X-Robots-Tag: noai/noimageai"; images are
         thumbnailed to 384 px for analysis only.
Stage 3  CLIP ViT-B/16 embeddings + the dermatology image-type head -> precision of the text filter,
         prevalence of dermatology images among non-matching rows -> extrapolation to 2.1 B.

Usage: python 71_laion_filter.py stage1|stage2|stage3
"""
import io
import json
import random
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import ahocorasick
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from common import EXTERNAL, WORK

L = EXTERNAL / "relaion"
OUT = WORK / "laion"
OUT.mkdir(exist_ok=True)
GENERIC = ["dermatitis", "eczema", "psoriasis", "melanoma", "skin cancer", "skin rash", "rash on",
           "dermatology", "dermatologist", "skin lesion", "skin condition", "skin disease", "basal cell carcinoma",
           "squamous cell carcinoma", "rosacea", "vitiligo", "urticaria", "hives", "impetigo", "ringworm",
           "tinea", "scabies", "shingles", "herpes zoster", "molluscum", "keratosis", "seborrheic",
           "acne vulgaris", "cystic acne", "hidradenitis", "lupus rash", "skin infection", "cellulitis",
           "skin ulcer", "pressure ulcer", "wart", "plantar wart", "fungal infection", "nail fungus",
           "onychomycosis", "alopecia", "pemphigus", "pemphigoid", "dermatofibroma", "nevus", "mole removal"]
AMBIGUOUS = {"mole", "scar", "acne", "tattoo", "freckles", "warts", "rash", "hives", "burn", "bruise",
             "sunburn", "wrinkles", "nevus", "lupus", "spider", "sting", "cyst", "wart"}


def lexicon():
    V = pd.read_parquet(WORK / "dx_vocab.parquet")
    atl = pd.read_parquet(WORK / "atlases.parquet", columns=["diagnosis"])
    freq = atl.diagnosis.str.lower().str.strip().value_counts()
    terms = {t for t in V.term if len(t) >= 6 and freq.get(t, 0) >= 5}
    terms |= set(GENERIC)
    return sorted(t for t in terms if t not in AMBIGUOUS)


def stage1():
    terms = lexicon()
    A = ahocorasick.Automaton()
    for t in terms:
        A.add_word(t, t)
    A.make_automaton()
    f = next(L.glob("part-00000*.parquet"))
    pf = pq.ParquetFile(f)
    hits, n = [], 0
    for b in pf.iter_batches(batch_size=500_000, columns=["key", "url", "caption", "width", "height", "punsafe", "pwatermark"]):
        df = b.to_pandas()
        n += len(df)
        caps = df.caption.fillna("").str.lower().values
        for i, c in enumerate(caps):
            for end, t in A.iter(c):
                s = end - len(t) + 1
                if (s == 0 or not c[s - 1].isalnum()) and (end + 1 == len(c) or not c[end + 1].isalnum()):
                    hits.append((i + n - len(df), t))
                    break
        print(n, len(hits), flush=True)
    idx = pd.DataFrame(hits, columns=["row", "term"])
    idx.to_parquet(OUT / "stage1_hits.parquet", index=False)
    json.dump({"rows": n, "lexicon_terms": len(terms), "text_hits": len(idx),
               "top_terms": idx.term.value_counts().head(40).to_dict()}, open(OUT / "stage1.json", "w"), indent=1)


def fetch(args):
    import requests
    from PIL import Image
    key, url, dst = args
    try:
        r = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0 (research crawler; img2dataset-compatible)"})
        xr = r.headers.get("X-Robots-Tag", "").lower()
        if any(t in xr for t in ("noai", "noimageai", "noindex", "noimageindex")):
            return key, "opt-out"
        if r.status_code != 200:
            return key, f"http {r.status_code}"
        im = Image.open(io.BytesIO(r.content)).convert("RGB")
        im.thumbnail((384, 384))
        im.save(dst, quality=90)
        return key, "ok"
    except Exception as e:
        return key, type(e).__name__


def stage2(n_cand=10**9, n_ctrl=20000):
    f = next(L.glob("part-00000*.parquet"))
    hits = pd.read_parquet(OUT / "stage1_hits.parquet")
    rows = pq.read_table(f, columns=["key", "url", "caption"]).to_pandas()
    rng = random.Random(0)
    cand = hits.sample(min(n_cand, len(hits)), random_state=0).row.values
    hitset = set(hits.row)
    ctrl = []
    while len(ctrl) < n_ctrl:
        r = rng.randrange(len(rows))
        if r not in hitset:
            ctrl.append(r)
    sel = pd.concat([rows.iloc[cand].assign(group="text_hit"), rows.iloc[ctrl].assign(group="control")])
    sel = sel.merge(hits.rename(columns={"row": "_row"}), left_index=True, right_on="_row", how="left")
    (L / "images").mkdir(exist_ok=True)
    sel["path"] = [str(L / "images" / f"{k}.jpg") for k in sel.key]
    with ThreadPoolExecutor(48) as ex:
        status = dict(ex.map(fetch, zip(sel.key, sel.url, sel.path)))
    sel["status"] = sel.key.map(status)
    sel.drop(columns=["_row"]).to_parquet(OUT / "stage2_sample.parquet", index=False)
    print(sel.groupby("group").status.value_counts())


def stage3():
    import joblib
    sel = pd.read_parquet(OUT / "stage2_sample.parquet")
    ok = sel[sel.status == "ok"].reset_index(drop=True)
    ok[["path"]].to_parquet(OUT / "embed_laion.parquet", index=False)
    if not (OUT / "laion_emb.npy").exists():
        subprocess.run([sys.executable, "30_embed.py", str(OUT / "embed_laion.parquet"), str(OUT / "laion_emb")],
                       check=True)
    E = np.load(OUT / "laion_emb.npy").astype(np.float32)
    heads = joblib.load(WORK / "heads_models.joblib")
    P = heads["type"].predict_proba(E)
    cls = list(heads["type"].classes_)
    ok["image_type"] = np.array(cls)[P.argmax(1)]
    ok["p_derm_core"] = P[:, [cls.index(c) for c in ("clinical", "dermoscopy", "histopathology")]].sum(1)
    ok["derm_core"] = ok.p_derm_core >= 0.5
    ok.to_parquet(OUT / "stage3_scored.parquet", index=False)
    s1 = json.load(open(OUT / "stage1.json"))
    g = ok.groupby("group").derm_core.mean()
    hit_rate = s1["text_hits"] / s1["rows"]
    est_per_shard = s1["text_hits"] * g.get("text_hit", 0) + (s1["rows"] - s1["text_hits"]) * g.get("control", 0)
    R = {"shard_rows": s1["rows"], "text_hits": s1["text_hits"], "text_hit_rate_pct": round(100 * hit_rate, 3),
         "downloaded_ok": ok.groupby("group").size().to_dict(),
         "download_status": sel.groupby("group").status.value_counts().unstack(0).fillna(0).astype(int).to_dict(),
         "derm_core_rate_text_hits_pct": round(100 * g.get("text_hit", 0), 2),
         "derm_core_rate_control_pct": round(100 * g.get("control", 0), 3),
         "image_types_text_hits": ok[ok.group == "text_hit"].image_type.value_counts().to_dict(),
         "estimated_derm_core_per_shard": int(est_per_shard),
         "estimated_derm_core_relaion2B_en": int(est_per_shard * 128),
         "share_of_derm_found_by_text_filter_pct": round(
             100 * s1["text_hits"] * g.get("text_hit", 0) / max(1, est_per_shard), 1),
         "top_terms_derm_precision": ok[ok.group == "text_hit"].groupby("term").derm_core.agg(["mean", "size"])
             .query("size >= 30").sort_values("mean", ascending=False).round(3).head(25).to_dict(orient="index")}
    json.dump(R, open(OUT / "laion.json", "w"), indent=1, default=str)
    print(json.dumps(R, indent=1, default=str))


LAION_GOLD_POS = [5, 14, 15, 76]   # hand labels: indices of dermatology images among 96 random text hits


def stage4():
    """Iteration 2: the round-1 head (medical sources only) flagged 3.7 % of random web images as
    dermatology - 0/48 of a checked sample were. Retrain with random web images as an extra
    `web_other` class (5-fold CV over the control images so that every control prediction is
    out-of-fold), then re-estimate prevalence and the text-filter precision."""
    import joblib
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import KFold
    ts = np.load(WORK / "heads_trainset.npz", allow_pickle=True)
    X0, y0, w0 = ts["X"].astype(np.float32), ts["y"], ts["w"]
    ok = pd.read_parquet(OUT / "stage3_scored.parquet")
    E = np.load(OUT / "laion_emb.npy").astype(np.float32)
    ctrl = np.where(ok.group.values == "control")[0]
    hits = np.where(ok.group.values == "text_hit")[0]
    core = ["clinical", "dermoscopy", "histopathology"]
    def fit(idx_web):
        X = np.concatenate([X0, E[idx_web]]); y = np.concatenate([y0, ["web_other"] * len(idx_web)])
        w = np.concatenate([w0, np.ones(len(idx_web))])
        return LogisticRegression(max_iter=4000, C=2.0, class_weight="balanced").fit(X, y, sample_weight=w)
    def p_core(m, X):
        P = m.predict_proba(X); cls = list(m.classes_)
        return P[:, [cls.index(c) for c in core]].sum(1), np.array(cls)[P.argmax(1)]
    oof = np.zeros(len(ok)); oof_type = np.empty(len(ok), dtype=object)
    for tr, te in KFold(5, shuffle=True, random_state=0).split(ctrl):
        m = fit(ctrl[tr])
        oof[ctrl[te]], oof_type[ctrl[te]] = p_core(m, E[ctrl[te]])
    m = fit(ctrl)
    oof[hits], oof_type[hits] = p_core(m, E[hits])
    joblib.dump(m, WORK / "heads_type_webaware.joblib")
    ok["p_derm_core_r2"], ok["type_r2"] = oof, oof_type
    ok["derm_core_r2"] = ok.p_derm_core_r2 >= 0.5
    ok.to_parquet(OUT / "stage4_scored.parquet", index=False)
    # medical gold set must not degrade
    g = pd.read_csv(WORK / "gold_types.csv")
    from common import load_local_embeddings
    P, LE = load_local_embeddings()
    pos = {p: i for i, p in enumerate(P.path)}
    g = g[g.path.isin(pos)]
    GOLD_TO_HEAD = {"chart": "figure_nonphoto", "diagram": "figure_nonphoto", "table_text": "figure_nonphoto",
                    "molecular": "figure_nonphoto"}
    pred = m.predict(LE[[pos[p] for p in g.path]].astype(np.float32))
    gold_acc = float((g.label.map(lambda l: GOLD_TO_HEAD.get(l, l)).values == pred).mean())
    # LAION gold: 96 random text hits, hand-labelled
    lg = pd.read_parquet(WORK / "sheets/laion_gold.parquet")
    lg["derm_true"] = [i in LAION_GOLD_POS for i in range(len(lg))]
    lg = lg.merge(ok[["path", "derm_core", "derm_core_r2"]], on="path")
    def pr(pred, true):
        tp = int((pred & true).sum())
        return {"tp": tp, "pred_pos": int(pred.sum()), "true_pos": int(true.sum()),
                "precision": tp / max(1, int(pred.sum())), "recall": tp / max(1, int(true.sum()))}
    s1 = json.load(open(OUT / "stage1.json"))
    rate_hit = float(ok.loc[hits, "derm_core_r2"].mean()); rate_ctrl = float(ok.loc[ctrl, "derm_core_r2"].mean())
    R = {"round1_control_flag_rate_pct": round(100 * float(ok.loc[ctrl, "derm_core"].mean()), 3),
         "round2_control_flag_rate_pct_oof": round(100 * rate_ctrl, 3),
         "round1_text_hit_flag_rate_pct": round(100 * float(ok.loc[hits, "derm_core"].mean()), 2),
         "round2_text_hit_flag_rate_pct": round(100 * rate_hit, 2),
         "laion_gold_true_derm_rate_text_hits_pct": round(100 * float(lg.derm_true.mean()), 2),
         "laion_gold_round1": pr(lg.derm_core.values, lg.derm_true.values),
         "laion_gold_round2": pr(lg.derm_core_r2.values, lg.derm_true.values),
         "medical_gold_acc_webaware_head_insample": round(gold_acc, 3),
         "est_derm_per_shard_round2": int(s1["text_hits"] * rate_hit + (s1["rows"] - s1["text_hits"]) * rate_ctrl),
         "control_positives_r2": int(ok.loc[ctrl, "derm_core_r2"].sum()),
         "text_hit_positives_r2": int(ok.loc[hits, "derm_core_r2"].sum())}
    R["est_derm_relaion2B_en_round2"] = R["est_derm_per_shard_round2"] * 128
    json.dump(R, open(OUT / "laion_round2.json", "w"), indent=1)
    print(json.dumps(R, indent=1))


if __name__ == "__main__":
    {"stage1": stage1, "stage2": stage2, "stage3": stage3, "stage4": stage4}[sys.argv[1]]()
