"""Aim 2 at full scale: dermatology content of all of Re-LAION-5B (2B-en + 2B-multi + 1B-nolang research-safe, 5.5 B pairs).

71_laion_filter.py image-classified one shard of 2B-en (1/128). Re-LAION releases only URLs and captions (no CLIP
embeddings), so classifying every image would mean downloading 5.5 B images. This script therefore combines
  scan     a text pass over EVERY caption of all 384 metadata shards (same lexicon as 71, plus generic dermatology terms
           in the main languages of 2B-multi), and a uniform random control sample of non-matching rows per shard;
           each shard is downloaded, scanned and deleted (2B-en part-00000, kept by 71, is reused);
  fetch    image download for the control sample and a random sample of text hits per subset (honours noai opt-outs);
  score    CLIP ViT-L/14 embeddings + the ViT-L/14 image-type head of 106 (head_vitl14.joblib), nudity screen first;
  summary  per-subset and total estimates with bootstrap intervals -> work/laion5b/summary.json.
Usage: python 105_relaion5b.py scan [--workers 6] | fetch | score | summary
"""
import argparse
import importlib
import json
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from common import EXTERNAL, WORK

OUT = WORK / "laion5b"
SCAN = OUT / "scan"
META = EXTERNAL / "relaion_meta"
SUBSETS = {"en": "laion/relaion2B-en-research-safe", "multi": "laion/relaion2B-multi-research-safe",
           "nolang": "laion/relaion1b-nolang-research-safe"}
CTRL_PER_SHARD = 320          # x 128 shards = ~41k control rows per subset (about half download)
HITS_PER_SUBSET = 20000       # random text hits to download per subset
CORE = ["clinical", "dermoscopy", "histopathology"]
FLAG = 0.5                    # p_derm_core threshold of a classifier flag
NUDITY_FLAG = 0.15            # as in 107_laion_index_score.py
# generic dermatology terms in the main languages of 2B-multi (de, fr, es, it, pt, nl, pl, ru, ja, zh)
MULTI = ["hautkrankheit", "hautausschlag", "neurodermitis", "schuppenflechte", "hautkrebs", "ekzem", "dermatologie",
         "hautarzt", "maladie de peau", "éruption cutanée", "eczéma", "dermatite", "cancer de la peau", "dermatologue",
         "enfermedad de la piel", "erupción cutánea", "eccema", "dermatitis atópica", "cáncer de piel", "dermatólogo",
         "malattia della pelle", "eruzione cutanea", "psoriasi", "dermatologo", "tumore della pelle",
         "doença de pele", "erupção cutânea", "psoríase", "dermatologista", "câncer de pele", "huidziekte",
         "huiduitslag", "huidkanker", "choroba skóry", "wysypka", "łuszczyca", "egzema", "atopowe zapalenie skóry",
         "czerniak", "дерматит", "псориаз", "экзема", "сыпь на коже", "меланома", "дерматолог", "кожное заболевание",
         "皮膚炎", "湿疹", "乾癬", "皮膚病", "皮膚科", "アトピー", "蕁麻疹", "皮肤病", "皮炎", "银屑病", "牛皮癣",
         "荨麻疹", "白癜风", "皮肤科", "黑色素瘤"]


def automaton():
    import ahocorasick
    g71 = importlib.import_module("71_laion_filter")
    A = ahocorasick.Automaton()
    for t in sorted(set(g71.lexicon()) | set(MULTI)):
        A.add_word(t, t)
    A.make_automaton()
    return A


def bounded(c, s, end, t):
    """Word boundaries for alphabetic scripts; CJK terms match anywhere."""
    if any("぀" <= ch <= "鿿" for ch in t):
        return True
    return (s == 0 or not c[s - 1].isalnum()) and (end + 1 == len(c) or not c[end + 1].isalnum())


def scan_shard(job):
    subset, fname = job
    shard = fname.split("-")[1]
    out = SCAN / f"{subset}_{shard}.parquet"
    if out.exists():
        return subset, shard, "done"
    from huggingface_hub import hf_hub_download
    local = EXTERNAL / "relaion" / fname if subset == "en" and shard == "00000" else None
    for attempt in range(5):
        try:
            p = local or Path(hf_hub_download(SUBSETS[subset], fname, repo_type="dataset", local_dir=META / subset))
            break
        except Exception:
            time.sleep(30 * (attempt + 1))
    else:
        return subset, shard, "download failed"
    A = automaton()
    rng = random.Random(f"{subset}{shard}")
    pf = pq.ParquetFile(p)
    n_rows = pf.metadata.num_rows
    ctrl_rows = set(rng.sample(range(n_rows), CTRL_PER_SHARD))
    hits, ctrl, n, n_caption = [], [], 0, 0
    cols = ["key", "url", "caption", "punsafe", "width", "height"]
    for b in pf.iter_batches(batch_size=500_000, columns=cols):
        df = b.to_pandas()
        caps = df.caption.fillna("").str.lower().values
        n_caption += int((df.caption.fillna("").str.len() > 0).sum())
        hit_i = {}
        for i, c in enumerate(caps):
            for end, t in A.iter(c):
                if bounded(c, end - len(t) + 1, end, t):
                    hit_i[i] = t
                    break
        if hit_i:
            hits.append(df.iloc[list(hit_i)].assign(term=list(hit_i.values()), row=[n + i for i in hit_i]))
        ci = [r - n for r in ctrl_rows if n <= r < n + len(df) and (r - n) not in hit_i]
        if ci:
            ctrl.append(df.iloc[ci].assign(row=[n + i for i in ci]))
        n += len(df)
    H = pd.concat(hits).assign(group="text_hit") if hits else pd.DataFrame()
    C = pd.concat(ctrl).assign(group="control", term=None) if ctrl else pd.DataFrame()
    R = pd.concat([H, C]).assign(subset=subset, shard=shard)
    R.attrs = {}
    tmp = out.with_suffix(".tmp")
    R.to_parquet(tmp, index=False)
    json.dump({"rows": n, "with_caption": n_caption, "text_hits": len(H)}, open(out.with_suffix(".json"), "w"))
    tmp.rename(out)
    if local is None:
        p.unlink(missing_ok=True)
    return subset, shard, f"{n} rows, {len(H)} hits"


def scan(workers):
    from huggingface_hub import HfApi
    SCAN.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    jobs = [(s, f) for s, r in SUBSETS.items() for f in sorted(api.list_repo_files(r, repo_type="dataset"))
            if f.startswith("part-") and f.endswith(".parquet")]
    print(len(jobs), "shards", flush=True)
    with ProcessPoolExecutor(workers) as ex:
        for s, sh, msg in ex.map(scan_shard, jobs):
            print(time.strftime("%H:%M"), s, sh, msg, flush=True)


def fetch():
    """Download the control rows (all) and HITS_PER_SUBSET random text hits of each subset. Control rows of 2B-en
    part-00000 are dropped: that shard's control images trained the round-2 head (71 stage4)."""
    g71 = importlib.import_module("71_laion_filter")
    R = pd.concat([pd.read_parquet(f) for f in sorted(SCAN.glob("*.parquet"))], ignore_index=True)
    R = R[~((R.subset == "en") & (R.shard == "00000") & (R.group == "control"))]
    sel = pd.concat([R[R.group == "control"]] +
                    [d.sample(min(HITS_PER_SUBSET, len(d)), random_state=0)
                     for _, d in R[R.group == "text_hit"].groupby("subset")], ignore_index=True)
    img = EXTERNAL / "relaion5b_images"
    for s in SUBSETS:
        (img / s).mkdir(parents=True, exist_ok=True)
    sel["path"] = [str(img / s / f"{sh}_{r}.jpg") for s, sh, r in zip(sel.subset, sel.shard, sel.row)]
    done = OUT / "fetch_status.parquet"
    old = pd.read_parquet(done).set_index("path").status.to_dict() if done.exists() else {}
    todo = sel[~sel.path.map(lambda p: old.get(p) is not None)]
    print(len(sel), "selected,", len(todo), "to fetch", flush=True)
    with ThreadPoolExecutor(64) as ex:
        for i, (k, st) in enumerate(ex.map(g71.fetch, zip(todo.path, todo.url, todo.path))):
            old[k] = st
            if i % 5000 == 4999:
                pd.DataFrame({"path": list(old), "status": list(old.values())}).to_parquet(done, index=False)
                print(i + 1, flush=True)
    pd.DataFrame({"path": list(old), "status": list(old.values())}).to_parquet(done, index=False)
    sel["status"] = sel.path.map(old)
    sel.to_parquet(OUT / "sample.parquet", index=False)
    print(sel.groupby(["subset", "group"]).status.apply(lambda s: (s == "ok").sum()))


def score():
    """CLIP ViT-L/14 embeddings (30_embed.py) + the ViT-L/14 head of 106 (the head used on the full 2B-en index, gold
    recall 0.95 vs 0.75 for the ViT-B/16 round-2 head) -> p_derm_core per image. Nudity screen first (CLIP text probes,
    same labels and threshold 0.15 as 107): flagged images are deleted unseen and never put on review sheets. Review
    sheets: every flagged control image and a random 150 flagged text hits per subset."""
    import subprocess
    import joblib
    from PIL import Image, ImageDraw
    S = pd.read_parquet(OUT / "sample.parquet")
    ok = S[S.status == "ok"].reset_index(drop=True)
    ok[["path"]].to_parquet(OUT / "embed.parquet", index=False)
    pref = OUT / "emb_vitl14"
    if not Path(f"{pref}.npy").exists():
        subprocess.run([sys.executable, "30_embed.py", str(OUT / "embed.parquet"), str(pref),
                        "--model", "ViT-L-14-quickgelu", "--pretrained", "openai"], check=True)
    E = np.load(f"{pref}.npy").astype(np.float32)
    good = np.load(f"{pref}.ok.npy")
    ok, E = ok[good].reset_index(drop=True), E[good]
    t = np.load(OUT / "nudity_text_vitl14.npy").astype(np.float32)  # 107 nudity: NUDITY_LABELS, row 0 = benign skin
    z = 100 * E @ t.T
    z = np.exp(z - z.max(1, keepdims=True))
    ok["p_exposed"] = z[:, 1:].sum(1) / z.sum(1)
    ok["nudity_flag"] = ok.p_exposed >= NUDITY_FLAG
    for f in ok.path[ok.nudity_flag]:
        Path(f).unlink(missing_ok=True)  # deleted unseen
    m = joblib.load(OUT / "head_vitl14.joblib")
    P = m.predict_proba(E)
    cls = list(m.classes_)
    ok["p_derm_core"] = P[:, [cls.index(c) for c in CORE]].sum(1)
    ok["image_type"] = np.array(cls)[P.argmax(1)]
    ok["flag"] = ok.p_derm_core >= FLAG
    ok.to_parquet(OUT / "scored.parquet", index=False)
    print(ok.groupby(["subset", "group"]).agg(n=("flag", "size"), flag=("flag", "sum"), rate=("flag", "mean"),
                                              nudity=("nudity_flag", "sum"),
                                              flag_nudity=("flag", lambda x: (x & ok.nudity_flag[x.index]).sum())))
    vis = ok[ok.flag & ~ok.nudity_flag]
    rev = pd.concat([vis[vis.group == "control"]] +
                    [d.sample(min(150, len(d)), random_state=0) for _, d in vis[vis.group == "text_hit"].groupby("subset")],
                    ignore_index=True)
    rev["rid"] = range(len(rev))
    rev.to_parquet(OUT / "review.parquet", index=False)
    (OUT / "sheets").mkdir(exist_ok=True)
    for k in range(0, len(rev), 48):
        sheet = Image.new("RGB", (8 * 200, 6 * 220), "white")
        dr = ImageDraw.Draw(sheet)
        for j, r in enumerate(rev.iloc[k:k + 48].itertuples()):
            im = Image.open(r.path).convert("RGB")
            im.thumbnail((196, 196))
            x, y = 200 * (j % 8), 220 * (j // 8)
            sheet.paste(im, (x + 2, y + 2))
            dr.text((x + 4, y + 200), f"{r.rid} {r.subset} {r.group[0]} {r.p_derm_core:.2f}", fill="black")
        sheet.save(OUT / "sheets" / f"sheet_{k // 48:03d}.jpg", quality=85)
    print(len(rev), "images for review in", OUT / "sheets")


def summary():
    """Estimate per subset: hits x P(derm | text hit) + non-hits x P(derm | control), where P = classifier flag rate
    x visually verified precision of the flags (review_verdicts.json: {"true": [rid, ...]}; rids not reviewed among
    text hits are not used). 95% intervals by bootstrap over sampled images and reviewed flags."""
    sc = pd.read_parquet(OUT / "scored.parquet")
    rev = pd.read_parquet(OUT / "review.parquet")
    true = set(json.load(open(OUT / "review_verdicts.json"))["true"])
    rev["true"] = rev.rid.isin(true)
    meta = {}
    for f in SCAN.glob("*.json"):
        s = f.stem.split("_")[0]
        d = json.load(open(f))
        m = meta.setdefault(s, {"rows": 0, "text_hits": 0, "shards": 0})
        m["rows"] += d["rows"]; m["text_hits"] += d["text_hits"]; m["shards"] += 1
    rng = np.random.default_rng(0)
    R, boot_tot = {}, 0
    for s, m in meta.items():
        out = {**m}
        parts = {}
        for g in ("text_hit", "control"):
            d = sc[(sc.subset == s) & (sc.group == g)]
            x = (d.flag & ~d.nudity_flag).values  # nudity-flagged images are never viewed: counted as not derm
            v = rev[(rev.subset == s) & (rev.group == g)].true.values
            parts[g] = (x, v)
            out[f"{g}_images"] = int(len(x)); out[f"{g}_flagged"] = int(x.sum())
            out[f"{g}_flags_reviewed"] = int(len(v)); out[f"{g}_flags_true"] = int(v.sum())
        n_hit, n_non = m["text_hits"], m["rows"] - m["text_hits"]
        def est(xh, vh, xc, vc):
            ph = xh.mean() * (vh.mean() if len(vh) else 0)
            pc = xc.mean() * (vc.mean() if len(vc) else 0)
            return n_hit * ph, n_non * pc
        eh, ec = est(*parts["text_hit"], *parts["control"])
        B = []
        for _ in range(2000):
            bs = [a[rng.integers(0, len(a), len(a))] if len(a) else a for a in (*parts["text_hit"], *parts["control"])]
            B.append(sum(est(*bs)))
        B = np.array(B)
        out.update({"est_derm_via_text_hits": int(eh), "est_derm_without_text_hit": int(ec), "est_derm_total": int(eh + ec),
                    "est_derm_total_95ci": [int(np.percentile(B, 2.5)), int(np.percentile(B, 97.5))],
                    "share_found_by_text_filter_pct": round(100 * eh / max(1, eh + ec), 1)})
        R[s] = out
        boot_tot = boot_tot + B
    R["total"] = {"rows": sum(R[s]["rows"] for s in meta), "text_hits": sum(R[s]["text_hits"] for s in meta),
                  "est_derm_total": sum(R[s]["est_derm_total"] for s in meta),
                  "est_derm_total_95ci": [int(np.percentile(boot_tot, 2.5)), int(np.percentile(boot_tot, 97.5))]}
    R["note"] = ("Classifier recall is below 1 (0.95 on the gold set), and images flagged by the nudity screen are counted "
                 "as not dermatological (never viewed), so the estimates are conservative. Precision of flags verified "
                 "visually.")
    json.dump(R, open(OUT / "summary.json", "w"), indent=1)
    print(json.dumps(R, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["scan", "fetch", "score", "summary"])
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    if a.stage == "scan":
        scan(a.workers)
    else:
        {"fetch": fetch, "score": score, "summary": summary}[a.stage]()


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))
    main()
