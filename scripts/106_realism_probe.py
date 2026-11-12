"""Realism probe for SynthDerm: p(real) from a logistic regression on CLIP ViT-B/16 embeddings, trained to separate
real clinical photographs (our corpus: clinical, canonical; plus permissive references) from synthetic images
(batches 1-5). The gate's diagnosis check measures legibility, not realism - caricature-like lesions pass it
(pilot review 2026-11-10) - so realism needs its own measure.

Shortcut control: every image is centre-cropped (60% of the shorter side, which removes most of the background; the
white studio background is a known synthetic tell that the realism prompts address separately), resized to 224 px and
re-encoded as JPEG q75, so resolution and compression differences between the sources cannot be used.
Usage: python 106_realism_probe.py train            # -> work/models/realism_probe.joblib (+ CV AUC)
       python 106_realism_probe.py score <manifest.parquet|dir> [...]   # p_real per image
"""
import io
import sys
import types
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from PIL import Image

from common import DATASET, WORK, device

OUT = WORK / "models/realism_probe_v2.joblib"   # v1 (no background greying, unfiltered refs): realism_probe.joblib
_BE = None


BG_GREY = True   # v2: near-white, unsaturated pixels -> neutral grey in every image (white studio is not a usable cue)


def prep(path):
    im = Image.open(path).convert("RGB")
    w, h = im.size; s = int(min(w, h) * 0.6)
    im = im.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s)).resize((224, 224), Image.BICUBIC)
    if BG_GREY:
        a = np.asarray(im, dtype=np.int16)
        bg = (a.min(2) > 225) & ((a.max(2) - a.min(2)) < 18)
        a[bg] = 128
        im = Image.fromarray(a.astype(np.uint8))
    b = io.BytesIO(); im.save(b, "JPEG", quality=75); b.seek(0)
    return Image.open(b).convert("RGB")


def embed(paths, tmp=WORK / "tmp_realism"):
    """CLIP embeddings of the normalised crops (written to a temp dir because the backend encodes from paths)."""
    global _BE
    import importlib
    sys.argv = [sys.argv[0]]
    ev = importlib.import_module("60_eval_clip")
    if _BE is None:
        _BE = ev.load_backend(types.SimpleNamespace(backend="openclip", pretrained="openai", model="ViT-B-16-quickgelu"), device())
    tmp.mkdir(parents=True, exist_ok=True)
    out = []
    for i in range(0, len(paths), 256):
        chunk = []
        for j, p in enumerate(paths[i:i + 256]):
            q = tmp / f"{i + j}.jpg"
            try:
                prep(p).save(q, quality=95); chunk.append(str(q))
            except Exception:
                chunk.append(None)
        ok = [c for c in chunk if c]
        X = np.asarray(_BE.encode_images(ok), dtype=np.float32) if ok else np.zeros((0, 512), np.float32)
        it = iter(X)
        out += [next(it) if c else np.full(X.shape[1] if len(X) else 512, np.nan, np.float32) for c in chunk]
    return np.vstack(out)


def train(n_real=3000, n_ref=1000, n_syn=4000):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_predict
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(0)
    I = pd.read_parquet(DATASET / "metadata/images.parquet", columns=["file", "image_type", "is_canonical"])
    I = I[(I.image_type == "clinical") & I.is_canonical]
    real = [str(DATASET / f) for f in I.file.sample(n_real, random_state=0)]
    R = pd.read_parquet(WORK / "refs/permissive_refs.parquet")
    R = R[R.usable & R.lesion_photo & ~R.has_text.fillna(True) & ~R.has_watermark.fillna(True)]
    # v2: only references the Splash vision pass confirmed as clinical skin photographs (no oral / intra-operative views)
    import json as _j
    desc = _j.load(open(WORK / "lora_data/multi/_descriptions.json"))
    pre = _j.load(open(WORK / "refs/ref_precheck.json")) if (WORK / "refs/ref_precheck.json").exists() else {}
    okd = {k for k, v in desc.items() if "error" not in v and v.get("is_clinical_skin_photo")}
    okp = {k for k, v in pre.items() if "error" not in v and v.get("single_photograph") and v.get("skin_condition_visible") and not v.get("overlay_graphics")}
    R = R[R.sha256.isin(okd) & (~R.sha256.isin(pre) | R.sha256.isin(okp))]
    ref = R.path.sample(min(n_ref, len(R)), random_state=0).tolist()
    pilot = set(pd.read_parquet(WORK / "synthetic/realism_pilot_rows.parquet").synth_id)
    syn = []
    for b in ["batch1", "batch2", "batch3", "batch4", "batch5_moss"]:
        fs = [f for f in (WORK / "synthetic" / b / "img").glob("*.jpg") if f.stem not in pilot]
        syn += [str(f) for f in rng.choice(fs, size=min(n_syn // 5, len(fs)), replace=False)]
    # unseen real test set (never used for fitting): more corpus photos + permissive references
    used = set(real) | set(ref)
    real_test = [str(DATASET / f) for f in I.file.sample(n_real + 600, random_state=1) if str(DATASET / f) not in used][:500]
    ref_test = [x for x in R.path.sample(min(len(R), n_ref + 400), random_state=1) if x not in used][:300]
    # diagnosis of each synthetic image (from the batch plans) for grouped cross-validation
    dx = {}
    for b in ["batch1", "batch2", "batch3", "batch4", "batch5_moss"]:
        pl = pd.read_parquet(WORK / "synthetic" / b / "plan.parquet", columns=["synth_id", "lifecycle_key"])
        dx.update(dict(zip(pl.synth_id, pl.lifecycle_key)))
    paths = real + ref + syn
    y = np.array([1] * (len(real) + len(ref)) + [0] * len(syn))
    src = np.array(["corpus"] * len(real) + ["permissive_ref"] * len(ref) + ["synthetic"] * len(syn))
    X = embed(paths)
    ok = ~np.isnan(X).any(1); X, y, src = X[ok], y[ok], src[ok]
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    clf = LogisticRegression(C=1.0, max_iter=3000)
    from sklearn.model_selection import GroupKFold
    pa = np.array(paths)[ok]
    groups = np.array([dx.get(Path(q).stem, "dx?") if t == "synthetic" else f"real{i % 40}" for i, (q, t) in enumerate(zip(pa, src))])
    p = cross_val_predict(clf, X, y, cv=GroupKFold(5), groups=groups, method="predict_proba")[:, 1]
    print("5-fold CV AUC, synthetic diagnoses held out by group:", round(roc_auc_score(y, p), 4))
    for s in ["corpus", "permissive_ref", "synthetic"]:
        print(f"  mean p_real {s:15s} {p[src == s].mean():.3f}  (n={int((src == s).sum())})")
    syn_b = np.array([q.split("/synthetic/")[1].split("/")[0] if "/synthetic/" in q else "" for q in np.array(paths)[ok]])
    for b in ["batch1", "batch2", "batch3", "batch4", "batch5_moss"]:
        m = syn_b == b
        if m.any(): print(f"  mean p_real {b:15s} {p[m].mean():.3f}")
    clf.fit(X, y)
    for name, tp in [("unseen real corpus photos", real_test), ("unseen permissive references", ref_test)]:
        Xt = embed(tp); okt = ~np.isnan(Xt).any(1); Xt = Xt[okt] / np.linalg.norm(Xt[okt], axis=1, keepdims=True)
        pt = clf.predict_proba(Xt)[:, 1]
        print(f"  {name}: n={len(pt)} mean p_real {pt.mean():.3f} | share p_real > 0.5: {(pt > 0.5).mean():.3f}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, OUT)
    print("saved", OUT)


def score(paths):
    clf = joblib.load(OUT)
    X = embed(paths)
    ok = ~np.isnan(X).any(1)
    p = np.full(len(paths), np.nan)
    Xn = X[ok] / np.linalg.norm(X[ok], axis=1, keepdims=True)
    p[ok] = clf.predict_proba(Xn)[:, 1]
    return p


if __name__ == "__main__":
    if sys.argv[1] == "train":
        train()
    else:
        for a in sys.argv[2:]:
            a = Path(a)
            ps = pd.read_parquet(a).path.tolist() if a.suffix == ".parquet" else [str(f) for f in sorted(a.glob("*.jpg"))]
            p = score(ps)
            print(a, "n", int(np.isfinite(p).sum()), "mean p_real", round(float(np.nanmean(p)), 3), "median", round(float(np.nanmedian(p)), 3))
