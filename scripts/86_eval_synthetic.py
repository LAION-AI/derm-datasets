"""Automatic quality gate for generated (synthetic) dermatology images — protocol in
dataset/docs/SYNTHETIC_PROTOCOL.md, plan from 85_synthetic_plan.py.

Per image (OpenAI CLIP ViT-B/16 embeddings, the space of the dataset heads):
  * image type (32_heads type head): must be a clinical photograph
  * Fitzpatrick type (32_heads ridge head): |predicted - target| <= 1
  * memorisation: max cosine to every real train-split image (what a generator fine-tuned on the dataset or
    used with references has seen); >= 0.98 = near-duplicate (the dataset's own threshold; real test photographs
    reach at most 0.971) -> reject
  * diagnosis consistency: zero-shot rank of the intended diagnosis among all planned diagnoses, with a
    dermatology CLIP (default: CLIP fine-tuned on ours + Derm1M, else OpenAI CLIP). The rank cut-off is
    calibrated by --selftest so that 80% of real test photographs pass (work/synthetic/calibration.json).
Per diagnosis x skin-type cell: diversity (mean pairwise cosine; lower = more diverse) and CMMD
(Jayasumana et al. 2024: MMD with Gaussian kernel, sigma 10, scale 1000, on unit CLIP embeddings) against the
real reference images of that diagnosis.
Also writes a blinded expert-review sheet: stratified sample of gate-passing images, mixed 1:1 with real
reference images of the same diagnosis, shuffled; the key is written to a separate file.

Usage: python 86_eval_synthetic.py --manifest generated.parquet --out ../work/synthetic/eval_run1
       (manifest columns: synth_id, path, concept_id, fst_target)
       python 86_eval_synthetic.py --selftest --out ../work/synthetic/selftest
       (real test-split images posing as synthetic, plus train images that must be flagged as memorised)
"""
import argparse
import importlib
import json
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd

from common import DATASET, WORK, device, load_local_embeddings

PLAN = DATASET / "synthetic/plan"
DUP_COS, REAL_PASS = 0.98, 0.80
NUDITY_FLAG = 0.30   # open tier: CLIP probability of exposed genitals / breasts / buttocks at or above this fails the gate
NUDITY_LABELS = ["a clinical close-up photograph of skin on a face, arm, hand, leg, foot, back or chest",
                 "a photograph showing exposed genitals or pubic area", "a photograph showing bare female breasts",
                 "a photograph showing bare buttocks or the anal region"]   # same screen as 87_synthetic_examples.py
CALIB = WORK / "synthetic/calibration.json"


def cmmd(x, y, sigma=10.0, scale=1000.0):
    g = 1 / (2 * sigma ** 2)
    k = lambda a, b: np.exp(-g * np.clip((a * a).sum(1)[:, None] + (b * b).sum(1)[None] - 2 * a @ b.T, 0, None))
    return float(scale * (k(x, x).mean() + k(y, y).mean() - 2 * k(x, y).mean()))


def selftest_manifest(n_test=300, n_train=100):
    """Real images with known diagnosis: test split (unseen -> should pass the memorisation check; FST target =
    Fitzpatrick17k expert label where available, else the head estimate) and train split (must be flagged)."""
    ref = pd.read_parquet(PLAN / "references.parquet")   # train split by construction
    I = pd.read_parquet(DATASET / "metadata/images.parquet",
                        columns=["sha256", "split", "image_type", "is_canonical", "fst_pred", "fst_gold_f17k"])
    dx = pd.read_parquet(WORK / "dx_samples_ours.parquet").explode("concepts").dropna(subset=["concepts"])
    S = pd.read_parquet(DATASET / "metadata/samples.parquet", columns=["src_path", "sha256", "source_type", "file"])
    dx = dx.merge(S[S.source_type == "atlas"], on="src_path").merge(I, on="sha256")
    planned = set(ref.concept_id)
    te = dx[(dx.split == "test") & (dx.image_type == "clinical") & dx.is_canonical & dx.concepts.isin(planned)]
    te = te.drop_duplicates("sha256")
    gold = te[te.fst_gold_f17k.notna()]
    te = pd.concat([gold.sample(min(n_test // 2, len(gold)), random_state=0), te[te.fst_gold_f17k.isna()]]).head(n_test)
    te = te.assign(fst_target=te.fst_gold_f17k.fillna(te.fst_pred), fst_source=np.where(te.fst_gold_f17k.notna(), "f17k_expert", "head"))
    tr = ref.drop_duplicates("sha256").sample(n_train, random_state=0)
    m = pd.concat([pd.DataFrame(dict(path=[str(DATASET / f) for f in te.file], concept_id=te.concepts.astype(int),
                                     fst_target=te.fst_target.astype(int), fst_source=te.fst_source, kind="real_test")),
                   pd.DataFrame(dict(path=[str(DATASET / f) for f in tr.file], concept_id=tr.concept_id.astype(int),
                                     fst_target=tr.fst_pred.astype(int), fst_source="head", kind="real_train"))],
                  ignore_index=True)
    m.insert(0, "synth_id", [f"self_{i:04d}" for i in range(len(m))])
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest")
    ap.add_argument("--out", required=True)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--dx-model", default=str(WORK / "models/clip_union_3k/model.pt"))
    ap.add_argument("--review-n", type=int, default=300)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    M = selftest_manifest() if a.selftest else (pd.read_parquet(a.manifest) if a.manifest.endswith(".parquet")
                                                else pd.read_json(a.manifest, lines=True) if a.manifest.endswith(".jsonl")
                                                else pd.read_csv(a.manifest))
    tab = pd.read_csv(PLAN / "conditioning_table.csv").set_index("concept_id")
    M = M[M.concept_id.isin(tab.index)].reset_index(drop=True)
    M["diagnosis"] = M.concept_id.map(tab.concept_name)
    M["fst_group"] = pd.cut(M.fst_target, [0, 2, 4, 6], labels=["I-II", "III-IV", "V-VI"]).astype(str)

    sys.argv = [sys.argv[0]]
    ev = importlib.import_module("60_eval_clip")
    dev = device()
    base = ev.load_backend(types.SimpleNamespace(backend="openclip", pretrained="openai", model="ViT-B-16-quickgelu"), dev)
    X = base.encode_images(M.path.tolist()).astype(np.float32)

    import joblib
    H = joblib.load(WORK / "heads_models.joblib")
    prob = H["type"].predict_proba(X)
    classes = np.array(H["type"].classes_)
    M["image_type"] = classes[prob.argmax(1)]
    M["p_clinical"] = prob[:, list(classes).index("clinical")]
    M["fst_pred_raw"] = H["fst"].predict(X)
    M["fst_pred"] = np.clip(np.rint(M.fst_pred_raw), 1, 6).astype(int)
    M["fst_ok"] = (M.fst_pred - M.fst_target).abs() <= 1

    # collage / multi-panel check (panel_check.py): split images are not single clinical photographs -> fail
    pc = importlib.import_module("panel_check")
    M["panel_flag"] = [pc.panel_flag(p)[0] for p in M.path]
    # degenerate output (blank / near-uniform image, e.g. NaN decode) -> fail
    from PIL import Image as _I
    M["degenerate"] = [float(np.asarray(_I.open(p).convert("L").resize((128, 128)), dtype=np.float32).std()) < 6 for p in M.path]

    # nudity screen (open tier): flagged images fail and go to human review, never into a release
    Z = 100 * X @ base.encode_texts(NUDITY_LABELS).T
    Pn = np.exp(Z - Z.max(1, keepdims=True)); Pn /= Pn.sum(1, keepdims=True)
    M["p_exposed"] = Pn[:, 1:].sum(1)
    M["nudity_flag"] = M.p_exposed >= NUDITY_FLAG

    # memorisation: nearest real train-split image
    P, E = load_local_embeddings()
    train = set(pd.read_parquet(DATASET / "metadata/images.parquet", columns=["sha256", "split"]).query("split == 'train'").sha256)
    keep = P.sha256.isin(train).values
    P, E = P[keep].reset_index(drop=True), E[keep]
    best, arg = np.full(len(M), -1.0), np.zeros(len(M), int)
    for s in range(0, len(E), 50000):
        C = X @ E[s:s + 50000].astype(np.float32).T
        j = C.argmax(1); v = C[np.arange(len(M)), j]
        upd = v > best; best[upd], arg[upd] = v[upd], j[upd] + s
    M["nn_real_cos"], M["nn_real_sha256"] = best, P.sha256.values[arg]
    M["memorised"] = M.nn_real_cos >= DUP_COS

    # diagnosis consistency
    dxm = a.dx_model if Path(a.dx_model).exists() else "openai"
    be = base if dxm == "openai" else ev.load_backend(types.SimpleNamespace(backend="openclip", pretrained=dxm,
                                                                            model="ViT-B-16-quickgelu"), dev)
    Xd = X if be is base else be.encode_images(M.path.tolist()).astype(np.float32)
    names = tab.concept_name.tolist()
    W = ev.zs_classifier(be, names)
    S = Xd @ W
    col = M.concept_id.map({c: i for i, c in enumerate(tab.index)}).values
    M["dx_rank"] = (S > S[np.arange(len(M)), col][:, None]).sum(1) + 1
    if a.selftest:
        max_rank = int(np.ceil(M[M.kind == "real_test"].dx_rank.quantile(REAL_PASS)))
        json.dump(dict(max_rank=max_rank, real_pass=REAL_PASS, dx_model=a.dx_model, n_vocab=len(names)), open(CALIB, "w"), indent=1)
    else:
        max_rank = json.load(open(CALIB))["max_rank"] if CALIB.exists() else 10
    M["dx_ok"] = M.dx_rank <= max_rank
    M["pass"] = (M.image_type == "clinical") & ~M.panel_flag & ~M.degenerate & M.fst_ok & ~M.memorised & M.dx_ok & (a.selftest | ~M.nudity_flag)

    # per cell: diversity and CMMD vs real references of the diagnosis
    ref = pd.read_parquet(PLAN / "references.parquet")
    pos = {h: i for i, h in enumerate(P.sha256)}
    cells = []
    for (c, g), d in M.groupby(["concept_id", "fst_group"]):
        xi = X[d.index]
        r = ref[(ref.concept_id == c) & ref.sha256.isin(pos)]
        row = dict(concept_id=c, diagnosis=tab.concept_name[c], fst_group=g, n=len(d), pass_rate=float(d["pass"].mean()),
                   diversity_mean_cos=float((xi @ xi.T)[np.triu_indices(len(d), 1)].mean()) if len(d) > 1 else None)
        if len(d) >= 10 and len(r) >= 10:
            row["cmmd_vs_real"] = cmmd(xi, E[[pos[h] for h in r.sha256]].astype(np.float32))
        cells.append(row)
    C = pd.DataFrame(cells)

    M.to_parquet(out / "per_image.parquet", index=False)
    C.to_csv(out / "per_cell.csv", index=False)
    summ = dict(n=len(M), dx_model=dxm, dx_max_rank=max_rank, pass_rate=float(M["pass"].mean()),
                nudity_flagged=int(M.nudity_flag.sum()), panel_flagged=int(M.panel_flag.sum()), degenerate=int(M.degenerate.sum()),
                clinical=float((M.image_type == "clinical").mean()), fst_within1=float(M.fst_ok.mean()),
                fst_exact=float((M.fst_pred == M.fst_target).mean()), memorised=float(M.memorised.mean()),
                dx_top1=float((M.dx_rank == 1).mean()), dx_top10=float(M.dx_ok.mean()),
                dx_rank_median=float(M.dx_rank.median()), n_diagnoses_vocab=len(names),
                by_fst_group=M.groupby("fst_group")[["pass", "fst_ok", "dx_ok"]].mean().round(3).to_dict(orient="index"))
    if "kind" in M:
        summ["by_kind"] = M.groupby("kind")[["pass", "memorised", "fst_ok", "dx_ok", "nn_real_cos"]].mean().round(3).to_dict(orient="index")
        summ["by_kind"] = {k: {kk: float(vv) for kk, vv in v.items()} for k, v in summ["by_kind"].items()}
        g = M[(M.kind == "real_test") & (M.fst_source == "f17k_expert")]
        summ["fst_vs_expert"] = dict(n=len(g), within1=float(g.fst_ok.mean()), exact=float((g.fst_pred == g.fst_target).mean()))

    # blinded expert-review sheet (synthetic gate-passers + matched real references, 1:1)
    if "kind" not in M:
        ok = M[M["pass"]]
        k = min(a.review_n, len(ok))
        syn = ok.groupby("fst_group", group_keys=False).apply(
            lambda d: d.sample(max(1, round(k * len(d) / len(ok))), random_state=0)) if k else ok.iloc[:0]
        real = [ref[ref.concept_id == c].sample(1, random_state=i).iloc[0] for i, c in enumerate(syn.concept_id)
                if (ref.concept_id == c).any()]
        R = pd.concat([pd.DataFrame(dict(path=syn.path, diagnosis=syn.diagnosis, truth="synthetic", item=syn.synth_id)),
                       pd.DataFrame(dict(path=[str(DATASET / r.file) for r in real], diagnosis=[r.concept_name for r in real],
                                         truth="real", item=[r.image_id for r in real]))]).sample(frac=1, random_state=0)
        R.insert(0, "review_id", [f"r{i:04d}" for i in range(len(R))])
        R[["review_id", "truth", "item", "path"]].to_csv(out / "review_key.csv", index=False)
        sheet = R[["review_id", "path", "diagnosis"]].copy()
        for q in ["real_or_synthetic (R/S)", "diagnosis_plausible (1-5)", "skin_type_I-VI", "artefacts (none/minor/major)",
                  "usable_for_training (y/n)", "comment"]:
            sheet[q] = ""
        sheet.to_csv(out / "review_sheet.csv", index=False)
        summ["review_items"] = len(R)
    json.dump(summ, open(out / "summary.json", "w"), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
