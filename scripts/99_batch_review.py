"""Post-process a generated SynthDerm batch (96_generate_synthetic.py output): quality gate -> safety split -> import into
the expert annotation app (annotator/) -> summary.

1. 86_eval_synthetic.py on <batch>/manifest.parquet (skipped if <batch>/eval/per_image.parquet is newer than the manifest).
2. Safety split: nudity-flagged images of MINORS are deleted unseen and never imported; nudity-flagged adult images are
   imported with tier `nudity_review` (gated in the app, gated at release; the user, a board-certified dermatologist,
   reviews complete medical images); all others are imported as open tier.
3. Summary: gate metrics overall, by Fitzpatrick group, by prompt description source (visual vs taxonomy), and for
   paired A/B rows (column ab_pair: same row and seed, visual vs taxonomy description) the paired diagnosis-rank result.
   Written to <batch>/review_summary.json.
Usage: python 99_batch_review.py ../work/synthetic/batch1 --set batch1
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
MINOR = {"infant", "child", "adolescent"}
G = ["fst_pred", "fst_ok", "dx_rank", "dx_ok", "p_exposed", "nudity_flag", "memorised", "pass"]


def screen(paths):
    """Nudity probability (same CLIP screen and threshold as 86) and Fitzpatrick head for rows the gate skipped
    (diagnoses without a planned concept id)."""
    import importlib, types
    import joblib
    import numpy as np
    sys.path.insert(0, str(ROOT / "scripts"))
    sys.argv = [sys.argv[0]]
    g86 = importlib.import_module("86_eval_synthetic")
    ev = importlib.import_module("60_eval_clip")
    base = ev.load_backend(types.SimpleNamespace(backend="openclip", pretrained="openai", model="ViT-B-16-quickgelu"), g86.device())
    X = base.encode_images(list(paths)).astype(np.float32)
    Z = 100 * X @ base.encode_texts(g86.NUDITY_LABELS).T
    P = np.exp(Z - Z.max(1, keepdims=True)); P /= P.sum(1, keepdims=True)
    fst = np.clip(np.rint(joblib.load(g86.WORK / "heads_models.joblib")["fst"].predict(X)), 1, 6).astype(int)
    return P[:, 1:].sum(1), fst, g86.NUDITY_FLAG


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("batch")
    ap.add_argument("--set", required=True)
    ap.add_argument("--no-import", action="store_true")
    ap.add_argument("--import-n", type=int, default=0, help="import only a review sample: open <= N, gated <= N/4, every diagnosis once (0 = all)")
    a = ap.parse_args()
    b = Path(a.batch).resolve()
    man_f, ev_dir = b / "manifest.parquet", b / "eval"
    if not (ev_dir / "per_image.parquet").exists() or (ev_dir / "per_image.parquet").stat().st_mtime < man_f.stat().st_mtime:
        subprocess.run([PY, str(ROOT / "scripts/86_eval_synthetic.py"), "--manifest", str(man_f), "--out", str(ev_dir),
                        "--review-n", "0"], check=True, cwd=ROOT / "scripts")
    M = pd.read_parquet(man_f)
    E = pd.read_parquet(ev_dir / "per_image.parquet").set_index("synth_id")[G]
    M = M.join(E.rename(columns=lambda c: "gate_" + c), on="synth_id")
    M = M[[Path(p).exists() for p in M.path]]
    un = M.gate_nudity_flag.isna()   # unmapped diagnoses: no gate result -> own nudity + skin-type screen
    if un.any():
        pe, fst, thr = screen(M.loc[un, "path"])
        M.loc[un, "gate_p_exposed"], M.loc[un, "gate_fst_pred"] = pe, fst
        M.loc[un, "gate_nudity_flag"] = pe >= thr
        M.loc[un, "gate_fst_ok"] = (M.loc[un, "gate_fst_pred"] - M.loc[un, "fst_target"]).abs() <= 1
    flag = M.gate_nudity_flag.astype(bool)
    unscreened = pd.Series(False, index=M.index)
    minors = M[(flag | unscreened) & M.age_group.isin(MINOR)]
    for p in minors.path:   # deleted unseen, never imported
        Path(p).unlink(missing_ok=True)
    M = M.drop(minors.index)
    flag, unscreened = flag.loc[M.index], unscreened.loc[M.index]
    open_ = M[~flag & ~unscreened]
    gated = M[flag | unscreened].assign(tier="nudity_review", synth_id=lambda d: a.set + "_" + d.synth_id)
    open_.to_parquet(b / "manifest_open.parquet", index=False)
    gated.to_parquet(b / "manifest_nudity_review.parquet", index=False)
    if a.import_n:   # review sample: spread over diagnoses and skin types, the rest stays in the batch folder
        frac = min(1.0, a.import_n / max(1, len(M)))
        def pick(d, cap):   # one image per diagnosis, then random rows up to the cap
            d = d.sample(frac=1.0, random_state=0)
            first = d.drop_duplicates("lifecycle_key").head(cap)
            return pd.concat([first, d.drop(first.index).head(max(0, cap - len(first)))])
        open_s = pick(open_, a.import_n) if len(open_) else open_
        gated_s = pick(gated, max(1, a.import_n // 4)) if len(gated) else gated
    else:
        open_s, gated_s = open_, gated
    open_s.to_parquet(b / "manifest_open_import.parquet", index=False)
    gated_s.to_parquet(b / "manifest_nudity_review_import.parquet", index=False)
    if not a.no_import:
        imp = [PY, str(ROOT / "annotator/import_items.py")]
        subprocess.run(imp + [str(b / "manifest_open_import.parquet"), "--set", a.set], check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
        if len(gated_s):
            subprocess.run(imp + [str(b / "manifest_nudity_review_import.parquet"), "--set", a.set + "_nudity_review",
                                  "--allow-gated"], check=True, cwd=ROOT, stdout=subprocess.DEVNULL)

    S = M[M.gate_pass.notna()]
    rate = lambda d: {"n": len(d), "pass": round(float(d.gate_pass.mean()), 3), "dx_ok": round(float(d.gate_dx_ok.mean()), 3),
                      "fst_ok": round(float(d.gate_fst_ok.mean()), 3), "dx_rank_median": float(d.gate_dx_rank.median())}
    summ = {"images": len(M) + len(minors), "deleted_minor_nudity_or_unscreened": len(minors), "open": len(open_),
            "nudity_review": int(flag.sum()), "no_dx_check_unmapped_dx": int(un.sum()), "gate": rate(S),
            "by_fst": {int(f): rate(d) for f, d in S.groupby("fst_target")}}
    if "ab_pair" not in S:
        S = S.assign(ab_pair=None)
    if "desc_source" in S:
        summ["by_desc_source"] = {k: rate(d) for k, d in S[S.ab_pair.isna()].groupby("desc_source")}
    if "ab_pair" in S and S.ab_pair.notna().any():
        ab = S[S.ab_pair.notna()].pivot_table(index="ab_pair", columns="desc_source", values=["gate_dx_rank", "gate_pass"])
        ab = ab.dropna()
        summ["ab_pairs"] = {"pairs": len(ab),
                            "visual_better_rank": round(float((ab["gate_dx_rank"]["visual"] < ab["gate_dx_rank"]["taxonomy"]).mean()), 3),
                            "pass_visual": round(float(ab["gate_pass"]["visual"].mean()), 3),
                            "pass_taxonomy": round(float(ab["gate_pass"]["taxonomy"].mean()), 3),
                            "median_rank_visual": float(ab["gate_dx_rank"]["visual"].median()),
                            "median_rank_taxonomy": float(ab["gate_dx_rank"]["taxonomy"].median())}
    worst = S.groupby("diagnosis").gate_pass.agg(["size", "mean"]).query("size >= 6").sort_values("mean")
    summ["worst_diagnoses"] = worst.head(15).round(3).reset_index().to_dict("records")
    summ["best_diagnoses"] = worst.tail(10).round(3).reset_index().to_dict("records")
    json.dump(summ, open(b / "review_summary.json", "w"), indent=1, default=str)
    print(json.dumps({k: v for k, v in summ.items() if "diagnoses" not in k}, indent=1, default=str))


if __name__ == "__main__":
    main()
