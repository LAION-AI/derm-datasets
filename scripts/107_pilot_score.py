"""Score the realism-pilot arms (work/synthetic/realism_pilot/<arm>/): realism probe v1 + v2 (106_realism_probe.py)
and the quality gate (86_eval_synthetic.py; diagnosis legibility, Fitzpatrick, panel / blank, nudity screen) - reported
separately, because the gate rewards legibility, not realism. Arm A = the batch-5 images of the same 80 pilot rows.
Writes work/synthetic/realism_pilot/scores.parquet (one row per arm x image) and prints a per-arm summary.
Usage: python 107_pilot_score.py [arm ...]       (default: every arm dir with a manifest)
"""
import importlib
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from common import WORK

PILOT = WORK / "synthetic/realism_pilot"
PY = sys.executable


def arm_manifests(names):
    rows = pd.read_parquet(WORK / "synthetic/realism_pilot_rows.parquet")
    out = {}
    a = rows.copy(); a["path"] = [str(WORK / f"synthetic/batch5_moss/img/{s}.jpg") for s in a.synth_id]
    a = a[[Path(p).exists() for p in a.path]]
    if not names or "A_old_batch5" in names:
        out["A_old_batch5"] = (a, WORK / "synthetic/batch5_moss/eval/per_image.parquet")
    for d in sorted(PILOT.iterdir()):
        if d.is_dir() and (d / "manifest.parquet").exists() and (not names or d.name in names):
            m = pd.read_parquet(d / "manifest.parquet")
            m = m[[Path(p).exists() for p in m.path]]
            ev = d / "eval/per_image.parquet"
            if not ev.exists() or ev.stat().st_mtime < (d / "manifest.parquet").stat().st_mtime:
                subprocess.run([PY, "86_eval_synthetic.py", "--manifest", str(d / "manifest.parquet"), "--out", str(d / "eval"),
                                "--review-n", "0"], check=True, cwd=Path(__file__).parent, stdout=subprocess.DEVNULL)
            out[d.name] = (m, ev)
    return out


def main():
    names = sys.argv[1:]   # read before 106.embed(), which resets sys.argv
    rp = importlib.import_module("106_realism_probe")
    arms = arm_manifests(names)
    res = []
    for name, (m, ev) in arms.items():
        paths = m.path.tolist()
        s = {}
        for ver, f in [("v1", "realism_probe.joblib"), ("v2", "realism_probe_v2.joblib")]:
            rp.OUT = WORK / "models" / f
            rp.BG_GREY = ver == "v2"
            s[ver] = rp.score(paths)
        E = pd.read_parquet(ev).set_index("synth_id")
        E = E[["pass", "dx_rank", "dx_ok", "fst_ok", "panel_flag", "degenerate", "nudity_flag"]]
        r = m[["synth_id", "diagnosis", "grp", "fst_target"] if "grp" in m else ["synth_id", "diagnosis", "fst_target"]].copy()
        r["arm"] = name; r["path"] = paths; r["p_real_v1"] = s["v1"]; r["p_real_v2"] = s["v2"]
        res.append(r.join(E, on="synth_id"))
    R = pd.concat(res, ignore_index=True)
    f = PILOT / "scores.parquet"
    if f.exists() and names:
        old = pd.read_parquet(f); R = pd.concat([old[~old.arm.isin(R.arm.unique())], R], ignore_index=True)
    R.to_parquet(f, index=False)
    common = set.intersection(*[set(g.synth_id) for _, g in R.groupby("arm")])
    print(f"rows present in every arm: {len(common)}")
    S = R.groupby("arm").agg(n=("synth_id", "size"), p_real_v1=("p_real_v1", "mean"), p_real_v2=("p_real_v2", "mean"),
                             p_real_v2_med=("p_real_v2", "median"), gate_pass=("pass", "mean"),
                             dx_rank_med=("dx_rank", "median"), fst_ok=("fst_ok", "mean"), panel=("panel_flag", "mean"),
                             blank=("degenerate", "mean"))
    print(S.round(3).to_string())
    C = R[R.synth_id.isin(common)].groupby("arm")[["p_real_v1", "p_real_v2", "pass", "dx_rank"]].agg(
        {"p_real_v1": "mean", "p_real_v2": "mean", "pass": "mean", "dx_rank": "median"})
    print("\nsame rows in every arm:\n" + C.round(3).to_string())


if __name__ == "__main__":
    main()
