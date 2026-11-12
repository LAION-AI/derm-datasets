"""Recompute linear-probe results for every evaluated model from its cached benchmark features
(work/eval/feats_<name>.npz), with one common, wider regularisation grid (the first evaluation
selected the grid edge C=30 on all benchmarks). Standardised features, L2 logistic regression,
C chosen on the LP val split, metrics on the LP test split.

Writes work/eval/lp_all.json and patches lp_* keys of each work/eval/<name>.json (old values kept as lp0_*).
"""
import glob
import json
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

from common import WORK

GRID = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]   # on standardised features


def main():
    B = pd.read_parquet(WORK / "bench.parquet")
    out = {}
    for f in sorted(glob.glob(str(WORK / "eval/feats_*.npz"))):
        name = os.path.basename(f)[6:-4]
        z = np.load(f, allow_pickle=True)
        rel = lambda p: p.split("/wbs/", 1)[-1]   # same files on the workstation and the GPU server
        pos = {rel(p): i for i, p in enumerate(z["paths"])}
        Fx = z["F"].astype(np.float32)
        res = {}
        for bench in ["PAD", "HAM", "F17K", "Daffodil"]:
            d = B[(B.bench == bench) & (B.task == "LP")]
            d = d.assign(path=d.path.map(rel))
            if not d.path.isin(pos).all():
                print("  missing features:", name, bench, int((~d.path.isin(pos)).sum())); continue
            sp = {s: (Fx[[pos[p] for p in d[d.split == s].path]], d[d.split == s].label.values) for s in ["train", "val", "test"]}
            sc = StandardScaler().fit(sp["train"][0])
            X = {s: sc.transform(v[0]) for s, v in sp.items()}
            best = max((accuracy_score(sp["val"][1], LogisticRegression(C=C, max_iter=5000).fit(X["train"], sp["train"][1])
                                       .predict(X["val"])), C) for C in GRID)
            m = LogisticRegression(C=best[1], max_iter=5000).fit(X["train"], sp["train"][1])
            pr = m.predict(X["test"])
            res[f"lp_{bench}_acc"] = accuracy_score(sp["test"][1], pr)
            res[f"lp_{bench}_bal_acc"] = balanced_accuracy_score(sp["test"][1], pr)
            res[f"lp_{bench}_f1"] = f1_score(sp["test"][1], pr, average="macro")
            res[f"lp_{bench}_C"] = best[1]
        if len(res) < 16:   # all four benchmarks required, otherwise leave the json untouched
            print("  skip", name); continue
        res["lp_mean_acc"] = float(np.mean([res[k] for k in res if k.endswith("_acc") and "bal" not in k]))
        out[name] = res
        jf = WORK / f"eval/{name}.json"
        if jf.exists():
            j = json.load(open(jf))
            for k in [k for k in list(j) if k.startswith("lp_") and not k.startswith("lp0_")]:
                j.setdefault("lp0_" + k[3:], j[k])
            j.update(res)
            json.dump(j, open(jf, "w"), indent=1)
        print(name, {k: round(v, 3) for k, v in res.items() if k.endswith("acc")})
    json.dump(out, open(WORK / "eval/lp_all.json", "w"), indent=1)


if __name__ == "__main__":
    main()
