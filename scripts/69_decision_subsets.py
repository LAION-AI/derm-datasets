"""Decision models on identical image subsets.

clef (27B) and clef-omni were evaluated on 300 random test images per benchmark (62_eval_decision.py --limit 300,
sample seed 0); d1-3B and clef-flash on all test images. This recomputes all four on the same 300-image subsets
from the cached per-image probabilities (work/eval/<model>_<bench>_probs.parquet). The full-set accuracy is
recomputed as a check against the stored results.

Writes work/eval/decision_subset300.json: {model: {bench: [acc, balanced acc, n, full-set acc]}} (in %).
"""
import json

import numpy as np
import pandas as pd

from common import WORK

MODELS = ["d1-3B", "clef-flash", "clef27_300", "clefomni_300", "h2o-lightning-4b", "pplx-decider-27b_300"]


def main():
    B = pd.read_parquet(WORK / "bench.parquet")
    B = B[B.task == "ZS"]
    rel = lambda p: p.split("/wbs/", 1)[-1]   # same files on Mac and donv
    B = B.assign(rp=B.path.map(rel))
    out = {}
    for bench in ["PAD", "HAM", "F17K", "Daffodil"]:
        sub = set(pd.read_parquet(WORK / f"eval/clef27_300_{bench}_probs.parquet").path.map(rel))
        assert sub == set(pd.read_parquet(WORK / f"eval/clefomni_300_{bench}_probs.parquet").path.map(rel))
        lab = B[B.bench == bench].set_index("rp").label
        for m in MODELS:
            if not (WORK / f"eval/{m}_{bench}_probs.parquet").exists():
                continue
            d = pd.read_parquet(WORK / f"eval/{m}_{bench}_probs.parquet")
            d["rp"] = d.path.map(rel)
            P = np.stack(d.probs.values)
            y = lab[d.rp].values.astype(int)
            s = d.rp.isin(sub).values
            hit = P[s].argmax(1) == y[s]
            out.setdefault(m, {})[bench] = [round(100 * hit.mean(), 1),
                                            round(100 * pd.Series(hit).groupby(y[s]).mean().mean(), 1),
                                            int(s.sum()), round(100 * (P.argmax(1) == y).mean(), 1)]
    json.dump(out, open(WORK / "eval/decision_subset300.json", "w"), indent=1)
    for m, v in out.items():
        print(m, v, "mean acc", round(np.mean([x[0] for x in v.values()]), 1))


if __name__ == "__main__":
    main()
