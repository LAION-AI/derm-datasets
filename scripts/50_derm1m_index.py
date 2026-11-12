"""Index the downloaded Derm1M (HF redlessone/Derm1M, v2) for comparison.

Writes work/derm1m.parquet with src_path pointing into external/Derm1M-hf/extracted.
Derm1M itself is CC BY-NC 4.0 (gated); it is used here only for analysis and
is never copied into dataset/.
"""
import pandas as pd

from common import EXTERNAL, WORK

BASE = EXTERNAL / "Derm1M-hf"


def main():
    t = pd.read_csv(BASE / "Derm1M_v2_pretrain.csv", low_memory=False).assign(split="train")
    v = pd.read_csv(BASE / "Derm1M_v2_validation.csv", low_memory=False).assign(split="validation")
    d = pd.concat([t, v], ignore_index=True)
    d["src_path"] = [str(BASE / "extracted" / f) for f in d.filename]
    d["folder"] = d.filename.str.split("/").str[0]
    d.to_parquet(WORK / "derm1m.parquet", index=False)
    print(d.shape)
    print(d.groupby(["split", "source"]).size())


if __name__ == "__main__":
    main()
