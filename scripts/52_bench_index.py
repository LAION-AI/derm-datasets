"""Index the Derm1M downstream benchmarks (PAD, HAM, F17K, Daffodil, SkinCAP)
plus expert Fitzpatrick labels from the official fitzpatrick17k.csv.

Writes work/bench.parquet: bench, task (ZS = zero-shot test set, LP = linear-probe
splits), split, path, label (int), diag (class name), fst (1-6 or NaN), url.
"""
import pandas as pd

from common import EXTERNAL, WORK

B = EXTERNAL / "derm1m-bench"


def main():
    f17 = pd.read_csv(EXTERNAL / "fitzpatrick17k/fitzpatrick17k.csv")
    f17 = f17.set_index("md5hash")
    rows = []
    for bench in ["PAD", "HAM", "F17K", "Daffodil"]:
        for task in ["ZS", "LP"]:
            d = pd.read_csv(B / f"data/meta/{bench}-{task}.csv")
            for r in d.itertuples(index=False):
                r = r._asdict()
                fst, url = None, None
                if bench == "F17K":
                    key = r["image_path"].split("/")[-1][:-4]
                    if key in f17.index:
                        v = f17.loc[key, "fitzpatrick_scale"]
                        fst = int(v) if v > 0 else None
                        url = f17.loc[key, "url"]
                if bench == "PAD" and pd.notna(r.get("fitspatrick")):
                    fst = int(r["fitspatrick"])
                rows.append(dict(bench=bench, task=task, split=r.get("split"), path=str(B / r["image_path"]),
                                 label=int(r["label"]), diag=r["diag"] if "diag" in r else r.get("diagnostic"),
                                 fst=fst, url=url))
    sc = pd.read_csv(B / "data/meta/skincap.csv")
    for r in sc.itertuples(index=False):
        fst = getattr(r, "fitzpatrick_scale")
        rows.append(dict(bench="SkinCAP", task="RET", split="test",
                         path=str(B / "data/skin_cap/images" / str(r.skincap_file_path)),
                         label=-1, diag=r.disease, fst=int(fst) if pd.notna(fst) and fst > 0 else None,
                         url=getattr(r, "url", None), caption=r.caption_zh_polish_en))
    df = pd.DataFrame(rows)
    import os
    df["exists"] = df.path.map(os.path.exists)
    print(df.groupby(["bench", "task", "split"]).agg(n=("path", "size"), exists=("exists", "mean"),
                                                      fst=("fst", "count")))
    df.to_parquet(WORK / "bench.parquet", index=False)


if __name__ == "__main__":
    main()
