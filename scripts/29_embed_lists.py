"""Build the image lists that 30_embed.py embeds (unique sha256, readable images only).

  embed_local_batch1  atlases + textbooks
  embed_local_batch2  journal figures not in batch1 and not byte-identical to a Derm1M file
                      (historical: Derm1M files were queued first)
  embed_local_batch3  any remaining local image (journal figures byte-identical to Derm1M)
  embed_derm1m        Derm1M images
  embed_bench         benchmark images (PAD, HAM, F17K, Daffodil, SkinCAP)
Existing lists are never rewritten (their row order defines the saved embeddings).
Then run, e.g.: python 30_embed.py ../work/embed_local_batch1.parquet ../work/emb/local_b1
"""
import pandas as pd

from common import WORK


def save(df, name):
    f = WORK / f"{name}.parquet"
    if f.exists():
        print(name, "exists:", len(pd.read_parquet(f)))
        return
    df.reset_index(drop=True).to_parquet(f, index=False)
    print(name, "written:", len(df))


def main():
    i = pd.read_parquet(WORK / "image_info.parquet")
    i = i[i.ok.astype(bool)]
    tab = lambda t: set(pd.read_parquet(WORK / f"{t}.parquet", columns=["src_path"]).src_path)
    d1, atl, tb, jr = tab("derm1m"), tab("atlases"), tab("textbooks"), tab("journals")
    pick = lambda s: i[i.src_path.isin(s)].drop_duplicates("sha256")[["src_path", "sha256"]].rename(columns={"src_path": "path"})
    b1 = pick(atl | tb)
    der = pick(d1)
    save(b1, "embed_local_batch1")
    save(der, "embed_derm1m")
    b1 = pd.read_parquet(WORK / "embed_local_batch1.parquet")
    der = pd.read_parquet(WORK / "embed_derm1m.parquet")
    j = pick(jr)
    save(j[~j.sha256.isin(set(b1.sha256) | set(der.sha256))], "embed_local_batch2")
    b2 = pd.read_parquet(WORK / "embed_local_batch2.parquet")
    allloc = pick(atl | tb | jr)
    save(allloc[~allloc.sha256.isin(set(b1.sha256) | set(b2.sha256))], "embed_local_batch3")
    b = pd.read_parquet(WORK / "bench.parquet").drop_duplicates("path")[["path"]]
    save(b, "embed_bench")


if __name__ == "__main__":
    main()
