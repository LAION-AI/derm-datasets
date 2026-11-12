"""Compute per-file image facts for every src_path in work/*.parquet sample tables.

sha256 (exact identity), decode check, format, width/height, mode, and two
64-bit perceptual hashes (pHash, dHash) for near-duplicate detection.
Incremental: rows already present in work/image_info.parquet are skipped.
Usage: python 20_image_info.py atlases textbooks journals
"""
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor

import imagehash
import pandas as pd
from PIL import Image
from tqdm import tqdm

from common import WORK, sha256_file

Image.MAX_IMAGE_PIXELS = None
warnings.filterwarnings("ignore")
OUT = WORK / "image_info.parquet"


def info(path):
    r = {"src_path": path, "ok": False}
    try:
        r["sha256"] = sha256_file(path)
        with Image.open(path) as im:
            r.update(format=im.format, width=im.width, height=im.height, mode=im.mode,
                     n_frames=getattr(im, "n_frames", 1))
            im.seek(0)
            g = im.convert("RGB")
            r["phash"] = str(imagehash.phash(g))
            r["dhash"] = str(imagehash.dhash(g))
            r["ok"] = True
    except Exception as e:  # truncated / not an image / error page saved as .jpg
        r["error"] = f"{type(e).__name__}: {e}"[:200]
    return r


def main(tables):
    paths = set()
    for t in tables:
        paths |= set(pd.read_parquet(WORK / f"{t}.parquet", columns=["src_path"]).src_path)
    done = pd.read_parquet(OUT) if OUT.exists() else pd.DataFrame(columns=["src_path"])
    todo = sorted(paths - set(done.src_path))
    print(f"{len(paths)} paths, {len(todo)} to process")
    res = []
    with ProcessPoolExecutor(10) as ex:
        for r in tqdm(ex.map(info, todo, chunksize=64), total=len(todo), mininterval=30):
            res.append(r)
    out = pd.concat([done, pd.DataFrame(res)], ignore_index=True) if res else done
    out.to_parquet(OUT, index=False)
    print("ok:", int(out.ok.sum()), "failed:", int((~out.ok.astype(bool)).sum()))


if __name__ == "__main__":
    main(sys.argv[1:])
