"""Flag text annotations / watermarks in the permissive reference pool with Apple Vision OCR (scripts/tools/vision_ocr,
on-device). About 9% of reference photos carry text (panel letters, scale labels, timestamps); by default these are
excluded from lesion projection instead of being cleaned (81_clean_refs.py is kept for selected rare references).
Arrows without text are not detected here.
Adds columns ocr_words, has_text, has_watermark to work/refs/permissive_refs.parquet (cache: work/refs/ocr.jsonl).
"""
import json
import subprocess
from pathlib import Path

import pandas as pd

from common import WORK

TOOL = Path(__file__).parent / "tools/vision_ocr"
CACHE = WORK / "refs/ocr.jsonl"
WATERMARK = ("©", "copyright", "www", ".com", ".org", "http", "all rights", "reserved")


def main():
    R = pd.read_parquet(WORK / "refs/permissive_refs.parquet")
    done = {}
    if CACHE.exists():
        for l in open(CACHE):
            r = json.loads(l); done[r["file"]] = r
    todo = [p for p in R[R.usable & R.lesion_photo].path.unique() if p not in done]
    with open(CACHE, "a") as f:
        for i in range(0, len(todo), 200):
            out = subprocess.run([str(TOOL), *todo[i:i + 200]], capture_output=True, text=True).stdout.splitlines()
            for l in out:
                r = json.loads(l); done[r["file"]] = r; f.write(json.dumps(r) + "\n")
            print(i + 200, "/", len(todo), flush=True)
    words = R.path.map(lambda p: [w["text"] for w in done.get(p, {}).get("words", []) if w.get("conf", 0) >= 0.5])
    R["ocr_words"] = words.map(lambda w: " | ".join(w))
    R["has_text"] = words.map(bool)
    R["has_watermark"] = words.map(lambda w: any(k in t.lower() for t in w for k in WATERMARK))
    R.to_parquet(WORK / "refs/permissive_refs.parquet", index=False)
    u = R[R.usable & R.lesion_photo]
    print({"lesion_photos": len(u), "with_text": int(u.has_text.sum()), "watermark": int(u.has_watermark.sum()),
           "clean": int((~u.has_text).sum())})


if __name__ == "__main__":
    main()
