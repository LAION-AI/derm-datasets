"""Render numbered contact sheets (image grids) for manual review / labelling.

Usage: python tool_contact_sheet.py <paths.parquet|txt> <out_prefix> [--n 30] [--cols 6] [--tile 220]
Each tile is labelled with its row index in the input so labels can be written
back as {index: label}.
"""
import argparse

import pandas as pd
from PIL import Image, ImageDraw

Image.MAX_IMAGE_PIXELS = None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths")
    ap.add_argument("out")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--cols", type=int, default=6)
    ap.add_argument("--tile", type=int, default=220)
    ap.add_argument("--start", type=int, default=0)
    a = ap.parse_args()
    if a.paths.endswith(".parquet"):
        paths = pd.read_parquet(a.paths).path.tolist()
    else:
        paths = [l.strip() for l in open(a.paths) if l.strip()]
    T = a.tile
    for s in range(a.start, len(paths), a.n):
        chunk = paths[s:s + a.n]
        rows = (len(chunk) + a.cols - 1) // a.cols
        sheet = Image.new("RGB", (a.cols * T, rows * (T + 18)), "white")
        d = ImageDraw.Draw(sheet)
        for j, p in enumerate(chunk):
            x, y = (j % a.cols) * T, (j // a.cols) * (T + 18)
            try:
                im = Image.open(p)
                im.seek(0)
                im = im.convert("RGB")
                im.thumbnail((T - 4, T - 4))
                sheet.paste(im, (x + 2, y + 18))
            except Exception:
                d.text((x + 10, y + 60), "unreadable", fill="red")
            d.text((x + 3, y + 3), str(s + j), fill="black")
        sheet.save(f"{a.out}_{s:05d}.jpg", quality=85)
        if s - a.start >= a.n * 50:
            break


if __name__ == "__main__":
    main()
