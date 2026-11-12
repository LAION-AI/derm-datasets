"""Collage / multi-panel detector for generated images (MOSS-DMD and klein sometimes split one image into 2-4
panels; the CLIP image-type head misses this). A panel boundary is a straight line through (nearly) the whole image,
between 15% and 85% of the width / height, that is either a gutter (>= 92% of its pixels near-white or near-black)
or a seam (the mean absolute step across the line is >= 4x the median step of all columns / rows and also clearly
larger than its neighbours'). Returns (flag, details)."""
import numpy as np
from PIL import Image


def _lines(g, axis):
    a = g if axis == 0 else g.T                      # lines are columns of a
    h, w = a.shape
    lo, hi = int(w * 0.15), int(w * 0.85)
    white = (a > 232).mean(0); black = (a < 18).mean(0)
    cand = [x for x in range(lo, hi) if white[x] >= 0.92 or black[x] >= 0.92]
    # a panel gutter is a THIN run (<= 3% of the width) with image content right next to it on both sides
    # (a wide background gap between two hands / feet / legs is not a gutter)
    gutter, runs = [], []
    for x in cand:
        if runs and x == runs[-1][-1] + 1: runs[-1].append(x)
        else: runs.append([x])
    for r in runs:
        l, rr = r[0] - 2, r[-1] + 2
        if len(r) <= max(2, int(0.03 * w)) and 0 <= l and rr < w:
            side_bg = lambda c: max(white[c], black[c])
            if side_bg(l) < 0.5 and side_bg(rr) < 0.5:
                gutter += r
    step = np.abs(np.diff(a, axis=1)).mean(0)        # mean step between column x and x+1
    med = np.median(step) + 1e-6
    # a seam is a hard step along (nearly) the whole line: most pixels on it jump strongly, not just a few
    jump = (np.abs(np.diff(a, axis=1)) > 25).mean(0)
    seam = [x for x in range(lo, hi - 1) if step[x] >= 4 * med and jump[x] >= 0.6
            and step[x] >= 2.5 * max(step[x - 3:x].max(), step[x + 1:x + 4].max())]
    return gutter, seam


def panel_flag(path, size=512):
    im = Image.open(path).convert("L"); im.thumbnail((size, size))
    g = np.asarray(im, dtype=np.float32)
    gc, sc = _lines(g, 0); gr, sr = _lines(g, 1)
    flag = bool(gc or gr or sc or sr)
    return flag, {"gutter_cols": len(gc), "gutter_rows": len(gr), "seam_cols": len(sc), "seam_rows": len(sr)}
