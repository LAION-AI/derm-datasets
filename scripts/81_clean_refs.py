"""Remove figure annotations (panel letters, arrows, asterisks, rulers, scale bars) from permissively licensed reference
images with an instruction edit (qwen_image_2.1-turbo on the Splash image server), with a fidelity check.

Only annotations are removed. Images with watermarks / copyright notices are NOT cleaned: removing copyright
management information is restricted (EU InfoSoc Directive Art. 7, US DMCA 1202) even when attribution is kept
elsewhere; such images are excluded from the reference pool instead. CC BY allows adaptations if changes are indicated:
every cleaned image is recorded as "modified: figure annotations removed" next to its attribution.

Pipeline: OCR (Tesseract, sparse text) on the original -> watermark / copyright text => excluded, not cleaned;
instruction edit -> resized to the original resolution -> COMPOSITED into the original (only the dilated, feathered
regions the edit changed STRONGLY, and only where the original looks graphic (achromatic near-black / near-white or
pure saturated colour - letters, arrows, rulers; never skin-toned lesion areas or jewellery), grown by hysteresis into
the letters' anti-aliased fringes; this removes global drift such as re-cropping) ->
rejected if the edit misaligned the image (median dE > 4), changed > 8% of it, if OCR still finds (or newly finds)
text, or if CLIP still sees an arrow annotation. Output: work/refs/cleaned/<sha256>.png + work/refs/cleaned.json.
Usage: python 81_clean_refs.py --sha <sha256> [...] | --sample N
"""
import argparse
import importlib
import json
import time

import numpy as np
import pandas as pd
from PIL import Image

from common import WORK

ex = importlib.import_module("87_synthetic_examples")
OUT = WORK / "refs/cleaned"
PROMPT = ("Remove all figure annotations from this clinical photograph: panel letters, numbers, arrows, arrowheads, asterisks, "
          "circles, rulers, scale bars and text labels. Fill those areas with the surrounding skin or background. Keep everything "
          "else exactly unchanged: the lesion, skin texture, colours, lighting, framing and resolution.")
MAX_MASK, MAX_DRIFT_P50, ARROW_MAX = 0.08, 4.0, 0.5
WATERMARK = ("©", "copyright", "www", ".com", ".org", "http", "all rights", "reserved", "dermnet", "elsevier", "springer", "wiley")


def ocr(img):
    """Words found by Tesseract (sparse-text mode; catches single panel letters). Returns [(text, conf, box)]."""
    import subprocess, tempfile, csv, io
    with tempfile.NamedTemporaryFile(suffix=".png") as f:
        img.convert("RGB").save(f.name)
        out = subprocess.run(["tesseract", f.name, "stdout", "--psm", "11", "tsv"], capture_output=True, text=True).stdout
    words = []
    for r in csv.DictReader(io.StringIO(out), delimiter="\t"):
        t = (r.get("text") or "").strip()
        try:
            conf = float(r.get("conf") or -1)
        except ValueError:
            conf = -1
        if t and conf >= 60 and any(ch.isalnum() or ch in "©®" for ch in t):
            words.append((t, conf, tuple(int(r[k]) for k in ("left", "top", "width", "height"))))
    return words


_CLIP = {}


def arrow_prob(img):
    """OpenAI CLIP zero-shot: probability that the photo still shows an arrow / pointer annotation."""
    import open_clip, torch
    if not _CLIP:
        m, _, tf = open_clip.create_model_and_transforms("ViT-B-16-quickgelu", pretrained="openai")
        tok = open_clip.get_tokenizer("ViT-B-16-quickgelu")
        with torch.no_grad():
            T = m.encode_text(tok(["a clinical photograph of skin with a black arrow annotation pointing at a lesion",
                                   "a clinical photograph of skin without any annotations"]))
        _CLIP.update(m=m.eval(), tf=tf, T=torch.nn.functional.normalize(T, dim=-1))
    with torch.no_grad():
        x = torch.nn.functional.normalize(_CLIP["m"].encode_image(_CLIP["tf"](img.convert("RGB"))[None]), dim=-1)
        return float((100 * x @ _CLIP["T"].T).softmax(-1)[0, 0])


def composite(src, edited, strong=22.0, max_blob=0.03):
    """Keep the original except in compact regions the edit changed strongly (annotations are small, high-contrast
    objects; the unwanted re-rendering of skin texture is diffuse and weak): both images blurred (sigma 1.5), dE >
    `strong`, connected components each < `max_blob` of the image, dilated and feathered."""
    import cv2
    from scipy import ndimage
    a = np.asarray(src.convert("RGB"), dtype=np.float32) / 255
    b = np.asarray(edited.convert("RGB").resize(src.size, Image.LANCZOS), dtype=np.float32) / 255
    blur = lambda x: cv2.GaussianBlur(x, (0, 0), 1.5)
    d = np.linalg.norm(cv2.cvtColor(blur(a), cv2.COLOR_RGB2LAB) - cv2.cvtColor(blur(b), cv2.COLOR_RGB2LAB), axis=2)
    drift = float(np.median(d))
    lab_, n = ndimage.label(d > strong)
    sizes = ndimage.sum(np.ones_like(d), lab_, index=np.arange(1, n + 1)) / d.size if n else np.array([])
    keep = np.isin(lab_, 1 + np.where(sizes < max_blob)[0]) if n else np.zeros_like(d, bool)
    # only GRAPHIC-looking regions of the original may be replaced (letters / arrows / rulers: near-black or near-white
    # and achromatic, or very saturated pure colours); skin-toned regions (possible lesions) and e.g. jewellery stay original
    La = cv2.cvtColor(a, cv2.COLOR_RGB2LAB)
    chroma = np.hypot(La[..., 1], La[..., 2])
    graphic = ((chroma < 12) & ((La[..., 0] < 22) | (La[..., 0] > 88))) | (chroma > 70)
    graphic = ndimage.binary_dilation(graphic, iterations=2)
    lab2, n2 = ndimage.label(keep)
    if n2:
        frac = ndimage.mean(graphic.astype(np.float32), lab2, index=np.arange(1, n2 + 1))
        keep = np.isin(lab2, 1 + np.where(frac >= 0.35)[0])
    # hysteresis: grow the compact strong changes into connected weaker changes (anti-aliased letter fringes, shadows)
    keep = ndimage.binary_propagation(keep, mask=ndimage.binary_dilation(d > 6, iterations=2))
    k = max(4, int(0.01 * max(src.size)))
    mask = ndimage.binary_dilation(keep, iterations=k).astype(np.float32)
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=k / 2)[..., None]
    out = Image.fromarray(np.uint8(np.clip(255 * (a * (1 - mask) + b * mask), 0, 255)))
    return out, float(mask.mean()), drift


def vision_text(img):
    """Words found by Apple Vision OCR (scripts/tools/vision_ocr; on-device)."""
    import subprocess, tempfile
    from pathlib import Path
    tool = Path(__file__).parent / "tools/vision_ocr"
    with tempfile.NamedTemporaryFile(suffix=".png") as f:
        img.convert("RGB").save(f.name)
        out = subprocess.run([str(tool), f.name], capture_output=True, text=True).stdout
    return [w["text"] for w in json.loads(out.splitlines()[0]).get("words", []) if w.get("conf", 0) >= 0.5]


def clean(row, api):
    import base64, io
    src = Image.open(row.path).convert("RGB")
    words = vision_text(src)
    if any(w in t.lower() for t in words for w in WATERMARK):
        return {"sha256": row.sha256, "ok": False, "reason": "watermark / copyright notice - excluded, not cleaned",
                "ocr_before": words}
    w, h = src.size
    s = 1024 / max(w, h)
    size = f"{max(256, int(w * s) // 16 * 16)}x{max(256, int(h * s) // 16 * 16)}"
    tmp = OUT / f"_in_{row.sha256[:16]}.png"
    src.save(tmp)
    d = ex.retry(lambda: ex.post_edit_sized(f"{api}/edits", tmp, PROMPT, size))
    tmp.unlink()
    edited = Image.open(io.BytesIO(base64.b64decode(d["data"][0]["b64_json"])))
    out, mask_share, drift = composite(src, edited)
    after = [t for t in vision_text(out) if t not in words]   # newly appearing text (e.g. hallucinated numbers)
    remaining = [t for t in vision_text(out) if t in words]
    reasons = []
    if drift > MAX_DRIFT_P50:
        reasons.append(f"edit changed the whole image (median dE {drift:.1f})")
    if mask_share > MAX_MASK:
        reasons.append(f"too much changed ({mask_share:.1%})")
    if after:
        reasons.append("new text appeared: " + " ".join(after))
    if remaining:
        reasons.append("text not removed: " + " ".join(remaining))
    out.save(OUT / f"{row.sha256}.png")
    return {"sha256": row.sha256, "ok": not reasons, "reason": "; ".join(reasons), "mask_share": mask_share, "drift_p50": drift,
            "ocr_before": words, "ocr_new": after, "ocr_remaining": remaining,
            "modification": "figure annotations removed (qwen_image_2.1-turbo edit, composited into the original)",
            "attribution": row.attribution, "license": row.license_raw}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sha", nargs="*", default=[])
    ap.add_argument("--sample", type=int, default=0)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    R = pd.read_parquet(WORK / "refs/permissive_refs.parquet")
    R = R[R.usable & R.lesion_photo]
    sel = R[R.sha256.isin(a.sha)] if a.sha else R.sample(a.sample, random_state=0)
    api = ex.resolve_api()
    log = json.load(open(WORK / "refs/cleaned.json")) if (WORK / "refs/cleaned.json").exists() else {}
    for r in sel.itertuples():
        t = time.time()
        res = clean(r, api)
        log[r.sha256] = res
        print(r.sha256[:12], "ok" if res["ok"] else "REJECT", res.get("reason", ""), f"mask {res.get('mask_share', 0):.3f}",
              f"drift {res.get('drift_p50', 0):.1f}", res.get("ocr_before"), "->", res.get("ocr_new"), res.get("ocr_remaining"), f"{time.time() - t:.0f}s", flush=True)
        json.dump(log, open(WORK / "refs/cleaned.json", "w"), indent=1)


if __name__ == "__main__":
    main()
