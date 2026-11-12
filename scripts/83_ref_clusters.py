"""Clusters of the permissive reference pool (CC BY / CC0 lesion photographs without text annotations; 79 / 82):
1) image, 2) caption text, 3) image + text embeddings, each laid out with PaCMAP and clustered with HDBSCAN.

Embeddings: the dermatology CLIP (ViT-B/16 fine-tuned on ours + Derm1M, work/models/clip_union_3k) for both images and
captions, so all three live in one space; joint = normalise(image + text). Captions: Derm1M caption / our figure caption
(first 77 tokens). Cluster names: 85-based naming script 84_ref_cluster_names.py (Splash 27B, captions + sample images).
Writes work/refs/clusters/{emb_image,emb_text,emb_joint}.npy, clusters.parquet (one row per image with x/y/cluster for
each view) and thumbnails (160 px JPEG) for the interactive page.
"""
import importlib
import json
import sys
import types
from pathlib import Path

import subprocess

import numpy as np
import pandas as pd
from PIL import Image

from common import WORK, device

OUT = WORK / "refs/clusters"


def encode_images_safe(be, paths, out, chunk=1000, bs=64):
    """Main-process image loading (no DataLoader workers), per-image error handling, huge images downscaled first,
    resumable chunk files; unreadable images get a zero vector and are listed in failed_images.json."""
    import torch
    Image.MAX_IMAGE_PIXELS = 400_000_000
    failed, parts = [], []
    for c, s in enumerate(range(0, len(paths), chunk)):
        f = out / f"_img_part{c:03d}.npy"
        if f.exists():
            parts.append(np.load(f)); continue
        feats = []
        end = min(s + chunk, len(paths))
        for b in range(s, end, bs):
            xs, ok = [], []
            for p in paths[b:min(b + bs, end)]:   # never cross into the next chunk
                try:
                    im = Image.open(p); im.draft("RGB", (1024, 1024)); im = im.convert("RGB")
                    im.thumbnail((1024, 1024))
                    xs.append(be.tf(im)); ok.append(True)
                except Exception as e:
                    xs.append(torch.zeros(3, 224, 224)); ok.append(False); failed.append([p, repr(e)[:120]])
            with torch.no_grad():
                z = torch.nn.functional.normalize(be.model.encode_image(torch.stack(xs).to(be.dev)).float(), dim=-1).cpu().numpy()
            z[~np.array(ok)] = 0
            feats.append(z)
        z = np.concatenate(feats)
        assert len(z) == end - s, (len(z), end - s)
        np.save(f, z); parts.append(z)
        print("images", min(s + chunk, len(paths)), "/", len(paths), flush=True)
    json.dump(failed, open(out / "failed_images.json", "w"))
    return np.concatenate(parts).astype(np.float32)


def main():
    if "--cluster" in sys.argv:
        return cluster()
    embed()
    # PaCMAP / HDBSCAN (numba) in the same process as a PyTorch MPS model segfaults on macOS: separate process
    subprocess.run([sys.executable, __file__, "--cluster"], check=True)


def embed():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "thumbs").mkdir(exist_ok=True)
    R = pd.read_parquet(WORK / "refs/permissive_refs.parquet")
    R = R[R.usable & R.lesion_photo & ~R.has_text].drop_duplicates("sha256").reset_index(drop=True)
    R["text"] = R.caption.fillna("").str.strip()
    R.loc[R.text == "", "text"] = R.diagnosis.fillna("skin lesion")
    print("references", len(R), flush=True)

    sys.argv = [sys.argv[0]]
    ev = importlib.import_module("60_eval_clip")
    be = ev.load_backend(types.SimpleNamespace(backend="openclip", pretrained=str(WORK / "models/clip_union_3k/model.pt"),
                                               model="ViT-B-16-quickgelu"), device())
    fi, ft = OUT / "emb_image.npy", OUT / "emb_text.npy"
    if not fi.exists():
        np.save(fi, encode_images_safe(be, R.path.tolist(), OUT))
    if not ft.exists():
        np.save(ft, be.encode_texts(R.text.tolist()).astype(np.float32))
    R.to_parquet(OUT / "_refs.parquet", index=False)


def cluster():
    import hdbscan
    import pacmap
    R = pd.read_parquet(OUT / "_refs.parquet")
    Ei, Et = np.load(OUT / "emb_image.npy"), np.load(OUT / "emb_text.npy")
    good = np.linalg.norm(Ei, axis=1) > 0
    R, Ei, Et = R[good].reset_index(drop=True), Ei[good], Et[good]
    Ej = Ei + Et
    Ej /= np.linalg.norm(Ej, axis=1, keepdims=True)
    np.save(OUT / "emb_joint.npy", Ej)

    for name, E in [("image", Ei), ("text", Et), ("joint", Ej)]:
        Y = pacmap.PaCMAP(n_components=2, n_neighbors=15, FP_ratio=8.0, random_state=0).fit_transform(E, init="pca")
        lab = hdbscan.HDBSCAN(min_cluster_size=40, min_samples=10).fit_predict(Y)
        if lab.max() + 1 < 30:   # dense views (images) collapse into a few large eom clusters: use leaf clusters instead
            lab = hdbscan.HDBSCAN(min_cluster_size=40, min_samples=10, cluster_selection_method="leaf").fit_predict(Y)
        R[f"x_{name}"], R[f"y_{name}"], R[f"c_{name}"] = Y[:, 0].round(3), Y[:, 1].round(3), lab
        print(name, "clusters", lab.max() + 1, "noise", round(float((lab < 0).mean()), 3), flush=True)

    for i, r in enumerate(R.itertuples()):
        t = OUT / f"thumbs/{r.sha256[:16]}.jpg"
        if not t.exists():
            try:
                im = Image.open(r.path); im.draft("RGB", (320, 320)); im = im.convert("RGB")
                im.thumbnail((160, 160)); im.save(t, quality=80)
            except Exception:
                pass
    keep = ["sha256", "path", "origin", "license_raw", "attribution", "diagnosis", "body_site", "caption", "fst_pred", "image_type"] + \
        [c for c in R.columns if c[:2] in ("x_", "y_", "c_")]
    R[keep].to_parquet(OUT / "clusters.parquet", index=False)
    json.dump({n: int(R[f"c_{n}"].max() + 1) for n in ["image", "text", "joint"]}, open(OUT / "summary.json", "w"))


if __name__ == "__main__":
    main()
