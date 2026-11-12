"""Shared paths and helpers for the dataset build.

All source folders (derma-data/, medical-datasets/) are treated as strictly
read-only: scripts only ever open them for reading. Intermediate artefacts go
to work/, the final dataset to dataset/.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DERMA = ROOT / "derma-data"
MED = ROOT / "medical-datasets"
WORK = ROOT / "work"
DATASET = ROOT / "dataset"
EXTERNAL = ROOT / "external"

WORK.mkdir(exist_ok=True)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".tif", ".tiff"}

# Columns every ingest script emits (one row = one image occurrence + its text).
SAMPLE_COLUMNS = [
    "source",          # short source key, e.g. "jaad", "dermnet", "braun-falco"
    "source_type",     # atlas | journal | textbook
    "src_path",        # absolute path of the original file (read-only)
    "caption",         # original caption / description text (as published)
    "title",           # article / chapter / page title
    "diagnosis",       # diagnosis label given by the source (raw)
    "body_site",       # localisation given by the source (raw)
    "age", "sex",      # as given by the source (raw strings)
    "extra",           # JSON string with any other source-specific fields
    "language",        # ISO 639-1 of the text
    "doi", "url",      # article DOI / page URL
    "image_url",       # original image URL if known
    "year",            # publication year if known
    "article_id",      # key into articles table (journals / textbook chapters)
    "figure_label",    # e.g. "Fig. 2", "Abb. 39.5"
    "license",         # licence string detected or registered for the item
]


def sha256_file(path: str | os.PathLike, bufsize: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(bufsize):
            h.update(chunk)
    return h.hexdigest()


_ws = re.compile(r"\s+")


def clean_text(s) -> str | None:
    if s is None:
        return None
    if not isinstance(s, str):
        s = str(s)
    s = unicodedata.normalize("NFC", s)
    s = _ws.sub(" ", s).strip()
    return s or None


def jdump(d) -> str | None:
    d = {k: v for k, v in (d or {}).items() if v not in (None, "", [], {})}
    return json.dumps(d, ensure_ascii=False) if d else None


def empty_row(**kw) -> dict:
    row = {c: None for c in SAMPLE_COLUMNS}
    row.update(kw)
    return row


LOCAL_BATCHES = ["local_b1", "local_b2", "local_b3"]


def load_local_embeddings(with_zeroshot=False):
    """All local (atlas/textbook/journal) unique images: index frame, embeddings[, zero-shot]."""
    import numpy as np
    import pandas as pd
    P = pd.concat([pd.read_parquet(WORK / f"embed_{b.replace('local_b', 'local_batch')}.parquet")
                   for b in LOCAL_BATCHES], ignore_index=True)
    E = np.concatenate([np.load(WORK / f"emb/{b}.npy") for b in LOCAL_BATCHES])
    if not with_zeroshot:
        return P, E
    Z = pd.concat([pd.read_parquet(WORK / f"zs_{b}.parquet") for b in LOCAL_BATCHES], ignore_index=True)
    return P, E, Z


def device():
    """Best available accelerator: CUDA (donv, 2x RTX 3090) > Apple MPS > CPU.
    Pin a CUDA process to one GPU with CUDA_VISIBLE_DEVICES."""
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
