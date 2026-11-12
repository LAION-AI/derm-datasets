#!/usr/bin/env python
"""Import (upsert) image items into the annotator DB. Never touches assessments.

    python annotator/import_items.py dataset/synthetic/examples/manifest.json --set examples_v1
    python annotator/import_items.py pilot/manifest.parquet --set lifecycle_pilot

Supported manifests: .json (list of objects, as in dataset/synthetic/examples/),
.jsonl, .parquet, .csv. Columns/keys recognised (others are kept as metadata):
  id:      synth_id | id | item_id            (fallback: image file stem)
  image:   path | file | image | image_path   (relative to the manifest dir, or absolute)
  prompt, caption, diagnosis | condition, variant, stage, fst_target | fst, age, sex, site, ...
Sibling checks.json / review.json / review_full.json (keyed by id) are attached as metadata;
items the review files mark as deleted are skipped. Missing image files are skipped.
Rows whose tier is not 'open' (or that live under clinician_gated/) are refused unless --allow-gated.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import annodb  # noqa: E402

PROJECT = Path(__file__).resolve().parent.parent
IMG_EXT = {".png", ".jpg", ".jpeg", ".webp"}
ID_KEYS = ("synth_id", "id", "item_id")
PATH_KEYS = ("path", "file", "image", "image_path")
GATED_MARKERS = ("clinician_gated",)


def clean(v):
    """Make a value JSON-safe (numpy scalars, NaN, arrays)."""
    if v is None:
        return None
    if hasattr(v, "item") and not isinstance(v, (list, dict, str)):
        try:
            v = v.item()
        except (ValueError, AttributeError):
            pass
    if hasattr(v, "tolist"):
        v = v.tolist()
    if isinstance(v, float) and math.isnan(v):
        return None
    if isinstance(v, dict):
        return {str(k): clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [clean(x) for x in v]
    if isinstance(v, (str, int, float, bool)):
        return v
    try:  # pandas NA / NaT
        import pandas as pd
        if pd.isna(v):
            return None
    except Exception:
        pass
    return str(v)


def load_rows(manifest: Path) -> list[dict]:
    suf = manifest.suffix.lower()
    if suf == ".json":
        data = json.loads(manifest.read_text())
        if isinstance(data, dict):  # {id: {...}} or {"items": [...]}
            data = data.get("items") or [dict(v, id=v.get("id", k)) for k, v in data.items()]
        return [clean(r) for r in data]
    if suf == ".jsonl":
        return [clean(json.loads(l)) for l in manifest.read_text().splitlines() if l.strip()]
    if suf == ".csv":
        with open(manifest, newline="", encoding="utf-8") as fh:
            return [{k: (v if v != "" else None) for k, v in r.items()} for r in csv.DictReader(fh)]
    if suf == ".parquet":
        import pandas as pd
        df = pd.read_parquet(manifest)
        return [{k: clean(v) for k, v in r.items()} for r in df.to_dict(orient="records")]
    raise SystemExit(f"unsupported manifest type: {manifest}")


def load_sidecar(path: Path | None):
    if path and path.exists():
        return json.loads(path.read_text())
    return {}


def is_deleted(entry) -> bool:
    if isinstance(entry, str):
        return entry.strip().lower().startswith("delet")
    if not isinstance(entry, dict):
        return False
    if entry.get("deleted") is True:
        return True
    for k in ("visual", "status", "review_status", "action", "decision"):
        v = entry.get(k)
        if isinstance(v, str) and v.strip().lower().startswith("delet"):
            return True
    return False


def first(row, keys):
    for k in keys:
        if row.get(k) not in (None, ""):
            return row[k], k
    return None, None


def resolve_image(raw: str, manifest_dir: Path, image_root: Path | None) -> Path | None:
    p = Path(str(raw)).expanduser()
    cands = [p] if p.is_absolute() else [c / p for c in (image_root, manifest_dir, PROJECT) if c]
    for c in cands:
        if c.is_file():
            return c.resolve()
    return None


def fst_label(v):
    roman = {1: "I", 2: "II", 3: "III", 4: "IV", 5: "V", 6: "VI"}
    try:
        return roman[int(v)]
    except (TypeError, ValueError, KeyError):
        return v


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("manifest", type=Path)
    ap.add_argument("--set", required=True, dest="set_name", help="source set name, e.g. lifecycle_pilot")
    ap.add_argument("--db", default=str(annodb.DEFAULT_DB))
    ap.add_argument("--image-root", type=Path, help="base dir for relative image paths (default: manifest dir)")
    ap.add_argument("--checks", type=Path, help="checks JSON keyed by id (default: sibling checks.json)")
    ap.add_argument("--review", type=Path, action="append",
                    help="review JSON keyed by id (default: sibling review.json and review_full.json)")
    ap.add_argument("--tier", help="override tier for all rows (default: row 'tier' column, else 'open')")
    ap.add_argument("--allow-gated", action="store_true", help="permit importing non-open (gated) tiers")
    ap.add_argument("--prefer-unredacted", action="store_true",
                    help="gated rows: use 'file' instead of 'file_redacted' (default: redacted)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    manifest = args.manifest.resolve()
    mdir = manifest.parent
    rows = load_rows(manifest)
    checks = load_sidecar(args.checks or mdir / "checks.json")
    review_files = args.review or [mdir / "review.json", mdir / "review_full.json"]
    reviews, deleted = {}, {}
    for rf in review_files:
        r = load_sidecar(rf)
        if isinstance(r, dict) and ("kept" in r or "deleted" in r):  # review_full.json layout
            deleted.update(r.get("deleted") or {})
            reviews.update(r.get("kept") or {})
        elif isinstance(r, dict):
            for k, v in r.items():
                (deleted if is_deleted(v) else reviews)[k] = v

    if not args.dry_run:
        annodb.backup(args.db, tag="pre-import-" + annodb.utcnow().replace(":", ""))
    con = annodb.connect(args.db)
    now = annodb.utcnow()
    stats = {"rows": len(rows), "inserted": 0, "updated": 0, "skipped_missing": 0,
             "skipped_deleted": 0, "refused_gated": 0, "deactivated": 0}
    seen = set()
    con.execute("BEGIN")
    for ordinal, row in enumerate(rows):
        sid, _ = first(row, ID_KEYS)
        tier = args.tier or row.get("tier") or annodb.OPEN_TIER
        raw_path, path_key = first(row, PATH_KEYS)
        in_gated_dir = any(m in str(manifest) or m in str(raw_path or "") for m in GATED_MARKERS)
        if in_gated_dir and tier == annodb.OPEN_TIER:
            tier = "clinician_gated"
        gated = tier != annodb.OPEN_TIER
        if gated and not args.allow_gated:
            stats["refused_gated"] += 1
            continue
        if gated and row.get("file_redacted") and not args.prefer_unredacted:
            raw_path, path_key = row["file_redacted"], "file_redacted"
        if sid is None and raw_path:
            sid = Path(str(raw_path)).stem
        if sid is None:
            stats["skipped_missing"] += 1
            continue
        sid = str(sid)
        item_id = f"{args.set_name}:{sid}"
        seen.add(item_id)
        if (sid in deleted or is_deleted(row.get("review_status")) or is_deleted(row.get("status"))
                or row.get("deleted") is True or is_deleted(reviews.get(sid))):
            stats["skipped_deleted"] += 1
            stats["deactivated"] += con.execute(
                "UPDATE items SET active=0, updated_at=? WHERE item_id=? AND active=1", (now, item_id)).rowcount
            continue
        img = resolve_image(raw_path, mdir, args.image_root) if raw_path else None
        if img is None or img.suffix.lower() not in IMG_EXT:
            stats["skipped_missing"] += 1
            stats["deactivated"] += con.execute(
                "UPDATE items SET active=0, updated_at=? WHERE item_id=? AND active=1", (now, item_id)).rowcount
            continue

        diagnosis = row.get("diagnosis") or row.get("condition")
        meta = {k: v for k, v in row.items() if k not in ("prompt", "caption") and v is not None}
        meta.setdefault("diagnosis", diagnosis)
        if "fst" not in meta and row.get("fst_target") is not None:
            meta["fst"] = row["fst_target"]
        if meta.get("fst") is not None:
            meta["fitzpatrick"] = fst_label(meta["fst"])
        if row.get("model") and "generator" not in meta:
            meta["generator"] = row["model"]
        meta["image_key"] = path_key
        if sid in checks:
            meta["auto_checks"] = clean(checks[sid])
        if sid in reviews:
            meta["prelim_review"] = clean(reviews[sid])
        base = row.get("base")
        if base and str(base) != sid:
            meta["edit_base_item"] = f"{args.set_name}:{base}"
        title_parts = [diagnosis, row.get("variant") if row.get("variant") != diagnosis else None, row.get("stage")]
        title = " · ".join(str(p) for p in title_parts if p) or sid

        exists = con.execute("SELECT 1 FROM items WHERE item_id=?", (item_id,)).fetchone()
        con.execute(
            """INSERT INTO items (item_id, source_id, source_set, tier, image_path, title, diagnosis, prompt,
                                  caption, metadata, manifest, ord, active, added_at, updated_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,1,?,?)
               ON CONFLICT(item_id) DO UPDATE SET
                 source_id=excluded.source_id, tier=excluded.tier, image_path=excluded.image_path,
                 title=excluded.title, diagnosis=excluded.diagnosis, prompt=excluded.prompt,
                 caption=excluded.caption, metadata=excluded.metadata, manifest=excluded.manifest,
                 ord=excluded.ord, active=1, updated_at=excluded.updated_at""",
            (item_id, sid, args.set_name, tier, str(img), title, diagnosis, row.get("prompt"),
             row.get("caption"), json.dumps(meta, ensure_ascii=False), str(manifest), ordinal, now, now))
        stats["updated" if exists else "inserted"] += 1
    if args.dry_run:
        con.execute("ROLLBACK")
    else:
        con.execute("COMMIT")
    total = con.execute("SELECT COUNT(*) FROM items WHERE source_set=? AND active=1", (args.set_name,)).fetchone()[0]
    print(json.dumps({**stats, "active_items_in_set": total, "dry_run": args.dry_run, "db": args.db}, indent=1))
    if stats["refused_gated"]:
        print(f"note: {stats['refused_gated']} gated rows refused (use --allow-gated to import them)", file=sys.stderr)


if __name__ == "__main__":
    main()
