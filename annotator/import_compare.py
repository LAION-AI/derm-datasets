#!/usr/bin/env python
"""Import side-by-side comparison groups (2-4 model outputs for the same plan row).

    python annotator/import_compare.py --set models_48 --group-key synth_id \
        --arm klein-4step=work/synthetic/pilot_klein4/manifest_48.parquet:work/synthetic/eval_pilot_klein4_48/per_image.parquet \
        --arm z-image-turbo=work/synthetic/pilot_zimage/manifest.parquet:work/synthetic/eval_pilot_zimage/per_image.parquet

    # one manifest, arms taken from a column (rows with a null group key are ignored):
    python annotator/import_compare.py --set prompt_ab --group-key ab_pair --split-col desc_source \
        --arm batch1=work/synthetic/batch1/manifest.parquet:work/synthetic/batch1/eval/per_image.parquet

A group = rows that share the group-key value across arms; only groups with >= 2 arms whose image exists are kept.
Existing items with the same image path are reused (assessment history carries over); otherwise items are created
in set '<comparison_set>/<label>'. Tier: 'nudity_review' (gated) if the eval row has nudity_flag true or there is no
eval row, else 'open'. Existing items keep their tier (an existing gated item stays gated; tiers are never
lowered); an existing open item is raised to 'nudity_review' only if its eval row has nudity_flag true. HARD RULE: rows with
age_group infant/child/adolescent (or age < 18) that are nudity-flagged or have no eval row are never imported and
never shown in a group. Re-import is idempotent and merges new arms into existing groups.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import annodb  # noqa: E402
from import_items import IMG_EXT, PROJECT, clean, fst_label, load_rows  # noqa: E402

GATED_TIER = "nudity_review"
MINOR_GROUPS = {"infant", "child", "adolescent"}
PLAN_KEYS = ["diagnosis", "variant", "stage", "site", "site_coarse", "fst_target", "age", "age_group", "sex",
             "ancestry", "view", "caption", "lifecycle_key", "concept_id", "seed"]
EVAL_KEYS = ["image_type", "p_clinical", "fst_pred", "fst_ok", "p_exposed", "nudity_flag", "nn_real_cos", "memorised",
             "dx_rank", "dx_ok", "pass"]
MAX_ARMS = 4


def resolve(p: str) -> Path:
    q = Path(p).expanduser()
    if q.is_absolute():
        return q
    return q if q.exists() else PROJECT / q


def parse_arm(spec: str):
    if "=" not in spec:
        raise SystemExit(f"--arm must be LABEL=MANIFEST[:EVAL], got {spec!r}")
    label, rest = spec.split("=", 1)
    manifest, _, ev = rest.partition(":")
    return label.strip(), resolve(manifest), (resolve(ev) if ev else None)


def is_minor(row) -> bool:
    if str(row.get("age_group") or "").strip().lower() in MINOR_GROUPS:
        return True
    try:
        return float(row.get("age")) < 18
    except (TypeError, ValueError):
        return False


def truthy(v) -> bool:
    return v is True or str(v).strip().lower() in ("true", "1", "yes")


def key_str(v):
    if v is None:
        return None
    if isinstance(v, float) and v.is_integer():
        v = int(v)
    s = str(v).strip()
    return None if s in ("", "nan", "None", "NaN", "<NA>") else s


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", required=True, dest="cset", help="comparison set name, e.g. models_48")
    ap.add_argument("--arm", action="append", required=True, help="LABEL=MANIFEST[:EVAL_PARQUET] (repeatable)")
    ap.add_argument("--group-key", default="synth_id")
    ap.add_argument("--split-col", help="take arm labels from this column of a single manifest (e.g. desc_source)")
    ap.add_argument("--db", default=str(annodb.DEFAULT_DB))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if "/" in args.cset or ":" in args.cset:
        raise SystemExit("--set must not contain '/' or ':'")

    arms = [parse_arm(a) for a in args.arm]
    if args.split_col and len(arms) != 1:
        raise SystemExit("--split-col needs exactly one --arm")

    # ---- load rows per arm: list of (label, row, eval_row or None)
    per_arm: list[tuple[str, list]] = []
    for label, manifest, evp in arms:
        rows = load_rows(manifest)
        ev_by_path, ev_by_id = {}, {}
        if evp:
            if not evp.exists():
                raise SystemExit(f"eval file not found: {evp}")
            for e in load_rows(evp):
                if e.get("path"):
                    ev_by_path[str(Path(str(e["path"])).resolve())] = e
                if e.get("synth_id"):
                    ev_by_id.setdefault(str(e["synth_id"]), e)
        out = {}
        for r in rows:
            p = str(Path(str(r.get("path") or r.get("file") or "")).expanduser())
            if p and not Path(p).is_absolute():
                p = str((manifest.parent / p))
            p = str(Path(p).resolve()) if p else ""
            e = ev_by_path.get(p) or (ev_by_id.get(str(r.get("synth_id"))) if not ev_by_path else None)
            lab = label
            if args.split_col:
                lab = key_str(r.get(args.split_col))
                if lab is None:
                    continue
            out.setdefault(lab, []).append((r, p, e))
        per_arm.extend(out.items())

    con = annodb.connect(args.db)
    by_path = {}
    for r in con.execute("""SELECT i.item_id, i.image_path, i.tier, i.active,
                                   (SELECT COUNT(*) FROM assessments a WHERE a.item_id=i.item_id) AS n
                            FROM items i ORDER BY i.added_at, i.item_id"""):
        cur = by_path.get(r["image_path"])
        if cur is None or (r["n"] > cur["n"]):  # prefer the item that already has assessments
            by_path[r["image_path"]] = dict(r)

    if not args.dry_run:
        annodb.backup(args.db, tag="pre-compare-" + annodb.utcnow().replace(":", ""))
    now = annodb.utcnow()
    st = {"arms": {lab: len(rows) for lab, rows in per_arm}, "reused_items": 0, "created_items": 0,
          "tier_raised": 0, "skipped_missing_image": 0, "skipped_minor_rule": 0, "skipped_inactive_item": 0,
          "skipped_no_key": 0, "groups_new": 0, "groups_updated": 0, "groups_too_small": 0}
    groups: dict[str, dict] = {}  # key -> {"arms": {label: item_id}, "plan": row, "ord": n}
    dropped: dict[str, set] = {}   # key -> labels skipped in this import (removed from existing groups too)
    con.execute("BEGIN")
    try:
        for lab, rows in per_arm:
            for r, path, ev in rows:
                key = key_str(r.get(args.group_key))
                if key is None:
                    st["skipped_no_key"] += 1
                    continue
                flagged = ev is None or truthy(ev.get("nudity_flag")) or ev.get("nudity_flag") is None
                if is_minor(r) and flagged:
                    st["skipped_minor_rule"] += 1
                    dropped.setdefault(key, set()).add(lab)
                    continue
                img = Path(path) if path else None
                if img is None or img.suffix.lower() not in IMG_EXT or not img.is_file():
                    st["skipped_missing_image"] += 1
                    dropped.setdefault(key, set()).add(lab)
                    continue
                tier = GATED_TIER if flagged else annodb.OPEN_TIER
                existing = by_path.get(str(img))
                if existing is not None:
                    if not existing["active"]:
                        st["skipped_inactive_item"] += 1
                        dropped.setdefault(key, set()).add(lab)
                        continue
                    item_id = existing["item_id"]
                    # existing items keep their (already vetted) tier; raise it only on an explicit eval flag
                    if ev is not None and truthy(ev.get("nudity_flag")) and existing["tier"] == annodb.OPEN_TIER:
                        con.execute("UPDATE items SET tier=?, updated_at=? WHERE item_id=?", (GATED_TIER, now, item_id))
                        existing["tier"] = GATED_TIER
                        st["tier_raised"] += 1
                    st["reused_items"] += 1
                else:
                    sid = str(r.get("synth_id") or img.stem)
                    source_set = f"{args.cset}/{lab}"
                    item_id = f"{source_set}:{sid}"
                    diagnosis = r.get("diagnosis") or r.get("condition")
                    meta = {k: clean(v) for k, v in r.items() if k not in ("prompt", "caption") and v is not None}
                    if meta.get("fst_target") is not None:
                        meta.setdefault("fst", meta["fst_target"])
                        meta["fitzpatrick"] = fst_label(meta["fst_target"])
                    meta["compare_arm"] = lab
                    if ev is not None:
                        meta["auto_checks"] = {k: clean(ev.get(k)) for k in EVAL_KEYS if k in ev}
                    title = " · ".join(str(x) for x in (diagnosis, r.get("variant"), r.get("stage")) if x) or sid
                    con.execute(
                        """INSERT INTO items (item_id, source_id, source_set, tier, image_path, title, diagnosis, prompt,
                                              caption, metadata, manifest, ord, active, added_at, updated_at)
                           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,1,?,?)
                           ON CONFLICT(item_id) DO UPDATE SET image_path=excluded.image_path,
                             tier=CASE WHEN items.tier='open' THEN excluded.tier ELSE items.tier END,
                             metadata=excluded.metadata, updated_at=excluded.updated_at""",
                        (item_id, sid, source_set, tier, str(img), title, diagnosis, r.get("prompt"), r.get("caption"),
                         json.dumps(meta, ensure_ascii=False), "compare:" + args.cset, len(groups), now, now))
                    by_path[str(img)] = {"item_id": item_id, "image_path": str(img), "tier": tier, "active": 1, "n": 0}
                    st["created_items"] += 1
                g = groups.setdefault(key, {"arms": {}, "plan": r, "ord": len(groups)})
                g["arms"][lab] = item_id

        for key in dropped:  # groups where every arm was skipped still need their stale arms removed
            groups.setdefault(key, {"arms": {}, "plan": None, "ord": len(groups)})
        for key, g in groups.items():
            group_id = f"{args.cset}:{key}"
            old = con.execute("SELECT * FROM compare_groups WHERE group_id=?", (group_id,)).fetchone()
            arms_list = [a for a in (json.loads(old["arms"]) if old else []) if a["label"] not in dropped.get(key, ())]
            if old and g["plan"] is None:
                g["plan"] = json.loads(old["plan"])
            have = {a["label"]: a for a in arms_list}
            for lab, iid in g["arms"].items():
                if lab in have:
                    have[lab]["item_id"] = iid
                else:
                    arms_list.append({"label": lab, "item_id": iid})
            if len(arms_list) < 2:
                st["groups_too_small"] += 1
                if old:  # keep the row (comparisons may reference it) but record the shrunken arm list
                    con.execute("UPDATE compare_groups SET arms=?, updated_at=? WHERE group_id=?",
                                (json.dumps(arms_list), now, group_id))
                continue
            if g["plan"] is None:
                continue
            if len(arms_list) > MAX_ARMS:
                raise SystemExit(f"group {group_id} would have {len(arms_list)} arms (max {MAX_ARMS})")
            r = g["plan"]
            plan = {k: clean(r.get(k)) for k in PLAN_KEYS if r.get(k) is not None}
            if plan.get("fst_target") is not None:
                plan["fitzpatrick"] = fst_label(plan["fst_target"])
            try:
                fst = int(r.get("fst_target"))
            except (TypeError, ValueError):
                fst = None
            con.execute(
                """INSERT INTO compare_groups (group_id, comparison_set, group_key, arms, plan, diagnosis, fst, ord,
                                               added_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(group_id) DO UPDATE SET arms=excluded.arms, plan=excluded.plan,
                     diagnosis=excluded.diagnosis, fst=excluded.fst, updated_at=excluded.updated_at""",
                (group_id, args.cset, key, json.dumps(arms_list), json.dumps(plan, ensure_ascii=False),
                 plan.get("diagnosis"), fst, old["ord"] if old else g["ord"], old["added_at"] if old else now, now))
            st["groups_updated" if old else "groups_new"] += 1
        con.execute("ROLLBACK" if args.dry_run else "COMMIT")
    except BaseException:
        con.execute("ROLLBACK")
        raise
    rows = con.execute("SELECT arms FROM compare_groups WHERE comparison_set=?", (args.cset,)).fetchall()
    sizes = {}
    for r in rows:
        n = len(json.loads(r["arms"]))
        sizes[n] = sizes.get(n, 0) + 1
    st["groups_in_set"] = len(rows)
    st["groups_by_n_arms"] = sizes
    st["dry_run"] = args.dry_run
    print(json.dumps(st, indent=1))


if __name__ == "__main__":
    main()
