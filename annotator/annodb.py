"""Shared SQLite schema and helpers for the dermatology annotator.

Assessments are append-only: SQLite triggers abort every UPDATE and DELETE on
the `assessments` table, and every assessment is also appended (fsync'd) to a
JSONL file as a second copy.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import sqlite3
import threading
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_DB = HERE / "annotations.sqlite"
OPEN_TIER = "open"  # the only tier served without the gated opt-in

VERDICTS = ["none", "caption", "image", "both", "discard"]
TRISTATE = ["yes", "no", "unsure"]
TRISTATE_FIELDS = ["dx_recognisable", "skin_type_correct", "site_correct", "stage_correct"]

SCHEMA = f"""
CREATE TABLE IF NOT EXISTS items (
    item_id      TEXT PRIMARY KEY,          -- '<set>:<source id>'
    source_id    TEXT NOT NULL,             -- id in the source manifest (synth_id / id)
    source_set   TEXT NOT NULL,
    tier         TEXT NOT NULL,             -- 'open' or a gated tier (e.g. 'clinician_gated')
    image_path   TEXT NOT NULL,             -- absolute path, served only via /img?id=<item_id>
    title        TEXT,
    diagnosis    TEXT,
    prompt       TEXT,                      -- full generation prompt
    caption      TEXT,                      -- training caption
    metadata     TEXT NOT NULL DEFAULT '{{}}', -- JSON: every other manifest field + auto checks
    manifest     TEXT,                      -- manifest file it was imported from
    ord          INTEGER NOT NULL DEFAULT 0,-- order within the manifest
    active       INTEGER NOT NULL DEFAULT 1,-- 0 = image missing / marked deleted on a later import
    added_at     TEXT NOT NULL,
    updated_at   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS items_set ON items(source_set, ord);
CREATE INDEX IF NOT EXISTS items_dx ON items(diagnosis);

CREATE TABLE IF NOT EXISTS assessments (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    item_id           TEXT NOT NULL REFERENCES items(item_id),
    created_at        TEXT NOT NULL,        -- UTC ISO 8601
    annotator         TEXT NOT NULL DEFAULT 'dermatologist',
    comment           TEXT NOT NULL DEFAULT '',
    verdict           TEXT CHECK (verdict IS NULL OR verdict IN ({",".join(repr(v) for v in VERDICTS)})),
    realism           INTEGER CHECK (realism IS NULL OR realism BETWEEN 1 AND 5),
    dx_recognisable   TEXT CHECK (dx_recognisable   IS NULL OR dx_recognisable   IN ('yes','no','unsure')),
    skin_type_correct TEXT CHECK (skin_type_correct IS NULL OR skin_type_correct IN ('yes','no','unsure')),
    site_correct      TEXT CHECK (site_correct      IS NULL OR site_correct      IN ('yes','no','unsure')),
    stage_correct     TEXT CHECK (stage_correct     IS NULL OR stage_correct     IN ('yes','no','unsure')),
    suggested_caption TEXT,
    item_hash         TEXT,                 -- sha256(image_path|prompt|caption)[:16] at assessment time
    client            TEXT                  -- remote address / user agent
);
CREATE INDEX IF NOT EXISTS assessments_item ON assessments(item_id, id);
CREATE INDEX IF NOT EXISTS assessments_time ON assessments(created_at);

CREATE TRIGGER IF NOT EXISTS assessments_no_update BEFORE UPDATE ON assessments
BEGIN SELECT RAISE(ABORT, 'assessments are append-only: UPDATE is not allowed'); END;
CREATE TRIGGER IF NOT EXISTS assessments_no_delete BEFORE DELETE ON assessments
BEGIN SELECT RAISE(ABORT, 'assessments are append-only: DELETE is not allowed'); END;
CREATE TRIGGER IF NOT EXISTS items_no_delete_assessed BEFORE DELETE ON items
WHEN EXISTS (SELECT 1 FROM assessments a WHERE a.item_id = OLD.item_id)
BEGIN SELECT RAISE(ABORT, 'item has assessments and cannot be deleted'); END;

-- latest assessment per item = current view
CREATE VIEW IF NOT EXISTS latest_assessments AS
SELECT a.* FROM assessments a
JOIN (SELECT item_id, MAX(id) AS id FROM assessments GROUP BY item_id) m ON a.id = m.id;

-- side-by-side comparison groups: 2-4 outputs (arms) for the same plan row
CREATE TABLE IF NOT EXISTS compare_groups (
    group_id       TEXT PRIMARY KEY,        -- '<comparison_set>:<group_key value>'
    comparison_set TEXT NOT NULL,
    group_key      TEXT NOT NULL,           -- value of the group-key column (e.g. synth_id / ab_pair)
    arms           TEXT NOT NULL,           -- JSON list of objects with label and item_id, canonical order
    plan           TEXT NOT NULL DEFAULT '{{}}', -- JSON shared plan metadata (diagnosis, stage, site, fst, caption, ...)
    diagnosis      TEXT,
    fst            INTEGER,
    ord            INTEGER NOT NULL DEFAULT 0,
    added_at       TEXT NOT NULL,
    updated_at     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS compare_groups_set ON compare_groups(comparison_set, ord);

-- group-level judgements; append-only like assessments
CREATE TABLE IF NOT EXISTS comparisons (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at       TEXT NOT NULL,         -- UTC ISO 8601
    annotator        TEXT NOT NULL DEFAULT 'dermatologist',
    group_id         TEXT NOT NULL REFERENCES compare_groups(group_id),
    comparison_set   TEXT NOT NULL,
    session_id       TEXT NOT NULL,
    shown_order      TEXT NOT NULL,         -- JSON item ids, left to right as displayed
    shown_labels     TEXT NOT NULL,         -- JSON arm labels in the same order
    blind            INTEGER NOT NULL,      -- 1 = model labels hidden while judging
    best_item_id     TEXT,                  -- NULL if not answered or 'none'
    no_best          INTEGER NOT NULL DEFAULT 0, -- 1 = explicitly "none is acceptable / no best"
    ranking          TEXT,                  -- JSON item ids best -> worst (may be partial)
    comment          TEXT NOT NULL DEFAULT '',
    combination_note TEXT,
    assessment_ids   TEXT NOT NULL DEFAULT '[]', -- JSON ids of the per-column assessment rows saved with it
    arms_sig         TEXT,
    client           TEXT
);
CREATE INDEX IF NOT EXISTS comparisons_group ON comparisons(group_id, id);
CREATE INDEX IF NOT EXISTS comparisons_time ON comparisons(created_at);
CREATE TRIGGER IF NOT EXISTS comparisons_no_update BEFORE UPDATE ON comparisons
BEGIN SELECT RAISE(ABORT, 'comparisons are append-only: UPDATE is not allowed'); END;
CREATE TRIGGER IF NOT EXISTS comparisons_no_delete BEFORE DELETE ON comparisons
BEGIN SELECT RAISE(ABORT, 'comparisons are append-only: DELETE is not allowed'); END;
CREATE TRIGGER IF NOT EXISTS compare_groups_no_delete_used BEFORE DELETE ON compare_groups
WHEN EXISTS (SELECT 1 FROM comparisons c WHERE c.group_id = OLD.group_id)
BEGIN SELECT RAISE(ABORT, 'group has comparisons and cannot be deleted'); END;
"""

# columns added after the first release (ALTER TABLE ADD COLUMN does not fire UPDATE triggers)
MIGRATIONS = [
    ("assessments", "compare_group", "TEXT"),
    ("assessments", "compare_session", "TEXT"),
]


def utcnow() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def connect(db_path: str | os.PathLike = DEFAULT_DB) -> sqlite3.Connection:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(db_path), timeout=30, isolation_level=None)  # autocommit; explicit BEGIN
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=FULL")
    con.execute("PRAGMA foreign_keys=ON")
    con.execute("PRAGMA busy_timeout=30000")
    con.executescript(SCHEMA)
    for table, col, typ in MIGRATIONS:
        have = {r[1] for r in con.execute(f"PRAGMA table_info({table})")}
        if col not in have:
            con.execute(f"ALTER TABLE {table} ADD COLUMN {col} {typ}")
    con.execute("CREATE INDEX IF NOT EXISTS assessments_group ON assessments(compare_group)")
    return con


def jsonl_path_for(db_path) -> Path:
    db_path = Path(db_path).resolve()
    if db_path == DEFAULT_DB.resolve():
        return HERE / "assessments.jsonl"
    return db_path.with_suffix(".assessments.jsonl")


def comparisons_jsonl_path_for(db_path) -> Path:
    db_path = Path(db_path).resolve()
    if db_path == DEFAULT_DB.resolve():
        return HERE / "comparisons.jsonl"
    return db_path.with_suffix(".comparisons.jsonl")


def backup_dir_for(db_path) -> Path:
    db_path = Path(db_path).resolve()
    if db_path == DEFAULT_DB.resolve():
        return HERE / "backups"
    return db_path.parent / (db_path.stem + "_backups")


def item_hash(row) -> str:
    h = hashlib.sha256(f"{row['image_path']}|{row['prompt'] or ''}|{row['caption'] or ''}".encode())
    return h.hexdigest()[:16]


def arms_sig(arms_list) -> str:
    """Fingerprint of a group's canonical arm list; saves are rejected if it changed since the page loaded."""
    return hashlib.sha256(json.dumps([a["item_id"] for a in arms_list]).encode()).hexdigest()[:12]


def backup(db_path, tag: str | None = None) -> Path | None:
    """Copy the DB with the online backup API. Daily file unless `tag` given; never overwrites."""
    db_path = Path(db_path)
    if not db_path.exists():
        return None
    bdir = backup_dir_for(db_path)
    bdir.mkdir(parents=True, exist_ok=True)
    name = tag or dt.date.today().isoformat()
    target = bdir / f"{db_path.stem}-{name}.sqlite"
    if target.exists():
        if not tag:
            return None  # daily backup already taken
        n = 1
        while (bdir / f"{db_path.stem}-{name}-{n}.sqlite").exists():
            n += 1
        target = bdir / f"{db_path.stem}-{name}-{n}.sqlite"
    tmp = target.with_suffix(".tmp")
    src = sqlite3.connect(str(db_path), timeout=30)
    dst = sqlite3.connect(str(tmp))
    try:
        src.backup(dst)
    finally:
        dst.close()
        src.close()
    os.replace(tmp, target)
    return target


_jsonl_lock = threading.Lock()


def _clean_choice(value, allowed):
    if value in (None, ""):
        return None
    value = str(value).strip().lower()
    if value not in allowed:
        raise ValueError(f"invalid value {value!r}; allowed: {allowed}")
    return value


def build_assessment(item, data: dict, client: str = "", compare_group=None, compare_session=None,
                     allow_empty: bool = False) -> dict | None:
    """Validate one assessment for `item`. Returns the record, or None if empty and allow_empty."""
    rec = {
        "item_id": item["item_id"],
        "created_at": utcnow(),
        "annotator": (str(data.get("annotator") or "").strip() or "dermatologist")[:100],
        "comment": str(data.get("comment") or "").strip(),
        "verdict": _clean_choice(data.get("verdict"), VERDICTS),
        "realism": None,
        "suggested_caption": (str(data.get("suggested_caption") or "").strip() or None),
        "item_hash": item_hash(item),
        "client": client[:300],
        "compare_group": compare_group,
        "compare_session": compare_session,
    }
    if data.get("realism") not in (None, ""):
        r = int(data["realism"])
        if not 1 <= r <= 5:
            raise ValueError("realism must be 1-5")
        rec["realism"] = r
    for f in TRISTATE_FIELDS:
        rec[f] = _clean_choice(data.get(f), TRISTATE)
    filled = [rec["comment"], rec["verdict"], rec["realism"], rec["suggested_caption"]] + [rec[f] for f in TRISTATE_FIELDS]
    if not any(v not in (None, "") for v in filled):
        if allow_empty:
            return None
        raise ValueError("empty assessment: write a comment or set at least one field")
    return rec


def _snapshot(item) -> dict:
    return {"source_set": item["source_set"], "tier": item["tier"], "image_path": item["image_path"],
            "title": item["title"], "diagnosis": item["diagnosis"], "prompt": item["prompt"],
            "caption": item["caption"], "metadata": json.loads(item["metadata"] or "{}")}


def _insert(con, table, rec) -> int:
    cols = list(rec)
    cur = con.execute(f"INSERT INTO {table} ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})",
                      [rec[c] for c in cols])
    return cur.lastrowid


def _append_lines(path: Path, lines: list[dict]):
    if not lines:
        return
    with open(path, "a", encoding="utf-8") as fh:
        for line in lines:
            fh.write(json.dumps(line, ensure_ascii=False) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def add_assessment(con: sqlite3.Connection, db_path, data: dict, client: str = "") -> dict:
    """Validate and append one assessment (DB row + fsync'd JSONL line). Returns the stored record."""
    item = con.execute("SELECT * FROM items WHERE item_id=?", (data.get("item_id"),)).fetchone()
    if item is None:
        raise KeyError("unknown item_id")
    rec = build_assessment(item, data, client)
    with _jsonl_lock:
        con.execute("BEGIN IMMEDIATE")
        try:
            rec = {"id": _insert(con, "assessments", rec), **rec}
            _append_lines(jsonl_path_for(db_path), [dict(rec, item_snapshot=_snapshot(item))])
            con.execute("COMMIT")
        except BaseException:
            con.execute("ROLLBACK")
            raise
    return rec


def add_comparison(con: sqlite3.Connection, db_path, group, session_id: str, blind: bool,
                   shown: list, labels: list, column_data: list, best_item_id, no_best: bool,
                   ranking: list | None, comment: str, combination_note: str, annotator: str,
                   arms_sig: str, client: str = "") -> dict:
    """Append one comparison plus one assessment row per filled column, atomically.

    shown: item rows in display order; column_data: dicts (quick fields + comment) per shown column.
    """
    annotator = (str(annotator or "").strip() or "dermatologist")[:100]
    recs = []
    for item, data in zip(shown, column_data):
        r = build_assessment(item, dict(data or {}, annotator=annotator), client,
                             compare_group=group["group_id"], compare_session=session_id, allow_empty=True)
        if r is not None:
            recs.append((item, r))
    comment = str(comment or "").strip()
    combination_note = str(combination_note or "").strip() or None
    if not (recs or best_item_id or no_best or ranking or comment or combination_note):
        raise ValueError("empty comparison: fill a column, pick a best output, or write a comment")
    comp = {
        "created_at": utcnow(), "annotator": annotator, "group_id": group["group_id"],
        "comparison_set": group["comparison_set"], "session_id": session_id,
        "shown_order": json.dumps([i["item_id"] for i in shown]), "shown_labels": json.dumps(labels),
        "blind": 1 if blind else 0, "best_item_id": best_item_id, "no_best": 1 if no_best else 0,
        "ranking": json.dumps(ranking) if ranking else None, "comment": comment,
        "combination_note": combination_note, "assessment_ids": "[]", "arms_sig": arms_sig,
        "client": client[:300],
    }
    with _jsonl_lock:
        con.execute("BEGIN IMMEDIATE")
        try:
            a_lines, ids = [], []
            for item, r in recs:
                r = {"id": _insert(con, "assessments", r), **r}
                ids.append(r["id"])
                a_lines.append(dict(r, item_snapshot=_snapshot(item)))
            comp["assessment_ids"] = json.dumps(ids)
            comp = {"id": _insert(con, "comparisons", comp), **comp}
            c_line = dict(comp, arms=json.loads(group["arms"]), plan=json.loads(group["plan"] or "{}"))
            _append_lines(jsonl_path_for(db_path), a_lines)
            _append_lines(comparisons_jsonl_path_for(db_path), [c_line])
            con.execute("COMMIT")
        except BaseException:
            con.execute("ROLLBACK")
            raise
    return {"comparison": comp, "assessment_ids": ids}
