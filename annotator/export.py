#!/usr/bin/env python
"""Read expert assessments (joined with item metadata) for analysis. Read-only.

    python annotator/export.py                         # all assessments, readable table
    python annotator/export.py --since 2026-10-14T00:00 --format jsonl
    python annotator/export.py --latest --set lifecycle_pilot   # only the current (latest) row per item
    python annotator/export.py --summary               # counts by verdict / diagnosis / quick fields
    python annotator/export.py --unassessed --set lifecycle_pilot  # items still without assessment
    python annotator/export.py --compare-summary [--set models_48] [--since T]  # model comparison stats
JSONL/JSON output also contains comparison records (record_type "comparison"); assessment records carry
record_type "assessment" and, if saved from the compare view, compare_group / compare_session.

Gated-tier items (e.g. nudity_review) are included by default; --open-only restricts output to the open tier.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import textwrap
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import annodb  # noqa: E402

QUICK = ["verdict", "realism", "dx_recognisable", "skin_type_correct", "site_correct", "stage_correct"]


def open_ro(path):
    con = sqlite3.connect(f"file:{Path(path).resolve()}?mode=ro", uri=True, timeout=30)
    con.row_factory = sqlite3.Row
    return con


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default=str(annodb.DEFAULT_DB))
    ap.add_argument("--since", help="UTC ISO timestamp prefix, e.g. 2026-10-14 or 2026-10-14T00:00")
    ap.add_argument("--set", dest="set_name")
    ap.add_argument("--item")
    ap.add_argument("--latest", action="store_true", help="only the latest assessment per item")
    ap.add_argument("--format", choices=["table", "jsonl", "json"], default="table")
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--unassessed", action="store_true", help="list items without any assessment")
    ap.add_argument("--include-gated", action="store_true", help="no-op (gated tiers are included by default)")
    ap.add_argument("--open-only", action="store_true", help="exclude gated-tier items (tier != open)")
    ap.add_argument("--compare-summary", action="store_true",
                    help="per comparison set and arm: win rate, realism, quick-field yes-rates, verdicts, position bias")
    args = ap.parse_args()
    args.include_gated = not args.open_only
    con = open_ro(args.db)
    if args.compare_summary:
        return compare_summary(con, args)

    tier = "1=1" if args.include_gated else f"i.tier='{annodb.OPEN_TIER}'"
    if args.unassessed:
        where, params = [tier, "i.active=1", "NOT EXISTS (SELECT 1 FROM assessments a WHERE a.item_id=i.item_id)"], []
        if args.set_name:
            where.append("i.source_set=?")
            params.append(args.set_name)
        for r in con.execute(f"SELECT item_id, title FROM items i WHERE {' AND '.join(where)} ORDER BY source_set, ord", params):
            print(f"{r['item_id']}\t{r['title']}")
        return

    src = "latest_assessments" if args.latest else "assessments"
    where, params = [tier], []
    if args.since:
        where.append("a.created_at >= ?")
        params.append(args.since)
    if args.set_name:
        where.append("i.source_set = ?")
        params.append(args.set_name)
    if args.item:
        where.append("a.item_id = ?")
        params.append(args.item)
    rows = con.execute(
        f"""SELECT a.*, i.source_set, i.source_id, i.tier, i.title, i.diagnosis, i.prompt, i.caption, i.metadata,
                   i.image_path
            FROM {src} a JOIN items i ON i.item_id = a.item_id
            WHERE {' AND '.join(where)} ORDER BY a.id""", params).fetchall()
    recs = []
    for r in rows:
        d = dict(r)
        d["metadata"] = json.loads(d["metadata"] or "{}")
        d.pop("client", None)
        recs.append(d)

    if args.summary:
        n_items = con.execute(f"SELECT COUNT(*) FROM items i WHERE i.active=1 AND {tier}"
                              + (" AND i.source_set=?" if args.set_name else ""),
                              [args.set_name] if args.set_name else []).fetchone()[0]
        latest = {}
        for d in recs:
            latest[d["item_id"]] = d  # ordered by id -> last wins
        print(f"assessments: {len(recs)}  items assessed: {len(latest)} / {n_items} active items"
              + (f"  (since {args.since})" if args.since else ""))
        print("(counts below use the latest assessment per item)")
        for f in QUICK:
            c = Counter(str(d[f]) if d[f] is not None else "-" for d in latest.values())
            print(f"  {f:18s} " + ", ".join(f"{k}: {v}" for k, v in c.most_common()))
        by_dx = defaultdict(Counter)
        for d in latest.values():
            by_dx[d["diagnosis"] or "(none)"][d["verdict"] or "-"] += 1
        print("\nverdict by diagnosis:")
        for dx, c in sorted(by_dx.items(), key=lambda kv: -sum(kv[1].values())):
            print(f"  {dx[:45]:45s} n={sum(c.values()):3d}  " + ", ".join(f"{k}: {v}" for k, v in c.most_common()))
        print(f"\nwith suggested caption: {sum(1 for d in latest.values() if d['suggested_caption'])}")
        return

    comps = [] if args.item else load_comparisons(con, args)
    if args.format in ("jsonl", "json"):
        out = [dict(d, record_type="assessment") for d in recs] + [dict(c, record_type="comparison") for c in comps]
        if args.format == "jsonl":
            for d in out:
                print(json.dumps(d, ensure_ascii=False))
        else:
            print(json.dumps(out, ensure_ascii=False, indent=1))
    else:
        for c in comps:
            best = c["best_label"] or ("none" if c["no_best"] else "-")
            print(f"comparison #{c['id']}  {c['created_at']}  {c['annotator']}  {c['group_id']}  blind={c['blind']}")
            print(f"  shown: {' | '.join(c['shown_labels'])}   best: {best}"
                  + (f"   ranking: {' > '.join(c['ranking_labels'])}" if c["ranking_labels"] else ""))
            if c["comment"]:
                print(textwrap.indent(textwrap.fill(c["comment"], 110), "  comment: ").replace("\n  comment: ", "\n           "))
            if c["combination_note"]:
                print(f"  combination: {c['combination_note']}")
            print()
        for d in recs:
            q = " · ".join(f"{f}={d[f]}" for f in QUICK if d[f] is not None)
            m = d["metadata"]
            print(f"#{d['id']}  {d['created_at']}  {d['annotator']}  {d['item_id']}")
            print(f"  item: {d['title']} | FST {m.get('fitzpatrick', m.get('fst'))} | {m.get('age')} {m.get('sex')} | {m.get('site')}")
            if q:
                print(f"  {q}")
            if d["comment"]:
                print(textwrap.indent(textwrap.fill(d["comment"], 110), "  comment: ").replace("\n  comment: ", "\n           "))
            if d["suggested_caption"]:
                print(f"  suggested caption: {d['suggested_caption']}")
            print(f"  caption: {d['caption']}")
            print()
        print(f"{len(recs)} assessment(s), {len(comps)} comparison(s)", file=sys.stderr)


def gated_items(con):
    return {r[0] for r in con.execute("SELECT item_id FROM items WHERE tier != ?", (annodb.OPEN_TIER,))}


def load_comparisons(con, args):
    """Comparisons joined with their group; item ids mapped to arm labels. Gated groups are skipped with --open-only."""
    where, params = ["1=1"], []
    if args.since:
        where.append("c.created_at >= ?")
        params.append(args.since)
    if args.set_name:
        where.append("c.comparison_set = ?")
        params.append(args.set_name)
    hidden = set() if args.include_gated else gated_items(con)
    out = []
    for r in con.execute(f"""SELECT c.*, g.arms, g.plan FROM comparisons c JOIN compare_groups g USING (group_id)
                             WHERE {' AND '.join(where)} ORDER BY c.id""", params):
        d = dict(r)
        d.pop("client", None)
        for k in ("shown_order", "shown_labels", "ranking", "assessment_ids", "arms", "plan"):
            d[k] = json.loads(d[k]) if d[k] else None
        if any(x in hidden for x in d["shown_order"]):
            continue
        lab = dict(zip(d["shown_order"], d["shown_labels"]))
        d["best_label"] = lab.get(d["best_item_id"])
        d["best_position"] = d["shown_order"].index(d["best_item_id"]) if d["best_item_id"] in lab else None
        d["ranking_labels"] = [lab.get(x, x) for x in (d["ranking"] or [])]
        out.append(d)
    return out


def compare_summary(con, args):
    comps = load_comparisons(con, args)
    latest = {}
    for c in comps:  # latest comparison per group = current judgement
        latest[c["group_id"]] = c
    hidden = set() if args.include_gated else gated_items(con)
    groups = {r["group_id"]: dict(r) for r in con.execute("SELECT * FROM compare_groups")}
    sets = sorted({g["comparison_set"] for g in groups.values()} if not args.set_name else {args.set_name})
    for cs in sets:
        gs = {k: g for k, g in groups.items() if g["comparison_set"] == cs}
        lc = [c for c in latest.values() if c["comparison_set"] == cs]
        print(f"== {cs}: {len(lc)} / {len(gs)} groups compared (latest comparison per group"
              + (f", since {args.since}" if args.since else "") + ")")
        if not lc:
            print()
            continue
        # per-arm label: shown, wins, rank
        arm = defaultdict(lambda: {"shown": 0, "wins": 0, "ranks": [], "realism": [], "verdict": Counter(),
                                   "yes": defaultdict(lambda: [0, 0]), "n_assess": 0})
        for c in lc:
            for iid, lab in zip(c["shown_order"], c["shown_labels"]):
                if iid in hidden:
                    continue
                arm[lab]["shown"] += 1
                if c["best_item_id"] == iid:
                    arm[lab]["wins"] += 1
                if c["ranking"] and iid in c["ranking"]:
                    arm[lab]["ranks"].append(c["ranking"].index(iid) + 1)
        # per-column assessments saved with comparisons: latest per (group, item)
        q = ["a.compare_group IS NOT NULL", "g.comparison_set = ?"]
        p = [cs]
        if args.since:
            q.append("a.created_at >= ?")
            p.append(args.since)
        la = {}
        for a in con.execute(f"""SELECT a.*, g.arms FROM assessments a JOIN compare_groups g ON g.group_id = a.compare_group
                                 WHERE {' AND '.join(q)} ORDER BY a.id""", p):
            if a["item_id"] not in hidden:
                la[(a["compare_group"], a["item_id"])] = a
        for (gid, iid), a in la.items():
            lab = {x["item_id"]: x["label"] for x in json.loads(a["arms"])}.get(iid, "?")
            A = arm[lab]
            A["n_assess"] += 1
            if a["realism"] is not None:
                A["realism"].append(a["realism"])
            if a["verdict"]:
                A["verdict"][a["verdict"]] += 1
            for f in annodb.TRISTATE_FIELDS:
                if a[f]:
                    A["yes"][f][1] += 1
                    A["yes"][f][0] += a[f] == "yes"
        none_n = sum(1 for c in lc if c["no_best"])
        print(f"  {'arm':26s} {'shown':>5s} {'best':>5s} {'win%':>6s} {'mean rank':>9s} {'n_ass':>5s} {'realism':>7s}  "
              f"{'dx%':>5s} {'skin%':>5s} {'site%':>5s} {'stage%':>6s}  verdicts")
        def pct(v):
            return f"{100 * v[0] / v[1]:.0f}" if v[1] else "-"
        for lab, A in sorted(arm.items(), key=lambda kv: -(kv[1]["wins"] / kv[1]["shown"] if kv[1]["shown"] else 0)):
            wr = f"{100 * A['wins'] / A['shown']:.0f}" if A["shown"] else "-"
            mr = f"{sum(A['ranks']) / len(A['ranks']):.2f}" if A["ranks"] else "-"
            rl = f"{sum(A['realism']) / len(A['realism']):.2f}" if A["realism"] else "-"
            y = A["yes"]
            print(f"  {lab[:26]:26s} {A['shown']:5d} {A['wins']:5d} {wr:>6s} {mr:>9s} {A['n_assess']:5d} {rl:>7s}  "
                  f"{pct(y['dx_recognisable']):>5s} {pct(y['skin_type_correct']):>5s} {pct(y['site_correct']):>5s} "
                  f"{pct(y['stage_correct']):>6s}  " + ", ".join(f"{k}: {v}" for k, v in A["verdict"].most_common()))
        print(f"  'none' chosen as best: {none_n}; no best given: {sum(1 for c in lc if not c['best_item_id'] and not c['no_best'])}")
        # position bias: win rate by shown position (blind comparisons separately)
        for blind in (1, 0):
            sub = [c for c in lc if c["blind"] == blind]
            if not sub:
                continue
            shown_at, won_at = Counter(), Counter()
            for c in sub:
                for i, iid in enumerate(c["shown_order"]):
                    shown_at[i] += 1
                if c["best_position"] is not None:
                    won_at[c["best_position"]] += 1
            print(f"  position bias ({'blind' if blind else 'not blind'}, n={len(sub)}): " + ", ".join(
                f"pos {i + 1}: {won_at[i]}/{shown_at[i]} = {100 * won_at[i] / shown_at[i]:.0f}%" for i in sorted(shown_at))
                + f"  (chance ~{100 / (sum(shown_at.values()) / len(sub)):.0f}%)")
        print()


if __name__ == "__main__":
    main()
