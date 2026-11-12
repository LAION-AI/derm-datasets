#!/usr/bin/env python
"""Expert annotation server for synthetic dermatology images (stdlib only).

    python annotator/app.py [--host 0.0.0.0] [--port 8765] [--db annotator/annotations.sqlite]
                            [--require-gate-token]

Images are served only by registered item id (GET /img?id=<item_id>); no file paths are
accepted from clients. Items whose tier is not 'open' are invisible unless the server runs
Gated-tier items (tier != 'open', e.g. nudity_review) are visible to everyone on the LAN by default (single-user
home network; the tier still marks what must be gated at RELEASE). With --require-gate-token they are hidden or
replaced by placeholders unless the request carries the access token cookie (visit /gated?token=<token> once).
"""
from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import mimetypes
import random
import re
import os
import secrets
import shutil
import sqlite3
import sys
import threading
import time
from http import HTTPStatus
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse, quote

sys.path.insert(0, str(Path(__file__).resolve().parent))
import annodb  # noqa: E402

HERE = Path(__file__).resolve().parent
STATIC = HERE / "static"
IMG_EXT = {".png", ".jpg", ".jpeg", ".webp"}
MAX_BODY = 1 << 20
LETTERS = "ABCD"
SESSION_RE = re.compile(r"[A-Za-z0-9_-]{8,64}")
# metadata keys that would reveal the model / arm while blinded
BLIND_HIDE = {"generator", "model", "compare_arm", "path", "file", "image_key", "desc_source", "source_set",
              "edit_base_item", "licence_note", "tier", "size", "steps", "cfg"}

CFG = {"db": str(annodb.DEFAULT_DB), "require_token": False, "gated_token": None}
PREVIEW_MAX = 1280                      # px, longest side of the cached preview served to the main views
PREVIEW_DIR = HERE / "cache" / "previews"
STATIC_FILES = {"common.js": "application/javascript; charset=utf-8"}
_preview_lock = threading.Lock()
_local = threading.local()


def db() -> sqlite3.Connection:
    con = getattr(_local, "con", None)
    if con is None:
        con = _local.con = annodb.connect(CFG["db"])
    return con


def log(msg: str):
    print(f"{annodb.utcnow()} {msg}", file=sys.stderr, flush=True)


class Handler(BaseHTTPRequestHandler):
    server_version = "DermAnnotator/1.0"
    protocol_version = "HTTP/1.1"

    def finish(self):
        try:
            super().finish()
        finally:
            con = getattr(_local, "con", None)
            if con is not None:
                con.close()
                _local.con = None

    # ---------- helpers ----------
    def log_message(self, fmt, *args):
        log(f"{self.client_address[0]} {fmt % args}")

    def gated_ok(self) -> bool:
        if not CFG["require_token"]:
            return True  # default: gated tiers visible without a token (tier still governs release)
        if not CFG["gated_token"]:
            return False
        c = SimpleCookie(self.headers.get("Cookie", ""))
        tok = c["gt"].value if "gt" in c else ""
        return bool(tok) and hmac.compare_digest(tok, CFG["gated_token"])

    def tier_clause(self, alias="i") -> str:
        return "1=1" if self.gated_ok() else f"{alias}.tier = '{annodb.OPEN_TIER}'"

    def send_common(self, status, ctype, length, extra=None):
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(length))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("X-Frame-Options", "DENY")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()

    def send_json(self, obj, status=200):
        body = json.dumps(obj, ensure_ascii=False).encode()
        self.send_common(status, "application/json; charset=utf-8", len(body), {"Cache-Control": "no-store"})
        self.wfile.write(body)

    def error(self, status, msg):
        self.send_json({"error": msg}, status)

    def q(self):
        return {k: v[-1] for k, v in parse_qs(urlparse(self.path).query).items()}

    # ---------- routing ----------
    def do_GET(self):
        path = urlparse(self.path).path
        try:
            route = {
                "/": self.page, "/index.html": self.page, "/healthz": self.health,
                "/api/config": self.api_config, "/api/items": self.api_items, "/api/item": self.api_item,
                "/api/assessments": self.api_assessments, "/img": self.image, "/gated": self.gated_login,
                "/compare": self.compare_page, "/api/compare/config": self.api_compare_config,
                "/api/compare/groups": self.api_compare_groups, "/api/compare/group": self.api_compare_group,
                "/cimg": self.compare_image, "/api/comparisons": self.api_comparisons,
            }.get(path)
            if route is None and path.startswith("/static/") and path[8:] in STATIC_FILES:
                route = lambda: self.static_file(path[8:])  # noqa: E731
            if route is None:
                return self.error(404, "not found")
            route()
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as e:  # pragma: no cover
            log(f"ERROR GET {path}: {e!r}")
            self.error(500, "internal error")

    def do_POST(self):
        path = urlparse(self.path).path
        try:
            if path not in ("/api/assess", "/api/compare/save"):
                return self.error(404, "not found")
            if not self.headers.get("Content-Type", "").startswith("application/json"):
                return self.error(415, "Content-Type must be application/json")
            n = int(self.headers.get("Content-Length") or 0)
            if n <= 0 or n > MAX_BODY:
                return self.error(413, "bad body size")
            data = json.loads(self.rfile.read(n))
            if not isinstance(data, dict):
                return self.error(400, "expected a JSON object")
            if path == "/api/compare/save":
                return self.compare_save(data)
            if not self.visible_item(data.get("item_id")):
                return self.error(404, "unknown item")
            client = f"{self.client_address[0]} {self.headers.get('User-Agent', '')}"
            rec = annodb.add_assessment(db(), CFG["db"], data, client=client)
            log(f"assessment {rec['id']} item={rec['item_id']} verdict={rec['verdict']}")
            self.send_json(rec, 201)
        except (ValueError, KeyError, json.JSONDecodeError) as e:
            self.error(400, str(e))
        except sqlite3.IntegrityError as e:
            self.error(400, f"rejected by database: {e}")
        except Exception as e:  # pragma: no cover
            log(f"ERROR POST {path}: {e!r}")
            self.error(500, "internal error")

    def do_HEAD(self):
        self.error(405, "method not allowed")

    def do_PUT(self):
        self.error(405, "method not allowed")

    do_DELETE = do_PATCH = do_PUT

    # ---------- handlers ----------
    def page(self, name="index.html"):
        body = (STATIC / name).read_bytes()
        self.send_common(200, "text/html; charset=utf-8", len(body), {
            "Cache-Control": "no-cache",
            "Content-Security-Policy": "default-src 'self'; img-src 'self' data:; style-src 'self' 'unsafe-inline'; "
                                       "script-src 'self' 'unsafe-inline'; connect-src 'self'; frame-ancestors 'none'",
        })
        self.wfile.write(body)

    def health(self):
        n = db().execute("SELECT COUNT(*) FROM items").fetchone()[0]
        self.send_json({"ok": True, "items": n, "gated_visible": True if not CFG["require_token"] else "token",
                        "pid": os.getpid()})

    def gated_login(self):
        tok = self.q().get("token", "")
        if not (CFG["require_token"] and CFG["gated_token"] and tok and hmac.compare_digest(tok, CFG["gated_token"])):
            return self.error(403, "gated tier not available")
        body = b"ok"
        self.send_common(303, "text/plain", len(body), {
            "Location": "/",
            "Set-Cookie": f"gt={tok}; HttpOnly; SameSite=Strict; Path=/; Max-Age=2592000",
        })
        self.wfile.write(body)

    def visible_item(self, item_id):
        if not item_id:
            return None
        return db().execute(f"SELECT * FROM items i WHERE i.item_id=? AND {self.tier_clause()}",
                            (item_id,)).fetchone()

    def api_config(self):
        con, tc = db(), self.tier_clause()
        sets = [dict(r) for r in con.execute(
            f"""SELECT i.source_set AS name, COUNT(*) AS n,
                       SUM(EXISTS(SELECT 1 FROM assessments a WHERE a.item_id=i.item_id)) AS assessed
                FROM items i WHERE i.active=1 AND {tc} GROUP BY i.source_set ORDER BY MIN(i.added_at)""")]
        self.send_json({"sets": sets, "verdicts": annodb.VERDICTS, "tristate": annodb.TRISTATE,
                        "gated_access": self.gated_ok()})

    def _scope(self, q):
        where, params = [f"i.active=1", self.tier_clause()], []
        if q.get("set"):
            where.append("i.source_set=?")
            params.append(q["set"])
        return where, params

    def api_items(self):
        q, con = self.q(), db()
        where, params = self._scope(q)
        dxs = [r[0] or "(none)" for r in con.execute(
            f"SELECT DISTINCT i.diagnosis FROM items i WHERE {' AND '.join(where)} ORDER BY 1", params)]
        if q.get("dx"):
            if q["dx"] == "(none)":
                where.append("i.diagnosis IS NULL")
            else:
                where.append("i.diagnosis=?")
                params.append(q["dx"])
        rows = con.execute(
            f"""SELECT i.item_id, i.source_set, i.source_id, i.title, i.diagnosis,
                       (SELECT COUNT(*) FROM assessments a WHERE a.item_id=i.item_id) AS n_assess,
                       (SELECT a.verdict FROM assessments a WHERE a.item_id=i.item_id ORDER BY a.id DESC LIMIT 1) AS last_verdict
                FROM items i WHERE {' AND '.join(where)} ORDER BY i.source_set, i.ord, i.item_id""", params).fetchall()
        total = len(rows)
        assessed = sum(1 for r in rows if r["n_assess"])
        flt = q.get("filter", "unassessed")
        if flt == "unassessed":
            rows = [r for r in rows if not r["n_assess"]]
        elif flt == "assessed":
            rows = [r for r in rows if r["n_assess"]]
        self.send_json({"items": [dict(r) for r in rows], "total": total, "assessed": assessed,
                        "diagnoses": dxs})

    def _item_payload(self, row):
        d = dict(row)
        d["metadata"] = json.loads(d["metadata"] or "{}")
        d.pop("image_path", None)  # never expose filesystem paths to clients
        d.pop("manifest", None)
        d["image_url"] = "/img?id=" + quote(d["item_id"], safe="")
        d["preview_url"] = d["image_url"] + "&v=preview"
        return d

    def api_item(self):
        row = self.visible_item(self.q().get("id"))
        if row is None:
            return self.error(404, "unknown item")
        item = self._item_payload(row)
        item["history"] = [dict(r) for r in db().execute(
            "SELECT * FROM assessments WHERE item_id=? ORDER BY id DESC", (row["item_id"],))]
        for h in item["history"]:
            h.pop("client", None)
        base_id = item["metadata"].get("edit_base_item")
        base = self.visible_item(base_id) if base_id else None
        item["base"] = None if base is None else {
            k: v for k, v in self._item_payload(base).items()
            if k in ("item_id", "source_id", "prompt", "image_url", "preview_url")}
        self.send_json(item)

    def api_assessments(self):
        q, con = self.q(), db()
        where, params = [self.tier_clause()], []
        if q.get("since"):
            where.append("a.created_at >= ?")
            params.append(q["since"])
        if q.get("set"):
            where.append("i.source_set = ?")
            params.append(q["set"])
        if q.get("item_id"):
            where.append("a.item_id = ?")
            params.append(q["item_id"])
        rows = con.execute(
            f"""SELECT a.*, i.source_set, i.tier, i.title, i.diagnosis, i.prompt, i.caption, i.metadata
                FROM assessments a JOIN items i ON i.item_id=a.item_id
                WHERE {' AND '.join(where)} ORDER BY a.id""", params).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["metadata"] = json.loads(d["metadata"] or "{}")
            d.pop("client", None)
            out.append(d)
        self.send_json({"count": len(out), "assessments": out})

    def image(self):
        row = self.visible_item(self.q().get("id"))
        if row is None:
            return self.error(404, "unknown item")
        self.send_image(row, preview=self.q().get("v") == "preview")

    def static_file(self, name):
        body = (STATIC / name).read_bytes()
        self.send_common(200, STATIC_FILES[name], len(body), {"Cache-Control": "no-cache"})
        self.wfile.write(body)

    def send_image(self, row, preview=False):
        p = Path(row["image_path"])
        if p.suffix.lower() not in IMG_EXT or not p.is_file():
            return self.error(404, "image missing")
        if preview:
            p = preview_for(p)
        st = p.stat()
        etag = f'"{st.st_mtime_ns:x}-{st.st_size:x}"'
        if self.headers.get("If-None-Match") == etag:
            self.send_common(304, "application/octet-stream", 0, {"ETag": etag})
            return
        ctype = mimetypes.guess_type(p.name)[0] or "application/octet-stream"
        self.send_common(200, ctype, st.st_size, {"ETag": etag, "Cache-Control": "private, max-age=3600"})
        with open(p, "rb") as fh:
            shutil.copyfileobj(fh, self.wfile, 1 << 20)

    # ---------- comparison mode ----------
    def compare_page(self):
        self.page("compare.html")

    def load_group(self, group_id):
        return db().execute("SELECT * FROM compare_groups WHERE group_id=?", (group_id,)).fetchone()

    @staticmethod
    def session_ok(session):
        return bool(session) and bool(SESSION_RE.fullmatch(session))

    def layout(self, group, session, blind):
        """Columns in display order: blind -> deterministic shuffle per (session, group); else canonical order."""
        arms = json.loads(group["arms"])
        if blind:
            seed = int(hashlib.sha256(f"{session}|{group['group_id']}".encode()).hexdigest()[:16], 16)
            arms = arms[:]
            random.Random(seed).shuffle(arms)
        return arms

    def visible_tiers_ok(self, item_row):
        return item_row is not None and (item_row["tier"] == annodb.OPEN_TIER or self.gated_ok())

    def api_compare_config(self):
        con = db()
        sets = [dict(r) for r in con.execute(
            """SELECT g.comparison_set AS name, COUNT(*) AS n,
                      SUM(EXISTS(SELECT 1 FROM comparisons c WHERE c.group_id=g.group_id)) AS compared
               FROM compare_groups g GROUP BY g.comparison_set ORDER BY MIN(g.added_at)""")]
        self.send_json({"sets": sets, "verdicts": annodb.VERDICTS, "gated_access": self.gated_ok()})

    def api_compare_groups(self):
        q, con = self.q(), db()
        where, params = ["1=1"], []
        if q.get("set"):
            where.append("g.comparison_set=?")
            params.append(q["set"])
        scope = " AND ".join(where)
        dxs = [r[0] for r in con.execute(f"SELECT DISTINCT g.diagnosis FROM compare_groups g WHERE {scope} AND g.diagnosis IS NOT NULL ORDER BY 1", params)]
        fsts = [r[0] for r in con.execute(f"SELECT DISTINCT g.fst FROM compare_groups g WHERE {scope} AND g.fst IS NOT NULL ORDER BY 1", params)]
        if q.get("dx"):
            where.append("g.diagnosis=?")
            params.append(q["dx"])
        if q.get("fst"):
            where.append("g.fst=?")
            params.append(int(q["fst"]))
        rows = con.execute(
            f"""SELECT g.group_id, g.comparison_set, g.group_key, g.diagnosis, g.fst, g.arms,
                       (SELECT COUNT(*) FROM comparisons c WHERE c.group_id=g.group_id) AS n_comp
                FROM compare_groups g WHERE {" AND ".join(where)} ORDER BY g.comparison_set, g.ord, g.group_id""",
            params).fetchall()
        gated = self.gated_ok()
        tiers = {} if gated else dict(con.execute("SELECT item_id, tier FROM items WHERE tier != ?", (annodb.OPEN_TIER,)))
        out = []
        for r in rows:
            arms = json.loads(r["arms"])
            if len(arms) < 2:
                continue
            d = {k: r[k] for k in ("group_id", "comparison_set", "group_key", "diagnosis", "fst", "n_comp")}
            d["n_arms"] = len(arms)
            d["n_hidden"] = sum(1 for a in arms if a["item_id"] in tiers)
            out.append(d)
        total, compared = len(out), sum(1 for d in out if d["n_comp"])
        flt = q.get("filter", "uncompared")
        if flt == "uncompared":
            out = [d for d in out if not d["n_comp"]]
        elif flt == "compared":
            out = [d for d in out if d["n_comp"]]
        self.send_json({"groups": out, "total": total, "compared": compared, "diagnoses": dxs, "fsts": fsts})

    def api_compare_group(self):
        q, con = self.q(), db()
        group = self.load_group(q.get("id"))
        session = q.get("session", "")
        if group is None:
            return self.error(404, "unknown group")
        if not self.session_ok(session):
            return self.error(400, "bad session id")
        blind = q.get("blind", "1") != "0"
        arms = self.layout(group, session, blind)
        if len(arms) < 2:
            return self.error(404, "group has fewer than 2 arms")
        # labels are revealed after this session saved a comparison for this group
        saved_here = con.execute("SELECT 1 FROM comparisons WHERE group_id=? AND session_id=? LIMIT 1",
                                 (group["group_id"], session)).fetchone() is not None
        reveal = (not blind) or saved_here
        pos_of = {a["item_id"]: i for i, a in enumerate(arms)}
        cols = []
        for i, a in enumerate(arms):
            item = con.execute("SELECT * FROM items WHERE item_id=?", (a["item_id"],)).fetchone()
            col = {"pos": i, "letter": LETTERS[i], "gated_hidden": False, "missing": item is None}
            if reveal:
                col["label"], col["item_id"] = a["label"], a["item_id"]
            if item is not None and not self.visible_tiers_ok(item):
                col["gated_hidden"] = True
            elif item is not None:
                meta = json.loads(item["metadata"] or "{}")
                if not reveal:
                    meta = {k: v for k, v in meta.items() if k not in BLIND_HIDE}
                col.update({
                    "image_url": f"/cimg?g={quote(group['group_id'], safe='')}&s={session}&b={int(blind)}&p={i}",
                    "preview_url": f"/cimg?g={quote(group['group_id'], safe='')}&s={session}&b={int(blind)}&p={i}&v=preview",
                    "prompt": item["prompt"], "caption": item["caption"], "title": item["title"],
                    "tier": item["tier"], "metadata": meta,
                    "history": [{k: v for k, v in dict(h).items() if k != "client"} for h in con.execute(
                        "SELECT * FROM assessments WHERE item_id=? ORDER BY id DESC", (item["item_id"],))],
                })
            cols.append(col)
        hist = []
        for c in con.execute("SELECT * FROM comparisons WHERE group_id=? ORDER BY id DESC", (group["group_id"],)):
            d = {k: c[k] for k in ("id", "created_at", "annotator", "blind", "no_best", "comment", "combination_note")}
            d["best_letter"] = LETTERS[pos_of[c["best_item_id"]]] if c["best_item_id"] in pos_of else None
            d["ranking_letters"] = [LETTERS[pos_of[x]] for x in json.loads(c["ranking"] or "[]") if x in pos_of]
            shown = json.loads(c["shown_order"])
            d["then_shown_as"] = "".join(LETTERS[pos_of[x]] if x in pos_of else "?" for x in shown)
            if reveal:
                d["shown_labels"] = json.loads(c["shown_labels"])
            hist.append(d)
        self.send_json({
            "group_id": group["group_id"], "comparison_set": group["comparison_set"], "group_key": group["group_key"],
            "plan": json.loads(group["plan"] or "{}"), "arms_sig": annodb.arms_sig(json.loads(group["arms"])),
            "blind": blind, "revealed": reveal, "columns": cols, "comparisons": hist,
            "gated_access": self.gated_ok()})

    def compare_image(self):
        q = self.q()
        group = self.load_group(q.get("g"))
        session = q.get("s", "")
        if group is None or not self.session_ok(session):
            return self.error(404, "unknown image")
        arms = self.layout(group, session, q.get("b", "1") != "0")
        try:
            a = arms[int(q.get("p", "-1"))]
        except (ValueError, IndexError):
            return self.error(404, "unknown image")
        item = db().execute("SELECT * FROM items WHERE item_id=?", (a["item_id"],)).fetchone()
        if not self.visible_tiers_ok(item):
            return self.error(404, "unknown image")
        self.send_image(item)

    def compare_save(self, data):
        con = db()
        group = self.load_group(data.get("group_id"))
        session = str(data.get("session") or "")
        if group is None:
            return self.error(404, "unknown group")
        if not self.session_ok(session):
            return self.error(400, "bad session id")
        if data.get("arms_sig") != annodb.arms_sig(json.loads(group["arms"])):
            return self.error(409, "this group changed since the page was loaded; reload it (your text is kept)")
        blind = bool(data.get("blind", True))
        arms = self.layout(group, session, blind)
        n = len(arms)
        items = [con.execute("SELECT * FROM items WHERE item_id=?", (a["item_id"],)).fetchone() for a in arms]
        if any(i is None for i in items):
            return self.error(409, "group references a missing item")
        visible = [self.visible_tiers_ok(i) for i in items]
        cols = data.get("columns") or []
        if not isinstance(cols, list) or len(cols) > n:
            return self.error(400, "bad columns")
        column_data = [None] * n
        for c in cols:
            if not isinstance(c, dict):
                return self.error(400, "bad column")
            p = int(c.get("pos", -1))
            if not 0 <= p < n:
                return self.error(400, "bad column position")
            column_data[p] = {k: c.get(k) for k in ("verdict", "realism", "dx_recognisable", "skin_type_correct",
                                                      "site_correct", "stage_correct", "comment", "suggested_caption")}
            if not visible[p] and any(v not in (None, "") for v in column_data[p].values()):
                return self.error(403, "cannot assess a gated item without gated access")

        def pos_item(p):
            p = int(p)
            if not 0 <= p < n:
                raise ValueError("bad position")
            if not visible[p]:
                raise PermissionError
            return items[p]["item_id"]

        try:
            best, no_best = data.get("best"), False
            if best == "none":
                best, no_best = None, True
            elif best not in (None, ""):
                best = pos_item(best)
            else:
                best = None
            ranking = data.get("ranking") or []
            if not isinstance(ranking, list) or len(set(map(int, ranking))) != len(ranking):
                return self.error(400, "ranking must list distinct positions")
            ranking = [pos_item(p) for p in ranking]
        except PermissionError:
            return self.error(403, "cannot choose a gated item without gated access")
        client = f"{self.client_address[0]} {self.headers.get('User-Agent', '')}"
        res = annodb.add_comparison(
            con, CFG["db"], group, session, blind, items, [a["label"] for a in arms], column_data, best, no_best,
            ranking, data.get("comment"), data.get("combination_note"), data.get("annotator"),
            annodb.arms_sig(json.loads(group["arms"])), client=client)
        comp = {k: v for k, v in res["comparison"].items() if k != "client"}
        log(f"comparison {comp['id']} group={group['group_id']} best={comp['best_item_id']} "
            f"assessments={res['assessment_ids']}")
        self.send_json({"comparison": comp, "assessment_ids": res["assessment_ids"],
                        "reveal": [{"pos": i, "letter": LETTERS[i], "label": a["label"], "item_id": a["item_id"]}
                                   for i, a in enumerate(arms)]}, 201)

    def api_comparisons(self):
        q, con = self.q(), db()
        where, params = ["1=1"], []
        if q.get("since"):
            where.append("c.created_at >= ?")
            params.append(q["since"])
        if q.get("set"):
            where.append("c.comparison_set = ?")
            params.append(q["set"])
        gated = self.gated_ok()
        tiers = {} if gated else dict(con.execute("SELECT item_id, tier FROM items WHERE tier != ?", (annodb.OPEN_TIER,)))
        out = []
        for r in con.execute(f"""SELECT c.*, g.arms, g.plan FROM comparisons c JOIN compare_groups g USING (group_id)
                                 WHERE {" AND ".join(where)} ORDER BY c.id""", params):
            d = dict(r)
            d.pop("client", None)
            for k in ("shown_order", "shown_labels", "ranking", "assessment_ids", "arms", "plan"):
                d[k] = json.loads(d[k]) if d[k] else None
            if any(x in tiers for x in d["shown_order"]):
                continue
            out.append(d)
        self.send_json({"count": len(out), "comparisons": out})


def preview_for(src: Path) -> Path:
    """Downscaled JPEG (longest side <= PREVIEW_MAX) cached by path+mtime+size; small JPEGs are served as is."""
    st = src.stat()
    try:
        from PIL import Image
    except ImportError:
        return src
    key = hashlib.sha1(f"{src}|{st.st_mtime_ns}|{st.st_size}|{PREVIEW_MAX}".encode()).hexdigest()
    out = PREVIEW_DIR / key[:2] / f"{key}.jpg"
    if out.exists():
        return out
    with _preview_lock:
        if out.exists():
            return out
        try:
            with Image.open(src) as im:
                if (max(im.size) <= PREVIEW_MAX and src.suffix.lower() in (".jpg", ".jpeg")
                        and st.st_size <= 700_000):
                    return src
                im.draft("RGB", (PREVIEW_MAX, PREVIEW_MAX))
                im = im.convert("RGB")
                im.thumbnail((PREVIEW_MAX, PREVIEW_MAX), Image.LANCZOS)
                out.parent.mkdir(parents=True, exist_ok=True)
                tmp = out.with_suffix(".tmp")
                im.save(tmp, "JPEG", quality=86, optimize=True, progressive=True)
                os.replace(tmp, out)
        except Exception as e:  # pragma: no cover - fall back to the original
            log(f"preview failed for {src}: {e!r}")
            return src
    return out


def backup_loop():
    while True:
        try:
            made = annodb.backup(CFG["db"])
            if made:
                log(f"backup written: {made}")
        except Exception as e:  # pragma: no cover
            log(f"backup failed: {e!r}")
        time.sleep(3600)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--db", default=str(annodb.DEFAULT_DB))
    ap.add_argument("--include-gated", action="store_true",
                    help="no-op, kept for compatibility: gated tiers are visible by default")
    ap.add_argument("--require-gate-token", action="store_true",
                    help="opt-in: hide gated-tier items (placeholders in /compare) unless the browser opened "
                         "/gated?token=<token> once")
    ap.add_argument("--gated-token-file", default=str(HERE / ".gated_token"),
                    help="token file for --require-gate-token (created with mode 600 if missing)")
    ap.add_argument("--no-backup", action="store_true", help="disable the daily backup thread")
    args = ap.parse_args()

    CFG["db"] = str(Path(args.db).resolve())
    annodb.connect(CFG["db"]).close()  # create schema
    if args.require_gate_token:
        tf = Path(args.gated_token_file)
        if not tf.exists():
            fd = os.open(tf, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "w") as fh:
                fh.write(secrets.token_urlsafe(32))
        CFG["gated_token"] = tf.read_text().strip()
        CFG["require_token"] = True
        log(f"gate token REQUIRED for gated tiers; access URL: http://<host>:{args.port}/gated?token=<contents of {tf}>")
    if not args.no_backup:
        threading.Thread(target=backup_loop, daemon=True).start()

    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    srv.daemon_threads = True
    gate = "token required" if CFG["require_token"] else "visible without token"
    log(f"serving on http://{args.host}:{args.port}  db={CFG['db']}  gated tiers: {gate}  pid={os.getpid()}")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
