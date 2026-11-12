"""Ingest the 14 web dermatology atlases in medical-datasets/dermatology/.

Join rules (verified, see dataset/docs/PROVENANCE.md):
* atlasdermatologico: images/{k:05d}.jpg <-> meta.json[k]. The `path` field in
  meta.json (and every derived jsonl) is shifted by +1 because the download loop
  incremented the counter before recording the path.
* hellenicdermatlas: images/{int(id)-1:04d}.jpg; dermatlas/iconotheque by index.
* dermnet: images/{i:05d}.jpg <-> dn_images.parquet row i.
* altmeyers: captions/urls mirror images/ (the *-backup folder maps to its parent).
* dermaamin: content/<title>.json, images/<title>/{i:03d}.jpg <-> image_links[i].
* viamedici: only entries with subject == "Dermatologie".
* restored (watermark-inpainted) copies: match/marks.tsv path -> key; NOT the
  row order of dermatology/meta.parquet (they diverge from row 21238 on).
Paths are NFC-normalised because macOS stores some names decomposed.

Also attaches pre-existing LLM annotations (Claude 3 Opus, 2024):
* llm_vision_*: image-based description / Fitzpatrick estimate / diagnosis
  (output/derma-claude-3-opus.jsonl, keyed by image path).
* llm_text_*: structured fields extracted from the caption text
  (output/meta-opus.jsonl, keyed by the shifted path for atlasdermatologico).
"""
import ast
import glob
import json
import os
import unicodedata
from urllib.parse import parse_qs, urlparse

import pandas as pd

from common import MED, WORK, clean_text, empty_row, jdump

D = MED / "dermatology"


def nfc(s):
    return unicodedata.normalize("NFC", s)


def lit(v):
    """Parse python-repr strings like "['a', 'b']" or "{'k': 'v'}"."""
    if isinstance(v, str) and v[:1] in "[{(":
        try:
            return ast.literal_eval(v)
        except Exception:
            return v
    return v


def joinlist(v, sep=", "):
    v = lit(v)
    if isinstance(v, (list, tuple)):
        v = sep.join(str(x) for x in v if str(x).strip())
    return clean_text(v) if v not in ("Not Entered",) else None


def load(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def row(source, path, language="en", **kw):
    return empty_row(source=source, source_type="atlas", src_path=nfc(str(path)),
                     language=language, **kw)


# ---------------------------------------------------------------- per source
def altmeyers():
    base = D / "altmeyers/high_res_3categories"
    misc = {}
    with open(MED / "meta-complete.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if "/altmeyers/" in r["img"]:
                misc[nfc(r["img"][2:])] = r.get("misc") or {}
    for p in sorted(glob.glob(str(base / "images/*/*/*/*.jpg"))):
        rel = os.path.relpath(p, MED)
        cap_p = p.replace("/images/", "/captions/").replace(".jpg", ".txt").replace("-backup/", "/")
        url_p = p.replace("/images/", "/urls/").replace(".jpg", ".txt").replace("-backup/", "/")
        cap = open(cap_p, encoding="utf-8").read() if os.path.exists(cap_p) else None
        img_url = open(url_p, encoding="utf-8").read().strip() if os.path.exists(url_p) else None
        loc, color, efflor = p.split("/")[-4:-1]
        m = misc.get(nfc(rel), {})
        yield row("altmeyers", p, caption=clean_text(cap), image_url=img_url,
                  diagnosis=clean_text(m.get("diagnosis_name")), body_site=loc,
                  extra=jdump({"icd10": m.get("diagnosis_code"), "color": color,
                               "efflorescence": efflor.replace("-backup", "")}))


def atlasdermatologico():
    meta = load(D / "atlasdermatologico/meta.json")
    for k, rec in enumerate(meta):
        p = D / f"atlasdermatologico/images/{k:05d}.jpg"
        if not p.exists():
            continue
        yield row("atlasdermatologico", p, diagnosis=clean_text(rec.get("diagnosis")),
                  url=rec.get("url"), image_url=rec.get("image_url"),
                  extra=jdump({"image_id": (rec.get("params") or {}).get("imageId")}))


def simple_path_meta(source, language="en", url_prefix=""):
    meta = load(D / source / "meta.json")
    for rec in meta:
        p = D / source / rec["path"].lstrip("./")
        if not p.exists():
            p = D / source / nfc(rec["path"].lstrip("./"))
        if not p.exists():
            continue
        url = rec.get("url")
        if url and url_prefix and not url.startswith("http"):
            url = url_prefix + url
        yield row(source, p, language=language, diagnosis=clean_text(rec.get("diagnosis")),
                  url=url, image_url=rec.get("image_url"))


def dermatlas():
    meta = load(D / "dermatlas/meta.json")
    for i, rec in enumerate(meta):
        p = D / f"dermatlas/images/{i:04d}.jpg"
        if not p.exists():
            continue
        demo = lit(rec.get("Patient Demographics")) or {}
        lesion = lit(rec.get("Lesion Characteristics")) or {}
        visit = lit(rec.get("Visit Specifics")) or {}
        demo = demo if isinstance(demo, dict) else {}
        lesion = lesion if isinstance(lesion, dict) else {}
        visit = visit if isinstance(visit, dict) else {}
        yield row("dermatlas", p, diagnosis=clean_text(rec.get("diagnosis")),
                  url=rec.get("url"), image_url=rec.get("image_url"),
                  age=joinlist(demo.get("Age at Visit")), sex=joinlist(demo.get("Gender")),
                  body_site=joinlist(lesion.get("Localization(s)")),
                  extra=jdump({
                      "ethnicity": joinlist(demo.get("Ethnicity")),
                      "primary_morphology": joinlist(lesion.get("Primary Morphology")),
                      "secondary_morphology": joinlist(lesion.get("Secondary Morphology")),
                      "descriptors": joinlist(lesion.get("Descriptors")),
                      "palpation": joinlist(lesion.get("Palpation")),
                      "colors": joinlist(lesion.get("Color(s)")),
                      "biopsy": joinlist(visit.get("Biopsy")),
                      "differential_diagnoses": joinlist(visit.get("Differential Diagnoses")),
                      "treatment": joinlist(visit.get("Actual Treatment")),
                  }))


def dermis():
    meta = load(D / "dermis/meta.json")
    for rec in meta:
        p = D / "dermis" / rec["path"].lstrip("./")
        if not p.exists():
            continue
        site = clean_text(rec.get("localization")) or clean_text(rec.get("localisation"))
        yield row("dermis", p, diagnosis=clean_text(rec.get("diagnosis")) or clean_text(rec.get("title")),
                  title=clean_text(rec.get("title")), url=rec.get("url"), image_url=rec.get("img-src"),
                  body_site=site, age=clean_text(rec.get("patient age")),
                  sex=clean_text(rec.get("patient sex")),
                  extra=jdump({"lesions": clean_text(rec.get("lesions")),
                               "additional_descriptions": clean_text(rec.get("additional descriptions")),
                               "descent": clean_text(rec.get("decent of patient"))}))


def dermnet():
    dn = pd.read_parquet(D / "dermnet/dn_images.parquet")
    for i, r in enumerate(dn.itertuples(index=False)):
        p = D / f"dermnet/images/{i:05d}.jpg"
        if not p.exists():
            continue
        q = parse_qs(urlparse(r.page).query)
        label = clean_text((q.get("label") or [""])[0])
        cap = clean_text((q.get("caption") or [""])[0])
        topic = r.url.rstrip("/").split("/")[-1].replace("-", " ")
        title = clean_text(r.title)
        text = cap or label or (title if title and not title.lower().startswith("figure") else None)
        yield row("dermnet", p, caption=text, title=title, diagnosis=clean_text(topic),
                  url="https://dermnetnz.org" + r.url,
                  image_url="https://dermnetnz.org" + r.watermark,
                  extra=jdump({"label": label, "is_pathology_topic": "pathology" in r.url}))


def hellenic():
    meta = load(D / "hellenicdermatlas/meta.json")
    for rec in meta:
        p = D / f"hellenicdermatlas/images/{int(rec['id']) - 1:04d}.jpg"
        if not p.exists():
            continue
        yield row("hellenicdermatlas", p, diagnosis=clean_text(rec.get("Diagnose")),
                  url=rec.get("href"), image_url=rec.get("image_src"),
                  extra=jdump({"category": clean_text(rec.get("Category")),
                               "contributor": clean_text(rec.get("Contributor")),
                               "date": rec.get("Date")}))


def iconotheque():
    for mp in sorted(glob.glob(str(D / "iconotheque/meta/*.json"))):
        i = os.path.splitext(os.path.basename(mp))[0]
        p = D / f"iconotheque/images/{i}.jpg"
        if not p.exists():
            continue
        rec = load(mp)
        diag = joinlist(rec.get("Diagnostic"), "; ") or joinlist(rec.get("Maladie"), "; ")
        desc = joinlist(rec.get("Description des lésions"), " ")
        yield row("iconotheque", p, language="fr", caption=desc, diagnosis=diag,
                  body_site=joinlist(rec.get("Localisation des lésions")),
                  age=joinlist(rec.get("Age patient")), sex=joinlist(rec.get("Sexe patient")),
                  url=rec.get("url"),
                  extra=jdump({"disease": joinlist(rec.get("Maladie")),
                               "chapter": joinlist(rec.get("Chapitre du cours")),
                               "elementary_lesion": joinlist(rec.get("Lésion élémentaire")),
                               "copyright": joinlist(rec.get("Droits d'auteur"))}))


def dermaamin():
    for mp in sorted(glob.glob(str(D / "dermaamin/content/*.json"))):
        rec = load(mp)
        title = rec["title"]
        folder = D / "dermaamin/images" / title.replace("/", " - ")
        links = lit(rec.get("image_links")) or []
        if not folder.exists():
            continue
        for p in sorted(folder.iterdir()):
            if p.suffix.lower() not in (".jpg", ".jpeg", ".png", ".gif") or p.stat().st_size < 1024:
                continue
            try:
                idx = int(p.stem)
            except ValueError:
                idx = -1
            img_url = links[idx] if 0 <= idx < len(links) else None
            yield row("dermaamin", p, diagnosis=clean_text(title), image_url=img_url,
                      extra=jdump({"disease_abstract": clean_text(rec.get("abstract"))}))


def viamedici():
    entries = load(D / "viamedici/blickdiagnosen.json")["navigationEntries"]
    for e in entries:
        if e.get("subject") != "Dermatologie":
            continue
        p = D / f"viamedici/images/{e['id']}.jpg"
        if not p.exists():
            continue
        yield row("viamedici", p, language="de", diagnosis=clean_text(e.get("label")),
                  caption=clean_text(e.get("caseDescription")), image_url=e.get("imageUrl"),
                  extra=jdump({"topic": e.get("topic"), "subtopic": e.get("subtopic")}))


SOURCES = {
    "altmeyers": altmeyers,
    "atlasdermatologico": atlasdermatologico,
    "danderm": lambda: simple_path_meta("danderm"),
    "dermaamin": dermaamin,
    "dermatlas": dermatlas,
    "dermatoweb": lambda: simple_path_meta("dermatoweb", language="es"),
    "dermis": dermis,
    "dermnet": dermnet,
    "hellenicdermatlas": hellenic,
    "iconotheque": iconotheque,
    "mddk": lambda: simple_path_meta("mddk"),
    "meddean": lambda: simple_path_meta(
        "meddean", url_prefix="https://www.meddean.luc.edu/lumen/meded/medicine/dermatology/melton/"),
    "uiowa": lambda: simple_path_meta("uiowa"),
    "viamedici": viamedici,
}


# ------------------------------------------------------- linked annotations
def restored_map():
    m = pd.read_csv(MED / "match/marks.tsv", sep="\t")
    out = {}
    for p, k in zip(m.path, m.key):
        p = p.replace("../", "")
        if "/dermis/images/" in p:  # marks.tsv was built on the old dermis layout
            p = "dermatology/dermis/images/" + os.path.basename(p)
        rp = MED / f"dermatology-restored/{int(k):05d}.jpg"
        if rp.exists():
            out[nfc(str(MED / p))] = str(rp)
    return out


def parse_json_text(t):
    if not isinstance(t, str):
        return None
    try:
        return json.loads(t)
    except Exception:
        a, b = t.find("{"), t.rfind("}")
        if a >= 0 and b > a:
            try:
                return json.loads(t[a:b + 1])
            except Exception:
                return None
    return None


def vision_llm_map():
    out = {}
    with open(MED / "output/derma-claude-3-opus.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            resp = r.get("response")
            if isinstance(resp, str):
                resp = parse_json_text(resp)
            if not isinstance(resp, dict) or "content" not in resp:
                continue
            j = parse_json_text(resp["content"][0].get("text", ""))
            if not isinstance(j, dict):
                continue
            out[nfc(str(MED / r["image"][2:]))] = j
    return out


def text_llm_map():
    """meta-opus.jsonl keyed by its own (shifted) path -> remap to real file."""
    out = {}
    with open(MED / "output/meta-opus.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            p = r["img"][2:].replace("/images_old/", "/images/")
            if "/atlasdermatologico/images/" in p:
                k = int(os.path.splitext(os.path.basename(p))[0]) - 1  # undo +1 shift
                p = f"dermatology/atlasdermatologico/images/{k:05d}.jpg"
            j = parse_json_text(r.get("meta"))
            if isinstance(j, dict):
                out[nfc(str(MED / p))] = j
    return out


def main():
    rows = []
    for name, fn in SOURCES.items():
        n0 = len(rows)
        rows.extend(fn())
        print(f"{name:20s} {len(rows) - n0:6d}")
    df = pd.DataFrame(rows)

    rmap = restored_map()
    df["restored_path"] = df.src_path.map(rmap)
    vis = vision_llm_map()
    txt = text_llm_map()

    def vfield(p, k):
        j = vis.get(p)
        v = j.get(k) if j else None
        return json.dumps(v, ensure_ascii=False) if isinstance(v, list) else clean_text(v)

    for k in ["description", "fitz_patrick_scale", "most_likely_diagnosis",
              "differential_diagnoses", "body_parts_seen"]:
        df[f"llm_vision_{k}"] = [vfield(p, k) for p in df.src_path]
    df["llm_text_structured"] = [json.dumps(txt[p], ensure_ascii=False) if p in txt else None
                                 for p in df.src_path]
    print("total", len(df), "| restored", df.restored_path.notna().sum(),
          "| llm_vision", df.llm_vision_description.notna().sum(),
          "| llm_text", df.llm_text_structured.notna().sum())
    df.to_parquet(WORK / "atlases.parquet", index=False)


if __name__ == "__main__":
    main()
