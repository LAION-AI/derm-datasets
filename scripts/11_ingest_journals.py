"""Ingest the journal scrapes in derma-data/ (figure images + captions + article metadata).

For every article html: parse metadata (DOI, title, date, type, language, authors,
keywords, abstract, licence) and the figure -> caption map, then pair every file in
the article's image folder with its caption *by filename* (never by folder name:
JAMA folders often hold a different article than their name says).

Outputs
  work/journals.parquet       one row per (article DOI, image file)
  work/articles.parquet       one row per unique DOI
  work/text_journals.parquet  article-level text (abstract + body) for the text corpus
"""
import json
import os
import re
from concurrent.futures import ProcessPoolExecutor
from urllib.parse import unquote

import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm

from common import DERMA, IMAGE_EXTS, WORK, clean_text, empty_row, jdump

PUBLISHER = {
    "ajcd": "springer", "dat": "springer", "die-dermatologie": "springer",
    "jaad": "elsevier", "jid": "elsevier",
    "jddg": "wiley", "bjd": "wiley", "jeadv": "wiley",
    "jama": "jama",
}
JOURNAL_NAME = {
    "ajcd": "American Journal of Clinical Dermatology",
    "dat": "Dermatology and Therapy",
    "die-dermatologie": "Die Dermatologie (formerly Der Hautarzt)",
    "jaad": "Journal of the American Academy of Dermatology",
    "jid": "Journal of Investigative Dermatology",
    "jddg": "Journal der Deutschen Dermatologischen Gesellschaft",
    "bjd": "British Journal of Dermatology",
    "jeadv": "Journal of the European Academy of Dermatology and Venereology",
    "jama": "JAMA Dermatology",
}
CC_RE = re.compile(r"creativecommons\.org/licenses/([a-z\-]+)/(\d\.\d)", re.I)
CC_TEXT_RE = re.compile(r"Creative Commons Attribution[- ]?(Non-?Commercial)?[- ]?(No-?Derivatives?|NoDerivs)?", re.I)


def txt(el):
    return clean_text(el.get_text(" ", strip=True)) if el else None


def meta(s, name, all_=False):
    tags = s.find_all("meta", attrs={"name": name})
    vals = [t.get("content") for t in tags if t.get("content")]
    if all_:
        return vals
    return vals[0] if vals else None


def cc_from_url(u):
    m = CC_RE.search(u or "")
    if not m:
        return None
    return f"CC {m.group(1).upper()} {m.group(2)}"


def cc_from_text(t):
    m = CC_TEXT_RE.search(t or "")
    if not m:
        return None
    parts = ["CC BY"]
    if m.group(1):
        parts.append("NC")
    if m.group(2):
        parts.append("ND")
    return "-".join(parts) + " (version n/s)"


def fname(u):
    return unquote((u or "").split("?")[0].split("/")[-1])


# ------------------------------------------------------------- publishers
def springer(s):
    figs = {}
    for box in s.select("div.c-article-section__figure"):
        img = box.select_one("figure picture img") or box.find("img")
        if not img:
            continue
        fn = fname(img.get("src") or img.get("data-src"))
        figs[fn] = (txt(box.select_one(".c-article-section__figure-caption")),
                    txt(box.select_one(".c-article-section__figure-description")))
    lic = None
    for x in s.find_all("script", type="application/ld+json"):
        try:
            ld = json.loads(x.string or "")
            lic = cc_from_url((ld.get("mainEntity") or {}).get("license"))
        except Exception:
            pass
        if lic:
            break
    rp = s.select_one('section[data-title="Rights and permissions"]')
    if not lic and rp:
        a = rp.select_one("a[rel=license]")
        lic = cc_from_url(a.get("href") if a else None) or cc_from_text(rp.get_text(" "))
    abstract = (txt(s.select_one('section[data-title="Abstract"]'))
                or txt(s.select_one('section[data-title="Zusammenfassung"]')) or meta(s, "dc.description"))
    kw = [txt(li) for li in s.select("ul.c-article-subject-list li")]
    body = s.select_one("div.c-article-body")
    return dict(
        figs=figs, license=lic, copyright=meta(s, "dc.rights"),
        article_type=meta(s, "citation_article_type") or meta(s, "dc.type"),
        language=meta(s, "citation_language") or meta(s, "dc.language"),
        date=meta(s, "citation_publication_date") or meta(s, "citation_online_date"),
        abstract=abstract, keywords=[k for k in kw if k], body=body,
    )


def elsevier(s):
    figs = {}
    for a in s.select("figure > a.icon-full-screen"):
        fig = a.parent
        lab = txt(fig.select_one(".figure__label"))
        cap = txt(fig.find("figcaption")) or ""
        if lab and cap.startswith(lab):
            cap = cap[len(lab):].strip()
        figs[fname(a.get("href"))] = (lab, cap or None)
    page = s.get_text(" ")
    m = re.search(r"User License:\s*Creative Commons[^|]*?\((CC[^)]*)\)", page)
    lic = m.group(1).replace("CC BY", "CC BY").strip() if m else None
    acc = s.select_one(".meta-panel__access")
    access = None
    if acc:
        cls = " ".join(acc.get("class", []))
        access = "open" if "--open" in cls else "open-archive" if "--archive" in cls else txt(acc)
    cp = re.search(r"Copyright:\s*©[^\n|]{0,150}", page)
    kws = meta(s, "citation_keywords", all_=True)
    kw = [k.strip() for k in (kws[0].split(";") if kws else []) if k.strip()]
    body = s.select_one("section#bodymatter") or s.select_one("section#bodymain") or s.select_one("div.Body")
    return dict(
        figs=figs, license=lic, access=access, copyright=cp.group(0)[11:].strip() if cp else None,
        article_type=txt(s.select_one(".meta-panel__type")),
        language="en", date=meta(s, "citation_date") or meta(s, "citation_online_date"),
        abstract=txt(s.select_one("section#author-abstract")) or meta(s, "citation_abstract"),
        keywords=kw, body=body,
        title_fallback=txt(s.select_one("h1[property=name]")),
    )


def wiley(s):
    figs = {}
    for fig in s.select("figure.figure"):
        rec = (txt(fig.select_one(".figure__title")), txt(fig.select_one(".figure__caption-text")))
        urls = [a["href"] for a in fig.find_all("a", href=True)] + \
               [i.get("data-lg-src") or i.get("src") or "" for i in fig.find_all("img")]
        for u in urls:
            fn = fname(u)
            if re.search(r"\.(jpe?g|png|gif)$", fn, re.I):
                figs.setdefault(fn, rec)
    cc = s.select_one('div.doi-access-wrapper a[href*="creativecommons.org/licenses"]')
    acc = s.select_one(".doi-access")
    cp = s.select_one(".copyright")
    abs_el = s.select_one("section.article-section__abstract")
    return dict(
        figs=figs, license=cc_from_url(cc.get("href")) if cc else None,
        access=txt(acc), copyright=txt(cp),
        article_type=txt(s.select_one("span.primary-heading")),
        language=None, abstract_lang=abs_el.get("lang") if abs_el else None,
        date=meta(s, "citation_publication_date") or meta(s, "citation_online_date"),
        abstract=txt(abs_el), keywords=meta(s, "citation_keywords", all_=True),
        body=s.select_one("section.article-section__full"),
    )


def jama(s):
    figs = {}
    for w in s.select("div.figure-table-wrapper"):
        a = w.select_one("a.view-large")
        if a:
            figs[fname(a.get("href"))] = (txt(w.select_one(".figure-label")), txt(w.select_one(".figure-caption")))
    note = txt(s.select_one(".paraopen-access-note"))
    lic = None
    if note:
        m = re.search(r"CC[- ]BY(?:[- ]NC)?(?:[- ]ND)?", note)
        lic = m.group(0).replace("CC-BY", "CC BY").replace("-", " ").replace("CC BY NC ND", "CC BY-NC-ND").replace("CC BY NC", "CC BY-NC") if m else None
    return dict(
        figs=figs, license=lic, copyright=note,
        article_type=txt(s.select_one(".meta-article-type")), language="en",
        date=meta(s, "citation_publication_date"),
        abstract=clean_text(BeautifulSoup(meta(s, "citation_abstract") or "", "lxml").get_text(" ")),
        keywords=meta(s, "citation_keyword", all_=True),
        body=s.select_one("div.widget-ArticleFulltext") or s.select_one("div.article-full-text"),
    )


PARSERS = {"springer": springer, "elsevier": elsevier, "wiley": wiley, "jama": jama}


# ------------------------------------------------------------ file roles
def role(journal, fn, all_names):
    pub = PUBLISHER[journal]
    low = fn.lower()
    if pub == "springer":
        if re.search(r"_tab\d", low):
            return "table"
        if re.search(r"_fig[a-z]_html", low) or re.search(r"_u\d", low):
            return "unlabelled"
        return "figure"
    if pub == "elsevier":
        if low.startswith("fx"):
            return "supplementary"
        if low.startswith("ga"):
            return "graphical_abstract"
        m = re.match(r"(gr\d+[a-z]?)\.(jpg|gif|png)$", low)
        if m and any(n.lower().startswith(m.group(1) + "_lrg") for n in all_names):
            return "thumbnail"
        return "figure"
    if pub == "wiley":
        if "-gra-" in low or "_gra" in low:
            return "graphical_abstract"
        if re.search(r"_t\d", low):
            return "table"
        if low.endswith(".png") and os.path.splitext(fn)[0] + ".jpg" in all_names:
            return "thumbnail"
        return "figure"
    if pub == "jama":
        if low.startswith("csm_"):
            return "other"
        m = re.search(r"\d(f|t|va)\d+(?:_[\d.]+)?\.png$", low)  # 2021+: _<unix time with decimals>
        if m:
            return {"f": "figure", "t": "table", "va": "visual_abstract"}[m.group(1)]
        return "other"
    return "figure"


def caption_for(fn, figs):
    if fn in figs:
        return figs[fn]
    base = re.sub(r"_lrg(?=\.)", "", fn)
    for k, v in figs.items():
        if re.sub(r"_lrg(?=\.)", "", k) == base:
            return v
    return (None, None)


FIGNUM = re.compile(r"(\d+)")


def body_text(body):
    if body is None:
        return None
    for t in body.select("figure, div.c-article-section__figure, div.figure-table-wrapper, script, style"):
        t.decompose()
    return clean_text(body.get_text(" "))


SENT = re.compile(r"(?<=[.!?])\s+(?=[A-ZÄÖÜ(])")


def mentions(text, label):
    """Up to 3 sentences of the body that cite this figure number (inline context)."""
    if not text or not label:
        return None
    m = FIGNUM.search(label)
    if m:
        n = m.group(1)
        pat = re.compile(rf"\b(Fig\.?|Figs\.?|Figure|Figures|Abb\.?|Abbildung(?:en)?)\s*{n}\b(?![.,]\d)", re.I)
    else:  # single unnumbered figure ("Figure.")
        pat = re.compile(r"\b(Fig\.?|Figure|Abb\.?|Abbildung)(?!\s*\d)\b", re.I)
    out = [s for s in SENT.split(text) if pat.search(s) and len(s) < 1200]
    return " ".join(out[:3]) or None


def language_of(journal, doi, parsed, label):
    if journal == "jddg":
        if re.search(r"_(g|suppl?)(\.x)?$", doi or ""):
            return "de"
        if parsed.get("abstract_lang") in ("de", "en"):
            return parsed["abstract_lang"]
        return "de" if label and label.lower().startswith("abb") else "en"
    lang = (parsed.get("language") or "en").lower()[:2]
    return lang


# --------------------------------------------------------------- per file
def process(args):
    journal, html_path = args
    try:
        raw = open(html_path, "rb").read()
    except Exception:
        return None
    if len(raw) < 200:
        return {"html": html_path, "journal": journal, "status": "empty"}
    s = BeautifulSoup(raw, "lxml")
    doi = meta(s, "citation_doi") or meta(s, "dc.identifier")
    if doi:
        doi = re.sub(r"^(doi:|https?://(dx\.)?doi\.org/)", "", doi.strip(), flags=re.I)
    p = PARSERS[PUBLISHER[journal]](s)
    title = clean_text(meta(s, "citation_title")) or p.get("title_fallback")
    body = body_text(p.pop("body"))
    figs = p.pop("figs")
    rel = os.path.relpath(html_path, DERMA / journal / "html")
    img_dir = DERMA / journal / "images" / os.path.splitext(rel)[0]
    images = []
    if img_dir.is_dir():
        for root, dirs, files in os.walk(img_dir):
            dirs[:] = [d for d in dirs if d != "output_images"]
            for f in files:
                if os.path.splitext(f)[1].lower() in IMAGE_EXTS and not f.startswith("."):
                    images.append(os.path.join(root, f))
    names = {os.path.basename(x) for x in images}
    rows = []
    for ip in sorted(images):
        fn = os.path.basename(ip)
        label, cap = caption_for(fn, figs)
        r = role(journal, fn, names)
        rows.append(dict(src_path=ip, file_role=r, figure_label=label, caption=cap,
                         context=mentions(body, label)))
    date = p.get("date") or ""
    ym = re.search(r"(19|20)\d\d", date)
    lang = language_of(journal, doi, p, next((r["figure_label"] for r in rows if r["figure_label"]), None))
    art = dict(
        html=html_path, journal=journal, status="ok" if doi else "no_doi", doi=doi, title=title,
        year=int(ym.group(0)) if ym else None, date=date, article_type=clean_text(p.get("article_type")),
        language=lang, authors=meta(s, "citation_author", all_=True), keywords=p.get("keywords") or [],
        abstract=p.get("abstract"), license=p.get("license"), access=p.get("access"),
        copyright=clean_text(p.get("copyright")), n_figures_html=len(figs), n_images=len(images),
        has_fulltext=body is not None and len(body or "") > 1000,
        body=body,
    )
    return {"article": art, "images": rows}


def main():
    tasks = []
    for j in PUBLISHER:
        for root, _, files in os.walk(DERMA / j / "html"):
            for f in files:
                if f.endswith(".html"):
                    tasks.append((j, os.path.join(root, f)))
    print(len(tasks), "html files")
    arts, imgs = [], []
    with ProcessPoolExecutor(10) as ex:
        for res in tqdm(ex.map(process, tasks, chunksize=16), total=len(tasks), mininterval=60):
            if not res or "article" not in res:
                if res:
                    arts.append(res)
                continue
            a = res["article"]
            arts.append(a)
            for r in res["images"]:
                imgs.append({**r, "doi": a["doi"], "journal": a["journal"], "html": a["html"]})
    A = pd.DataFrame(arts)
    I = pd.DataFrame(imgs)
    A.to_parquet(WORK / "journal_html_raw.parquet", index=False)

    # one article record per DOI: prefer the html with full text / most figures
    ok = A[A.status == "ok"].copy()
    ok["_score"] = ok.has_fulltext.astype(int) * 1000 + ok.n_figures_html
    best = ok.sort_values("_score", ascending=False).drop_duplicates("doi")
    best = best.assign(base_doi=best.doi.str.replace(r"_(g|suppl?)(?=(\.x)?$)", "", regex=True))
    best["journal_name"] = best.journal.map(JOURNAL_NAME)

    # image rows: unique per (doi, filename), keep only rows with a DOI
    I = I[I.doi.notna()].copy()
    I["fn"] = I.src_path.map(os.path.basename)
    I["_cap"] = I.caption.notna().astype(int)
    I = I.sort_values("_cap", ascending=False).drop_duplicates(["doi", "fn"])
    meta_cols = best.set_index("doi")[["journal", "title", "year", "language", "license", "access",
                                       "copyright", "article_type", "base_doi"]]
    I = I.join(meta_cols, on="doi", rsuffix="_a")

    samples = []
    for r in I.itertuples(index=False):
        lic = r.license if isinstance(r.license, str) else None
        samples.append(empty_row(
            source=r.journal, source_type="journal", src_path=r.src_path,
            caption=r.caption, title=r.title, language=r.language, doi=r.doi,
            url=f"https://doi.org/{r.doi}", year=r.year, article_id=r.doi,
            figure_label=r.figure_label,
            license=lic or (f"© {r.copyright}" if isinstance(r.copyright, str) else "© publisher (all rights reserved)"),
            extra=jdump({"file_role": r.file_role, "context": r.context, "article_type": r.article_type,
                         "access": r.access, "base_doi": r.base_doi}),
        ))
    S = pd.DataFrame(samples)
    S.to_parquet(WORK / "journals.parquet", index=False)
    best.drop(columns=["_score", "body"]).to_parquet(WORK / "articles.parquet", index=False)
    best[["doi", "journal", "title", "year", "language", "license", "article_type", "abstract", "body"]] \
        .rename(columns={"doi": "doc_id", "journal": "source"}) \
        .to_parquet(WORK / "text_journals.parquet", index=False)

    print("articles (unique DOI):", len(best), "| image rows:", len(S))
    print(S.assign(role=I.file_role.values, cap=S.caption.notna()).groupby("source")
          .agg(n=("src_path", "size"), captioned=("cap", "mean")).round(3))
    print(best.groupby("journal").agg(n=("doi", "size"), cc=("license", lambda x: x.notna().sum())))


if __name__ == "__main__":
    main()
