"""Ingest the two Springer textbooks (Braun-Falco 7th ed., Fritsch 3rd ed.).

Each figure in the chapter markdown follows the pattern
    ![Image](../../images/<file>)   (one or more images on one line)
    Abb. <n>
    <caption line>
Braun-Falco: one flat .md per chapter (Anhang.md is a scraper artefact, skipped).
Fritsch: nested index.md files where parents repeat children -> use only the
top-level chapters/*/index.md.

Outputs work/textbooks.parquet (figure samples) and
work/text_textbooks.parquet (chapter-level full text for the text corpus).
"""
import glob
import os
import re

import pandas as pd

from common import DERMA, WORK, clean_text, empty_row, jdump

IMG = re.compile(r"!\[[^\]]*\]\(([^)\s]+\.(?:jpg|jpeg|png|gif))\)", re.I)
ABB = re.compile(r"^Abb\.\s*(\d+(?:\.\d+)?)\s*$")
UNESC = re.compile(r"\\([\\`*_{}\[\]()#+\-.!|<>])")

BOOKS = {
    "braun-falco": dict(
        title="Braun-Falco's Dermatologie, Venerologie und Allergologie, 7. Auflage",
        isbn="978-3-662-49544-5", year=2018, publisher="Springer",
        files=lambda: sorted(glob.glob(str(DERMA / "braun-falco/chapters/*/*.md"))),
    ),
    "fritsch": dict(
        title="Fritsch/Schwarz: Dermatologie Venerologie, 3. Auflage",
        isbn="978-3-662-53647-6", year=2018, publisher="Springer",
        files=lambda: sorted(glob.glob(str(DERMA / "fritsch/chapters/*/index.md"))),
    ),
}


def figures(md_text):
    """Yield (image_basename, fig_no, caption)."""
    lines = md_text.split("\n")
    pending, i = [], 0
    while i < len(lines):
        s = lines[i].strip()
        imgs = IMG.findall(s)
        if imgs and not IMG.sub("", s).strip():
            pending += [os.path.basename(p) for p in imgs]
            i += 1
            continue
        if not s:
            i += 1
            continue
        m = ABB.match(s)
        if m and pending:
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            cap = lines[j].strip() if j < len(lines) else ""
            if cap.startswith(("#", "![")):
                cap = ""
            for p in pending:
                yield p, m.group(1), UNESC.sub(r"\1", cap)
            pending, i = [], j + 1
            continue
        pending = []
        i += 1


def chapter_title(path, book):
    if book == "fritsch":
        name = os.path.basename(os.path.dirname(path))
    else:
        name = os.path.splitext(os.path.basename(path))[0]
    return name


def main():
    samples, texts = [], []
    for book, meta in BOOKS.items():
        img_dir = DERMA / book / "images"
        seen = set()
        for f in meta["files"]():
            txt = open(f, encoding="utf-8").read()
            chap = chapter_title(f, book)
            if book == "fritsch" and chap == "Dermatologie Venerologie":
                continue  # 22-word stub
            m = re.match(r"(\d+)\.\s*(.*)", chap)
            chap_no = m.group(1) if m else None
            part = os.path.basename(os.path.dirname(f)) if book == "braun-falco" else None
            article_id = f"{book}:ch{chap_no or chap}"
            body = IMG.sub("", txt)
            texts.append(dict(
                doc_id=article_id, source=book, source_type="textbook", title=chap,
                section=part, language="de", year=meta["year"],
                n_words=len(body.split()), text=body, src_path=f,
                license=f"© {meta['year']} {meta['publisher']} (all rights reserved)",
            ))
            for img, fig_no, cap in figures(txt):
                path = img_dir / img
                if not path.exists() or img in seen:
                    continue
                seen.add(img)
                label = f"Abb. {fig_no}" if "." in fig_no else f"Abb. {chap_no}.{fig_no}"
                samples.append(empty_row(
                    source=book, source_type="textbook", src_path=str(path),
                    caption=clean_text(cap), title=clean_text(chap), language="de",
                    year=meta["year"], article_id=article_id, figure_label=label,
                    license=f"© {meta['year']} {meta['publisher']} (all rights reserved)",
                    extra=jdump({"book": meta["title"], "isbn": meta["isbn"], "part": part}),
                ))
        n_files = len([p for p in os.listdir(img_dir) if not p.startswith(".")])
        n_cap = sum(1 for s in samples if s["source"] == book and s["caption"])
        print(f"{book}: {n_cap} captioned figures / {n_files} image files")
    pd.DataFrame(samples).to_parquet(WORK / "textbooks.parquet", index=False)
    pd.DataFrame(texts).to_parquet(WORK / "text_textbooks.parquet", index=False)
    print("chapters:", len(texts), "words:", sum(t["n_words"] for t in texts))


if __name__ == "__main__":
    main()
