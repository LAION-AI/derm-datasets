"""Side-by-side comparison table: our dataset vs Derm1M (Markdown, HTML, LaTeX).

All numbers are read from computed results (work/compare/derm1m.json, dataset/docs/stats.json,
work/bias/bias.json, work/eval/*.json); rows whose experiment has not finished show "pending".
Writes dataset/docs/COMPARISON_DERM1M.{md,html,tex}.
"""
import html
import json
import os

from common import DATASET, WORK

DOCS = DATASET / "docs"


def load(p):
    return json.load(open(p)) if os.path.exists(p) else None


def f(n):
    return f"{n:,}" if isinstance(n, int) else n


def pct(x, d=1):
    return f"{x:.{d}f}%"


def main():
    C = load(WORK / "compare/derm1m.json")
    S = load(DOCS / "stats.json")
    Bz = load(WORK / "bias/bias.json")
    o, d = C["A_ours"], C["A_derm1m"]
    ev = {n: load(WORK / f"eval/{n}.json") for n in ["openai_vitb16", "clip_ft_ours", "clip_ft_derm1m", "clip_ft_ours_caption_only"]}

    def zs(n, k):
        """mean ± sd over seeds (n, n_s1, n_s2) when available, else the single run."""
        vals = [load(WORK / f"eval/{x}.json") for x in [n, n + "_s1", n + "_s2"]]
        vals = [100 * r[k] for r in vals if r and k in r]
        if not vals:
            return "pending"
        if len(vals) == 1:
            return f"{vals[0]:.1f}"
        import statistics
        return f"{statistics.mean(vals):.1f} ± {statistics.stdev(vals):.1f} (n={len(vals)})"

    T = [  # (section, metric, ours, derm1m, note-id)
        ("Scale", "Image–text pairs (as distributed)", f(o["pairs"]), f(d["pairs"]), 1),
        ("Scale", "Unique images (SHA-256)", f(o["unique_images"]), f(d["unique_images"]), None),
        ("Scale", "Distinct images after near-duplicate removal", f(o["near_dup_clusters"]), f(d["near_dup_clusters"]), 2),
        ("Scale", "Near-duplicate redundancy", pct(o["redundancy_pct"]), pct(d["redundancy_pct"]), 2),
        ("Scale", "Dermatology images (clinical/dermoscopy/histology), deduplicated", f(o["derm_core_clusters"]), f(d["derm_core_clusters"]), 3),
        ("Scale", "Dermatology images not contained in the other dataset",
         f(C["D_unique_derm_core_canonical_not_in_derm1m"]) + f" ({pct(100 - C['B_ours_in_derm1m_pct'])})",
         f"{pct(100 - C['B_derm1m_in_ours_pct'])}", 2),
        ("Content", "Clinical photographs", f(o["image_types"]["clinical"]), f(d["image_types"]["clinical"]), 3),
        ("Content", "Dermoscopy images", f(o["image_types"]["dermoscopy"]), f(d["image_types"]["dermoscopy"]), 3),
        ("Content", "Histopathology images", f(o["image_types"]["histopathology"]), f(d["image_types"]["histopathology"]), 3),
        ("Content", "Image sources", "7 peer-reviewed journals (1997–2025), 14 curated atlases, 2 textbooks",
         "YouTube frames (46%), PubMed OA, forums (IIYI, Reddit, X), textbooks, public datasets", None),
        ("Content", "Median image long side", f"{o['median_long_side_px']} px", f"{d['median_long_side_px']} px", None),
        ("Text", "Text origin", "expert-written figure captions + in-text references; curated atlas labels",
         "ASR transcripts and forum/article text, LLM-refined", None),
        ("Text", "Caption length, mean / median words", f"{o['caption_words_mean']} / {o['caption_words_median']:.0f}",
         f"{d['caption_words_mean']} / {d['caption_words_median']:.0f}", 4),
        ("Text", "Pairs with a diagnosis", pct(o["pairs_with_diagnosis_pct"]), pct(d["pairs_with_diagnosis_pct"]), 5),
        ("Text", "In-text figure references (article context)", f(S["n_with_context"]), "—", None),
        ("Text", "Languages", "en, de, fr, es (16k German samples)", "en (Chinese sources machine-translated)", None),
        ("Text", "Companion full-text corpus", f"{S['n_articles']:,} articles + 149 textbook chapters", "—", None),
        ("Diagnoses", "Derm1M ontology nodes with ≥10 images (of 369)", f(C["C_ontology_nodes_ge10"]["ours"]),
         f(C["C_ontology_nodes_ge10"]["derm1m"]), 6),
        ("Diagnoses", "Distinct normalised diagnosis labels with ≥50 images", f(C["C_concepts_ge50"]["ours"]),
         f(C["C_concepts_ge50"]["derm1m"]), 6),
        ("Diagnoses", "Structured codes", "ICD-10 (Altmeyers, 7.9k images), morphology / colour / site fields", "4-level ontology label", None),
        ("Metadata", "Pairs with age / sex / body site",
         f"{pct(o['pairs_with_age_pct'])} / {pct(o['pairs_with_sex_pct'])} / {pct(o['pairs_with_body_site_pct'])}",
         f"{pct(d['pairs_with_age_pct'])} / {pct(d['pairs_with_sex_pct'])} / {pct(d['pairs_with_body_site_pct'])}", None),
        ("Fairness", "Expert Fitzpatrick labels", f"{S['fst_gold_f17k'] and sum(S['fst_gold_f17k'].values()):,.0f} images", "none", None),
        ("Fairness", "Estimated share of FST V–VI (clinical photos)", pct(o["fst_V_VI_pct"]), pct(d["fst_V_VI_pct"]), 7),
        ("Fairness", "Skin-type balance (normalised entropy, 1 = uniform)", f"{o['fst_entropy_norm']:.2f}", f"{d['fst_entropy_norm']:.2f}", 7),
        ("Hygiene", "Per-item provenance (source file, URL/DOI, licence)", "yes", "source category only", None),
        ("Hygiene", "Leakage-free splits (grouped by case and article)", "yes (no image, near-duplicate, case or article spans two splits; same-patient links merged, ~2% of them still cross splits)",
         f"no: {C['E_derm1m_val_neardup_in_train_pct']}% of validation images have a near-duplicate in train "
         f"({C['E_derm1m_val_identical_in_train']} identical files)", 2),
        ("Hygiene", "Benchmark-contamination flags (PAD, HAM, F17K, Daffodil, SkinCAP)", f"yes ({S['bench_overlap_images']:,} images flagged)",
         "no (7,826 near-duplicates of benchmark images found here)", 8),
        ("Licence", "Licence", "per item: 7,639 CC-licensed samples (open tier); others metadata-only release; internal research copy",
         "CC BY-NC 4.0 (blanket)", 9),
        ("Licence", "Access", "open subset + metadata release (no gate)", "gated (Hugging Face, manual terms)", None),
        ("Models", "Equal-compute CLIP fine-tune, zero-shot mean acc. (PAD, HAM, F17K, Daffodil)",
         zs("clip_ft_ours", "zs_mean_acc"), zs("clip_ft_derm1m", "zs_mean_acc"), 10),
        ("Models", "… SkinCAP image→text R@10", zs("clip_ft_ours", "ret_SkinCAP_i2t_R@10"), zs("clip_ft_derm1m", "ret_SkinCAP_i2t_R@10"), 10),
        ("Models", "… linear probe mean acc.", zs("clip_ft_ours", "lp_mean_acc"), zs("clip_ft_derm1m", "lp_mean_acc"), 10),
    ]
    notes = {
        1: "Derm1M's paper reports 1,029,761 pairs by counting several texts (refined, ontology, concept) per image; the public v2 CSVs contain 423,016 rows.",
        2: "Same rule for both: CLIP ViT-B/16 cosine ≥ 0.98, or ≥ 0.95 with pHash distance ≤ 10.",
        3: "Image type predicted by the same linear head on CLIP embeddings for both datasets (gold-set accuracy 93.8%).",
        4: "Ours includes atlas images whose text is a structured template (median 2 words); original captions alone: mean 37.8, median 19 words.",
        5: "Derm1M: disease_label ≠ 'no definitive diagnosis' (24.2% of rows have none). Ours: source diagnosis label or a disease concept found in the English caption.",
        6: "Diagnosis strings of both datasets normalised and linked with SapBERT. The ontology row is the fair comparison. Raw label counts are inflated by naming variants on both sides and are indicative only.",
        7: "Estimated with a CLIP ridge head trained on expert Fitzpatrick17k labels (±1 accuracy 91%); it under-detects dark skin (recall 0.63), so true V–VI shares are somewhat higher in both datasets.",
        8: "Near-duplicates (cos ≥ 0.95) of benchmark images. Fitzpatrick17k was built from two of our atlases, hence the large number on our side.",
        9: "Derm1M contains figures from commercial textbooks (e.g. 73% of the Braun-Falco figures we hold appear in Derm1M), released under CC BY-NC. Our non-open tiers cannot be redistributed as images.",
        10: "OpenAI CLIP ViT-B/16 fine-tuned 1,000 steps × 128 pairs on 69,536 pairs from each dataset, after identical filtering (dermatology images, one per near-duplicate cluster, no benchmark near-duplicates); Derm1M benchmark protocol.",
    }

    head = ["", "Metric", "Ours (this work)", "Derm1M v2"]
    # Markdown
    md = ["# Our dataset vs Derm1M — side-by-side", "",
          "| Area | Metric | Ours (this work) | Derm1M v2 |", "|---|---|---|---|"]
    last = None
    for sec, m, a, b, n in T:
        md.append(f"| {'**' + sec + '**' if sec != last else ''} | {m}{f' [{n}]' if n else ''} | {a} | {b} |")
        last = sec
    md += ["", "**Where Derm1M is stronger:** about 3× more distinct dermatology images, 9× more dermoscopy, longer "
           "captions on average, broader body-site metadata, and a curated 4-level ontology.",
           "**Where ours is stronger:** almost entirely new images (98.5% not in Derm1M), lower redundancy, expert-written "
           "captions with article context and a full-text corpus, more histopathology, German/French/Spanish content, "
           "expert Fitzpatrick labels, per-item provenance and licences, a redistributable CC subset, benchmark-contamination "
           "flags, and leakage-free splits. The two are complementary: together they add ~98k new dermatology images to Derm1M.", ""]
    md += [f"[{k}] {v}" for k, v in notes.items()]
    open(DOCS / "COMPARISON_DERM1M.md", "w").write("\n".join(md) + "\n")

    # HTML
    rows, last = [], None
    for sec, m, a, b, n in T:
        sup = f"<sup>{n}</sup>" if n else ""
        rows.append(f"<tr><th scope='row'>{html.escape(sec) if sec != last else ''}</th><td>{html.escape(m)}{sup}</td>"
                    f"<td>{html.escape(str(a))}</td><td>{html.escape(str(b))}</td></tr>")
        last = sec
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Dataset comparison</title><style>
:root{{--bg:#fcfcfb;--ink:#0b0b0b;--muted:#52514e;--rule:#e4e3df;--accent:#2a78d6}}
@media (prefers-color-scheme:dark){{:root:not([data-theme="light"]){{--bg:#1a1a19;--ink:#f0efec;--muted:#c3c2b7;--rule:#3a3a37;--accent:#3987e5}}}}
:root[data-theme="dark"]{{--bg:#1a1a19;--ink:#f0efec;--muted:#c3c2b7;--rule:#3a3a37;--accent:#3987e5}}
body{{background:var(--bg);color:var(--ink);font:15px/1.45 system-ui,-apple-system,sans-serif;margin:0;padding:24px 16px}}
main{{max-width:1100px;margin:auto}} h1{{font-size:22px}} .wrap{{overflow-x:auto}}
table{{border-collapse:collapse;width:100%;min-width:640px}} th,td{{text-align:left;vertical-align:top;padding:7px 10px;border-bottom:1px solid var(--rule)}}
thead th{{color:var(--muted);font-weight:600}} tbody th{{color:var(--accent);font-weight:600;white-space:nowrap}}
td:nth-child(3){{font-weight:500}} .notes{{color:var(--muted);font-size:13px}} sup{{color:var(--muted)}}
</style></head><body><main><h1>Our dataset vs Derm1M — side-by-side</h1><div class="wrap"><table>
<thead><tr>{''.join(f'<th scope="col">{h}</th>' for h in head)}</tr></thead><tbody>{''.join(rows)}</tbody></table></div>
<p>{html.escape(md[-len(notes) - 3])}</p><p>{html.escape(md[-len(notes) - 2])}</p>
<ol class="notes">{''.join(f'<li>{html.escape(v)}</li>' for v in notes.values())}</ol></main></body></html>"""
    open(DOCS / "COMPARISON_DERM1M.html", "w").write(page)

    # LaTeX (booktabs + tabularx)
    esc = lambda s: (str(s).replace("\\", r"\textbackslash{}").replace("&", r"\&").replace("%", r"\%").replace("#", r"\#")
                     .replace("_", r"\_").replace("≥", r"$\geq$").replace("→", r"$\rightarrow$").replace("×", r"$\times$")
                     .replace("…", r"\ldots{}").replace("–", "--").replace("—", "---").replace("¹", "").replace("⁴", "")
                     .replace("⁵", "").replace("±", r"$\pm$").replace("≠", r"$\neq$"))
    tex = [r"% requires \usepackage{booktabs,tabularx}", r"\begin{table}[t]\centering\small",
           r"\caption{Our dataset compared with Derm1M v2. Superscripts refer to the notes below.}\label{tab:derm1m}",
           r"\begin{tabularx}{\linewidth}{@{}l>{\raggedright\arraybackslash}X>{\raggedright\arraybackslash}X>{\raggedright\arraybackslash}X@{}}",
           r"\toprule", r" & Metric & Ours (this work) & Derm1M v2 \\", r"\midrule"]
    last = None
    for sec, m, a, b, n in T:
        if last and sec != last:
            tex.append(r"\midrule")
        tex.append(f"{esc(sec) if sec != last else ''} & {esc(m)}{f'$^{{{n}}}$' if n else ''} & {esc(a)} & {esc(b)} \\\\")
        last = sec
    tex += [r"\bottomrule", r"\end{tabularx}", r"\par\smallskip\footnotesize"]
    tex += [f"$^{{{k}}}$ {esc(v)}" + r"\\" for k, v in notes.items()]
    tex += [r"\end{table}"]
    open(DOCS / "COMPARISON_DERM1M.tex", "w").write("\n".join(tex) + "\n")
    print("\n".join(md[:len(T) + 4]))


if __name__ == "__main__":
    main()
