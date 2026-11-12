"""Report figures (static, print). Palette: validated reference categorical slots 1-3
(#2a78d6, #eb6834, #1baf7a; all-pairs CVD-safe), values directly labelled (aqua is below 3:1 on white).
Writes report/figures/*.png (300 dpi) and *.csv with the plotted numbers (table view).
"""
import json

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import DATASET, WORK

matplotlib.use("Agg")
OUT = WORK.parent / "report/figures"
OUT.mkdir(parents=True, exist_ok=True)
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": MUTED,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlecolor": INK, "figure.facecolor": SURF,
                     "axes.facecolor": SURF, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlelocation": "left"})


def save(fig, name, table):
    fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight")
    table.to_csv(OUT / f"{name}.csv")
    plt.close(fig)


def fig_composition():
    S = pd.read_parquet(DATASET / "metadata/samples.parquet")
    names = {"jaad": "JAAD", "jddg": "JDDG", "die-dermatologie": "Die Dermatologie", "jid": "JID", "ajcd": "AJCD",
             "jama": "JAMA Dermatol.", "dat": "Dermatol. Ther.", "dermaamin": "Dermaamin", "dermnet": "DermNet NZ",
             "atlasdermatologico": "Atlas Dermatológico", "altmeyers": "Altmeyers", "dermis": "DermIS",
             "hellenicdermatlas": "Hellenic Atlas", "iconotheque": "Iconothèque", "dermatlas": "Interact. Derm. Atlas",
             "danderm": "DanDerm", "braun-falco": "Braun-Falco", "fritsch": "Fritsch"}
    t = S.groupby(["source_type", "source"]).size().rename("samples").reset_index()
    t = t[t.samples >= 500].sort_values("samples")
    col = {"journal": C1, "atlas": C2, "textbook": C3}
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    y = np.arange(len(t))
    ax.barh(y, t.samples, color=[col[s] for s in t.source_type], height=0.72, edgecolor=SURF, linewidth=2)
    ax.set_yticks(y, [names.get(s, s) for s in t.source])
    for i, v in enumerate(t.samples):
        ax.text(v + 300, i, f"{v:,}", va="center", fontsize=7.5, color=INK)
    ax.grid(axis="x", color=GRID, linewidth=0.6); ax.set_axisbelow(True)
    ax.set_xlabel("image–text samples")
    ax.set_title("Dataset composition by source (sources with ≥ 500 samples)")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=col[k], label=k) for k in ["journal", "atlas", "textbook"]], frameon=False, loc="lower right")
    save(fig, "fig1_composition", t.set_index("source"))


def fig_fst():
    bz = json.load(open(WORK / "bias/bias.json"))
    C = json.load(open(WORK / "compare/derm1m.json"))
    d = C["A_derm1m"]["fst_dist_clinical"]
    dd = {int(k): v for k, v in d.items()}
    tot = sum(dd.values())
    rows = pd.DataFrame({
        "Ours (estimated)": [bz["fst_ours_clinical"]["dist_pct"][str(k)] if str(k) in bz["fst_ours_clinical"]["dist_pct"]
                             else bz["fst_ours_clinical"]["dist_pct"][k] for k in range(1, 7)],
        "Derm1M (estimated)": [100 * dd.get(k, 0) / tot for k in range(1, 7)],
        "Fitzpatrick17k (expert)": [bz["fst_fitzpatrick17k_benchmark"]["dist_pct"][str(k)] if str(k) in bz["fst_fitzpatrick17k_benchmark"]["dist_pct"]
                                    else bz["fst_fitzpatrick17k_benchmark"]["dist_pct"][k] for k in range(1, 7)],
    }, index=["I", "II", "III", "IV", "V", "VI"])
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    x = np.arange(6); w = 0.26
    for i, (c, colr) in enumerate(zip(rows.columns, [C1, C2, C3])):
        b = ax.bar(x + (i - 1) * w, rows[c], w, color=colr, label=c, edgecolor=SURF, linewidth=1.5)
        for xi, v in zip(x + (i - 1) * w, rows[c]):
            ax.text(xi, v + 0.8, f"{v:.0f}", ha="center", fontsize=6.5, color=INK)
    ax.set_xticks(x, rows.index); ax.set_xlabel("Fitzpatrick skin type"); ax.set_ylabel("% of clinical photos")
    ax.grid(axis="y", color=GRID, linewidth=0.6); ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.2), fontsize=7.5)
    ax.set_ylim(0, rows.values.max() * 1.12)
    ax.set_title("Skin-type distribution of clinical photographs")
    save(fig, "fig2_fitzpatrick", rows)


def fig_types():
    C = json.load(open(WORK / "compare/derm1m.json"))
    keys = ["clinical", "dermoscopy", "histopathology", "microscopy_other", "radiology", "other_photo", "figure_nonphoto"]
    lab = ["clinical", "dermoscopy", "histopathology", "other microscopy", "radiology", "other photo", "non-photo figure"]
    o, d = C["A_ours"]["image_types"], C["A_derm1m"]["image_types"]
    t = pd.DataFrame({"Ours": [o.get(k, 0) for k in keys], "Derm1M": [d.get(k, 0) for k in keys]}, index=lab)
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    y = np.arange(len(t)); h = 0.38
    for i, (c, colr) in enumerate(zip(t.columns, [C1, C2])):
        ax.barh(y + (i - 0.5) * h, t[c], h, color=colr, label=c, edgecolor=SURF, linewidth=1.5)
        for yi, v in zip(y + (i - 0.5) * h, t[c]):
            ax.text(v * 1.08 if v > 0 else 1, yi, f"{v:,}", va="center", fontsize=6.5, color=INK)
    ax.set_xscale("log"); ax.set_yticks(y, t.index); ax.invert_yaxis()
    ax.set_xlabel("unique images (log scale)"); ax.grid(axis="x", color=GRID, linewidth=0.6); ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="lower right"); ax.set_title("Image types (same classifier applied to both datasets)")
    save(fig, "fig3_image_types", t)


def _seeds(name, key):
    xs = []
    for n in [name, name + "_s1", name + "_s2"]:
        f = WORK / f"eval/{n}.json"
        if f.exists() and isinstance(json.load(open(f)).get(key), (int, float)):
            xs.append(100 * json.load(open(f))[key])
    return (np.mean(xs), np.std(xs, ddof=1) if len(xs) > 1 else 0.0) if xs else (np.nan, 0.0)


def fig_clip():
    """Equal-compute CLIP fine-tuning (1k steps, mean ± SD over 3 seeds) and per-skin-type zero-shot accuracy."""
    models = [("OpenAI CLIP (no fine-tuning)", "openai_vitb16", "#a3a29e"), ("Fine-tuned on ours", "clip_ft_ours", C1),
              ("Fine-tuned on Derm1M", "clip_ft_derm1m", C2)]
    metrics = [("PAD\nAUROC", "zs_PAD_auroc"), ("HAM\nAUROC", "zs_HAM_auroc"), ("Daffodil\nAUROC", "zs_Daffodil_auroc"),
               ("F17K\ntop-1", "zs_F17K_top1"), ("SkinCAP\ni→t R@10", "ret_SkinCAP_i2t_R@10"),
               ("SkinCAP\nt→i R@10", "ret_SkinCAP_t2i_R@10"), ("Linear probe\nmean acc.", "lp_mean_acc")]
    groups = [("FST I–II", "zs_F17K_acc_fst_I-II"), ("FST III–IV", "zs_F17K_acc_fst_III-IV"), ("FST V–VI", "zs_F17K_acc_fst_V-VI")]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(9.6, 3.4), gridspec_kw={"width_ratios": [3.2, 1.3]})
    rows = {}
    for axx, mets, title in [(ax, metrics, "Zero-shot, retrieval and linear probe (Derm1M protocol)"), (bx, groups, "Fitzpatrick17k zero-shot by skin type")]:
        x = np.arange(len(mets)); w = 0.26
        for i, (lab, n, col) in enumerate(models):
            ms = [_seeds(n, k) for _, k in mets]
            rows.update({(lab, m[0].replace("\n", " ")): v for m, v in zip(mets, ms)})
            err = [e for _, e in ms]
            axx.bar(x + (i - 1) * w, [m for m, _ in ms], w, yerr=err if any(err) else None, color=col, label=lab,
                    edgecolor=SURF, linewidth=1.5, error_kw=dict(ecolor=INK, lw=0.8, capsize=2))
        axx.set_xticks(x, [m for m, _ in mets], fontsize=7.5)
        axx.grid(axis="y", color=GRID, linewidth=0.6); axx.set_axisbelow(True); axx.set_title(title)
    ax.set_ylabel("score (%)"); bx.set_ylabel("top-1 accuracy (%)")
    fig.text(0.01, -0.2, "Fine-tuning: 1,000 steps × 128 pairs, 69,536 pairs per dataset; bars = mean of 3 seeds, whiskers = SD.",
             fontsize=7, color=MUTED)
    ax.legend(frameon=False, ncol=3, loc="upper left", bbox_to_anchor=(0, -0.2), fontsize=8)
    t = pd.DataFrame([(k[0], k[1], v[0], v[1]) for k, v in rows.items()], columns=["model", "metric", "mean", "sd"]).set_index(["model", "metric"])
    save(fig, "fig4_clip_finetuning", t.round(2))


if __name__ == "__main__":
    fig_composition(); fig_fst(); fig_types(); fig_clip()
    print(sorted(p.name for p in OUT.iterdir()))
