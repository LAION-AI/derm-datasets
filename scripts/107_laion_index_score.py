"""Classify every image of the Re-LAION-2B-en kNN index (laion/relaion2B-en-research-safe-index) and build relaion-derm.

The index stores OpenAI CLIP ViT-L/14 image embeddings of 966,492,509 Re-LAION-2B-en images as
  IndexPreTransform( LinearTransform 768 -> 896, orthonormal: A^T A = I ) -> IndexIVFPQ( 131,072 lists, HNSW quantizer,
  PQ 128 x 8 bit, by_residual, inner product ),
merged from the four HF parts into external/relaion_index/image.index and read with faiss mmap; metadata.hdf5 (url,
caption, image_path; same order as the index ids) is read in place with the h5py family driver.
Checked on 400 already-downloaded Re-LAION images: embedded with ViT-L-14-quickgelu/openai, their nearest index entry
is the same image (median inner product 0.93); PQ reconstruction keeps cosine 0.978.

The head of 106 is linear (multinomial logistic), so its logits are computed exactly on the PQ reconstruction without
rebuilding vectors: with v_c = A w_c, logit_c = b_c + v_c . centroid(list) + sum_m v_c[m] . PQ_m[code_m], i.e. one
per-list term plus a 128 x 256 lookup table per class (numpy gather over all codes).

Commands (in order):
  sample  [--n 200000]   uniform random index entries, reconstructed to 768-d -> work/laion5b/index_random.npz
                         (web_other negatives for 106 in the quantised domain)
  score                  all 966M entries -> work/laion5b/index_scored.parquet (id, p_core, top class) for p_core >= 0.2,
                         index_score_hist.json (histogram over all entries), index_scored_sample.parquet (1 in 1000)
  nudity                 CLIP text-probe nudity screen (ViT-L/14 text tower, LUT pass) for the scored rows ->
                         index_nudity.parquet; flagged rows (p_exposed >= 0.15) are never downloaded, viewed or released
  meta                   url / caption for the scored rows -> index_scored_meta.parquet
  review  [--per-bin 96] stratified download per p_core bin -> review sheets for visual precision estimation
  quantize SRC.npy DST.npy  PQ round trip of embeddings (used by 106; faiss and sklearn/torch OpenMP runtimes clash
                         in one process)
  export  --thr 0.9      relaion-derm open tier: work/laion5b/relaion_derm.parquet (url, caption, p_core, bin precision
                         from the visual review, class, index id); nudity-flagged rows only in an internal file
"""
import argparse
import importlib
import json
import time

import faiss
import h5py
import numpy as np
import pandas as pd

from common import EXTERNAL, WORK

OUT = WORK / "laion5b"
IDX = EXTERNAL / "relaion_index" / "image.index"
META = str(EXTERNAL / "relaion_index" / "metadata.hdf5.part_%03d")
CORE = ["clinical", "dermoscopy", "histopathology"]
CH = 100_000
# Nudity screen as in 86_eval_synthetic.py (CLIP text probes), here with the ViT-L/14 text tower on the index vectors and a
# stricter threshold, because web images may show minors: flagged images are never downloaded, viewed or released.
NUDITY_FLAG = 0.15
NUDITY_LABELS = ["a clinical close-up photograph of skin on a face, arm, hand, leg, foot, back or chest",
                 "a photograph showing exposed genitals or pubic area", "a photograph showing bare female breasts",
                 "a photograph showing bare buttocks or the anal region"]
WORKERS = 8
BINS = [0.2, 0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 1.0001]


def load():
    idx = faiss.read_index(str(IDX), faiss.IO_FLAG_MMAP | faiss.IO_FLAG_READ_ONLY)
    pt = faiss.downcast_index(idx)
    vt = faiss.downcast_VectorTransform(pt.chain.at(0))
    A = faiss.vector_to_array(vt.A).reshape(vt.d_out, vt.d_in)
    ivf = faiss.downcast_index(faiss.extract_index_ivf(pt))
    il = faiss.downcast_InvertedLists(ivf.invlists)
    sizes = np.array([il.list_size(i) for i in range(ivf.nlist)], dtype=np.int64)
    return idx, A, ivf, il, sizes


def meta():
    return h5py.File(META, "r", driver="family", memb_size=42949672960)["dataset"]


def list_arrays(il, l, n, cs):
    codes = faiss.rev_swig_ptr(il.get_codes(l), n * cs).reshape(n, cs)
    ids = faiss.rev_swig_ptr(il.get_ids(l), n)
    return codes, ids


def sample(n):
    idx, A, ivf, il, sizes = load()
    rng = np.random.default_rng(0)
    pos = np.sort(rng.choice(int(sizes.sum()), n, replace=False))
    cum = np.cumsum(sizes)
    lists = np.searchsorted(cum, pos, side="right")
    offs = pos - (cum[lists] - sizes[lists])
    C = faiss.downcast_index(ivf.quantizer).reconstruct_n(0, ivf.nlist)
    codes, ids = np.empty((n, ivf.code_size), np.uint8), np.empty(n, np.int64)
    for l in np.unique(lists):
        m = lists == l
        c, i = list_arrays(il, int(l), int(sizes[l]), ivf.code_size)
        codes[m], ids[m] = c[offs[m]], i[offs[m]]
    Y = ivf.pq.decode(codes) + C[lists]
    X = (Y @ A).astype(np.float32)
    np.savez(OUT / "index_random.npz", X=X.astype(np.float16), ids=ids, lists=lists)
    print("sampled", n, "norm median", float(np.median(np.linalg.norm(X, axis=1))))


def quantize(src, dst):
    """PQ round trip (encode + decode through the index) of 768-d embeddings, to evaluate the head on quantised input."""
    idx = load()[0]
    np.save(dst, idx.sa_decode(idx.sa_encode(np.load(src).astype(np.float32))))


def tables(ivf, A, head):
    """Per-list logit term (nlist x C) and PQ lookup table (M*256 x C) of the linear head in index space."""
    W, b = head.coef_.astype(np.float64), head.intercept_.astype(np.float64)  # C x 768
    V = A.astype(np.float64) @ W.T  # 896 x C
    Cent = faiss.downcast_index(ivf.quantizer).reconstruct_n(0, ivf.nlist).astype(np.float64)
    T = Cent @ V + b  # nlist x C
    pq = ivf.pq
    PQ = faiss.vector_to_array(pq.centroids).reshape(pq.M, pq.ksub, pq.dsub).astype(np.float64)
    L = np.einsum("mkd,mdc->mkc", PQ, V.reshape(pq.M, pq.dsub, -1)).reshape(pq.M * pq.ksub, -1)
    return T.astype(np.float32), L.astype(np.float32)


_W = {}


def score_range(rng_):
    """Worker: p_core for all entries of lists [l0, l1)."""
    if not _W:
        idx, _, ivf, il, sizes = load()
        z = np.load(OUT / "index_tables.npz")
        _W.update(idx=idx, ivf=ivf, il=il, sizes=sizes, T=z["T"], L=z["L"], core=z["core"],
                  off=(np.arange(ivf.pq.M) * ivf.pq.ksub).astype(np.int32))
    ivf, il, sizes, T, L, core, off = (_W[k] for k in ["ivf", "il", "sizes", "T", "L", "core", "off"])
    l0, l1 = rng_
    rng = np.random.default_rng(int(l0))
    out = {k: [] for k in ["id", "p", "c", "sid", "sp"]}
    hist, n_all = np.zeros(1000, np.int64), 0
    for l in range(int(l0), int(l1)):
        n = int(sizes[l])
        if n == 0:
            continue
        codes, ids = list_arrays(il, l, n, ivf.code_size)
        for s in range(0, n, CH):
            z = L[codes[s:s + CH].astype(np.int32) + off].sum(1) + T[l]
            z = np.exp(z - z.max(1, keepdims=True))
            p = z / z.sum(1, keepdims=True)
            pc = p[:, core].sum(1)
            idb = ids[s:s + CH]
            hist += np.bincount(np.minimum((pc * 1000).astype(np.int64), 999), minlength=1000)
            m = pc >= 0.2
            out["id"].append(idb[m].copy()); out["p"].append(pc[m].astype(np.float32))
            out["c"].append(p[m].argmax(1).astype(np.int8))
            r = rng.random(len(pc)) < 0.001
            out["sid"].append(idb[r].copy()); out["sp"].append(pc[r].astype(np.float32))
        n_all += n
    res = {k: np.concatenate(v) if v else np.zeros(0) for k, v in out.items()}
    res["c"] = res["c"].astype(np.int8)
    res.update(hist=hist, n=n_all)
    return res


def score():
    import joblib
    idx, A, ivf, il, sizes = load()
    head = joblib.load(OUT / "head_vitl14.joblib")
    classes = list(head.classes_)
    core = [classes.index(c) for c in CORE]
    T, L = tables(ivf, A, head)
    off = (np.arange(ivf.pq.M) * ivf.pq.ksub).astype(np.int32)
    # self-check on the random sample: LUT logits == head on reconstructed vectors
    if (OUT / "index_random.npz").exists():
        z = np.load(OUT / "index_random.npz")
        l0 = int(z["lists"][0])
        c, i = list_arrays(il, l0, int(sizes[l0]), ivf.code_size)
        j = int(np.where(i == z["ids"][0])[0][0])
        lut = T[l0] + L[c[j].astype(np.int32) + off].sum(0)
        ref = head.decision_function(z["X"][:1].astype(np.float32))[0]
        print("LUT check max |diff|", float(np.abs(lut - ref).max()))
    np.savez(OUT / "index_tables.npz", T=T, L=L, core=np.array(core))
    cum = np.cumsum(sizes)
    cuts = np.unique(np.searchsorted(cum, np.linspace(0, cum[-1], 513)[1:-1]))
    ranges = list(zip(np.r_[0, cuts], np.r_[cuts, ivf.nlist]))
    keep_id, keep_p, keep_c, samp_id, samp_p = [], [], [], [], []
    hist = np.zeros(1000, np.int64)
    t0, done = time.time(), 0
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(WORKERS) as ex:
        for k, r in enumerate(ex.map(score_range, ranges)):
            hist += r["hist"]
            keep_id.append(r["id"]); keep_p.append(r["p"]); keep_c.append(r["c"])
            samp_id.append(r["sid"]); samp_p.append(r["sp"])
            done += r["n"]
            if k % 16 == 0:
                el = time.time() - t0
                print(f"range {k}/{len(ranges)} {done:,} vectors {done / el:,.0f}/s kept {sum(map(len, keep_id)):,}",
                      flush=True)
    df = pd.DataFrame({"id": np.concatenate(keep_id), "p_core": np.concatenate(keep_p),
                       "top": pd.Categorical.from_codes(np.concatenate(keep_c), classes)})
    df.sort_values("id").to_parquet(OUT / "index_scored.parquet", index=False)
    pd.DataFrame({"id": np.concatenate(samp_id), "p_core": np.concatenate(samp_p)}).to_parquet(
        OUT / "index_scored_sample.parquet", index=False)
    rep = {"n": int(hist.sum()), "hist_per_mille": hist.tolist(),
           "n_ge": {str(t): int(hist[int(round(t * 1000)):].sum()) for t in [0.2, 0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99]},
           "top_class_counts_kept": df.top.value_counts().to_dict(), "seconds": round(time.time() - t0)}
    json.dump(rep, open(OUT / "index_score_hist.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in rep.items() if k != "hist_per_mille"}, indent=1))


TEXT_EMB = """
import sys, json, numpy as np, open_clip, torch
m, _, _ = open_clip.create_model_and_transforms("ViT-L-14-quickgelu", pretrained="openai")
with torch.no_grad():
    t = m.encode_text(open_clip.get_tokenizer("ViT-L-14-quickgelu")(json.loads(sys.argv[2]))).float()
np.save(sys.argv[1], torch.nn.functional.normalize(t, dim=-1).numpy())
"""  # own process without faiss: the torch and faiss OpenMP runtimes clash


_N = {}


def nudity_range(rng_):
    if not _N:
        idx, _, ivf, il, sizes = load()
        z = np.load(OUT / "nudity_tables.npz")
        _N.update(idx=idx, ivf=ivf, il=il, sizes=sizes, T=z["T"], L=z["L"], keep=np.load(OUT / "index_scored_ids.npy"),
                  off=(np.arange(ivf.pq.M) * ivf.pq.ksub).astype(np.int32))
    ivf, il, sizes, T, L, keep, off = (_N[k] for k in ["ivf", "il", "sizes", "T", "L", "keep", "off"])
    ids_out, p_out = [], []
    for l in range(int(rng_[0]), int(rng_[1])):
        n = int(sizes[l])
        if n == 0:
            continue
        codes, ids = list_arrays(il, l, n, ivf.code_size)
        m = keep[np.minimum(np.searchsorted(keep, ids), len(keep) - 1)] == ids  # keep is sorted
        if not m.any():
            continue
        z = L[codes[m].astype(np.int32) + off].sum(1) + T[l]
        z = np.exp(z - z.max(1, keepdims=True))
        ids_out.append(ids[m].copy()); p_out.append((z[:, 1:].sum(1) / z.sum(1)).astype(np.float32))
    return (np.concatenate(ids_out), np.concatenate(p_out)) if ids_out else (np.zeros(0, np.int64), np.zeros(0, np.float32))


def nudity():
    import subprocess
    import sys
    src = OUT / "nudity_text_vitl14.npy"
    subprocess.run([sys.executable, "-c", TEXT_EMB, str(src), json.dumps(NUDITY_LABELS)], check=True)
    t = np.load(src).astype(np.float64)
    idx, A, ivf, il, sizes = load()

    class Probe:  # CLIP zero-shot logits 100 * <x, t_k> as a linear head
        coef_, intercept_ = 100 * t, np.zeros(len(t))
    T, L = tables(ivf, A, Probe)
    np.savez(OUT / "nudity_tables.npz", T=T, L=L)
    np.save(OUT / "index_scored_ids.npy", pd.read_parquet(OUT / "index_scored.parquet", columns=["id"]).id.values)
    cum = np.cumsum(sizes)
    cuts = np.unique(np.searchsorted(cum, np.linspace(0, cum[-1], 513)[1:-1]))
    ranges = list(zip(np.r_[0, cuts], np.r_[cuts, ivf.nlist]))
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(WORKERS) as ex:
        res = list(ex.map(nudity_range, ranges))
    df = pd.DataFrame({"id": np.concatenate([r[0] for r in res]), "p_exposed": np.concatenate([r[1] for r in res])})
    df.sort_values("id").to_parquet(OUT / "index_nudity.parquet", index=False)
    print("nudity screened", len(df), "flagged", int((df.p_exposed >= NUDITY_FLAG).sum()))


def attach_meta(ids):
    """url, caption, image_path for sorted index ids, read block-wise from metadata.hdf5."""
    d = meta()
    ids = np.asarray(ids)
    out = {k: np.empty(len(ids), dtype=object) for k in ["url", "caption", "image_path"]}
    B = 2952 * 400
    for s in range(0, d["url"].shape[0], B):
        lo, hi = np.searchsorted(ids, [s, s + B])
        if lo == hi:
            continue
        for k in out:
            blk = d[k][s:min(s + B, d[k].shape[0])]
            out[k][lo:hi] = [x.decode("utf-8", "replace") for x in blk[ids[lo:hi] - s]]
    return pd.DataFrame(out)


def meta_cmd():
    df = pd.read_parquet(OUT / "index_scored.parquet")
    m = attach_meta(df.id.values)
    pd.concat([df.reset_index(drop=True), m], axis=1).to_parquet(OUT / "index_scored_meta.parquet", index=False)
    print("meta", len(df))


def review(per_bin):
    f71 = importlib.import_module("71_laion_filter")
    from concurrent.futures import ThreadPoolExecutor
    from PIL import Image, ImageDraw
    df = pd.read_parquet(OUT / "index_scored_meta.parquet")
    nud = pd.read_parquet(OUT / "index_nudity.parquet")  # required: flagged images are never downloaded or viewed
    df = df.merge(nud, on="id", how="left")
    assert df.p_exposed.notna().all()
    print("excluded unseen by nudity screen:", int((df.p_exposed >= NUDITY_FLAG).sum()))
    df = df[df.p_exposed < NUDITY_FLAG]
    df["bin"] = pd.cut(df.p_core, BINS, right=False, labels=[f"{a:.2f}-{b:.2f}" for a, b in zip(BINS, BINS[1:])])
    df = df.sample(frac=1, random_state=0)
    s = df[df.groupby("bin", observed=True).cumcount() < per_bin].sort_values("p_core").reset_index(drop=True)
    img = EXTERNAL / "relaion_index_review"
    img.mkdir(exist_ok=True)
    s["path"] = [str(img / f"{i}.jpg") for i in s.id]
    with ThreadPoolExecutor(64) as ex:
        st = dict(ex.map(lambda r: f71.fetch((r.id, r.url, r.path)), s.itertuples()))
    s["status"] = s.id.map(st)
    s = s[s.status == "ok"].reset_index(drop=True)
    s["rid"] = np.arange(len(s))
    sheets = OUT / "index_sheets"
    sheets.mkdir(exist_ok=True)
    for k, g in s.groupby(np.arange(len(s)) // 48):
        sheet = Image.new("RGB", (8 * 200, 6 * 220), "white")
        dr = ImageDraw.Draw(sheet)
        for j, r in enumerate(g.itertuples()):
            try:
                with Image.open(r.path) as im:
                    im = im.convert("RGB")
                    im.thumbnail((196, 196))
                    sheet.paste(im, ((j % 8) * 200 + 2, (j // 8) * 220 + 2))
            except Exception:
                pass
            dr.text(((j % 8) * 200 + 4, (j // 8) * 220 + 202), f"#{r.rid}", fill="black")
        sheet.save(sheets / f"sheet_{k:03d}.jpg", quality=85)
    s.drop(columns=["caption"]).to_parquet(OUT / "index_review.parquet", index=False)
    print(s.groupby("bin", observed=True).size())


def export(thr):
    """relaion-derm open tier (p_core >= thr, not nudity-flagged) + internal nudity-review list (never released)."""
    df = pd.read_parquet(OUT / "index_scored_meta.parquet")
    d = df[df.p_core >= thr].rename(columns={"id": "index_id", "top": "image_type"})
    nud = pd.read_parquet(OUT / "index_nudity.parquet").rename(columns={"id": "index_id"})
    d = d.merge(nud, on="index_id", how="left")
    d["nudity_flag"] = d.p_exposed >= NUDITY_FLAG
    prec = json.load(open(OUT / "index_precision.json"))["bins"]
    edges = [float(b["bin"].strip("[)").split(",")[0]) for b in prec]
    d["bin_precision"] = np.array([b["precision"] for b in prec])[np.searchsorted(edges, d.p_core, side="right") - 1]
    cols = ["url", "caption", "p_core", "bin_precision", "image_type", "index_id", "image_path"]
    d[~d.nudity_flag][cols].sort_values("p_core", ascending=False).to_parquet(OUT / "relaion_derm.parquet", index=False)
    d[d.nudity_flag][cols + ["p_exposed"]].to_parquet(OUT / "relaion_derm_nudity_review_INTERNAL.parquet", index=False)
    o = d[~d.nudity_flag]
    print("relaion-derm open", len(o), "expected true", int(o.bin_precision.sum()), o.image_type.value_counts().to_dict(),
          "| nudity review (internal)", int(d.nudity_flag.sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["sample", "score", "nudity", "meta", "review", "export", "quantize"])
    ap.add_argument("src", nargs="?")
    ap.add_argument("dst", nargs="?")
    ap.add_argument("--n", type=int, default=200_000)
    ap.add_argument("--per-bin", type=int, default=96)
    ap.add_argument("--thr", type=float, default=0.9)
    a = ap.parse_args()
    {"sample": lambda: sample(a.n), "score": score, "meta": meta_cmd, "review": lambda: review(a.per_bin),
     "export": lambda: export(a.thr),
     "quantize": lambda: quantize(a.src, a.dst), "nudity": nudity}[a.cmd]()


if __name__ == "__main__":
    main()
