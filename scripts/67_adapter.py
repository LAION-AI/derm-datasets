"""Frozen-backbone contrastive adapters: ours vs Derm1M for several backbones (OpenAI CLIP ViT-B/16,
MedSigLIP-448, EmbeddingGemma-2), at equal data (the 69,536 training pairs per dataset of the CLIP
fine-tuning experiment) and equal compute.

Adapter: g(x) = normalise(x + x A), A zero-initialised (identity at start), separately for image and
text embeddings, plus a learnable temperature; symmetric InfoNCE, batch 1024, AdamW, early stopping
on a 5% held-out split. Conditions: identity (no adapter), ours, derm1m, both (40k); 3 seeds each.
Evaluation as 60_eval_clip.py: zero-shot (PAD/HAM/F17K/Daffodil), linear probe (all four),
SkinCAP retrieval, on the backbone's benchmark embeddings (work/backbone/<name>/bench_img.npy).

Usage: python 67_adapter.py openai_vitb16 medsiglip_448 embeddinggemma2
Writes work/eval/adapter_<backbone>.json
"""
import json
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, top_k_accuracy_score

from common import WORK


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


class Adapter(torch.nn.Module):
    def __init__(self, d):
        super().__init__()
        self.Ai = torch.nn.Parameter(torch.zeros(d, d))
        self.At = torch.nn.Parameter(torch.zeros(d, d))
        self.logit_scale = torch.nn.Parameter(torch.tensor(np.log(1 / 0.07), dtype=torch.float32))

    def img(self, x):
        return F.normalize(x + x @ self.Ai, dim=-1)

    def txt(self, x):
        return F.normalize(x + x @ self.At, dim=-1)


def train(I, T, seed, epochs=30, bs=1024, lr=1e-3, wd=0.05):
    torch.manual_seed(seed)
    n = len(I)
    perm = np.random.default_rng(seed).permutation(n)
    va, tr = perm[: n // 20], perm[n // 20:]
    I, T = torch.tensor(I, dtype=torch.float32), torch.tensor(T, dtype=torch.float32)
    m = Adapter(I.shape[1])
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=wd)

    def loss_on(idx):
        zi, zt = m.img(I[idx]), m.txt(T[idx])
        lg = m.logit_scale.exp().clamp(max=100) * zi @ zt.T
        y = torch.arange(len(idx))
        return (F.cross_entropy(lg, y) + F.cross_entropy(lg.T, y)) / 2

    best, state, bad = 1e9, None, 0
    for ep in range(epochs):
        order = tr[np.random.default_rng(seed * 100 + ep).permutation(len(tr))]
        for s in range(0, len(order), bs):
            loss = loss_on(torch.tensor(order[s:s + bs]))
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            vl = float(loss_on(torch.tensor(va)))
        if vl < best - 1e-4:
            best, state, bad = vl, {k: v.clone() for k, v in m.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= 3:
                break
    m.load_state_dict(state)
    return m, best


def evaluate(m, feats, prompts, B, sc_txt, sc_paths):
    paths, Fb = feats["paths"], torch.tensor(feats["F"].astype(np.float32))
    with torch.no_grad():
        Fa = (m.img(Fb) if m else F.normalize(Fb, dim=-1)).numpy()
    pos = {p: i for i, p in enumerate(paths)}
    res = {}
    from importlib import import_module
    classes = import_module("60_eval_clip").CLASSES
    for bench in ["PAD", "HAM", "F17K", "Daffodil"]:
        raw = torch.tensor(prompts[bench + "_raw"].astype(np.float32))           # (C, 8, D)
        with torch.no_grad():
            W = (m.txt(raw) if m else F.normalize(raw, dim=-1)).mean(1)
            W = F.normalize(W, dim=-1).numpy().T
        d = B[(B.bench == bench) & (B.task == "ZS")]
        X = Fa[[pos[p] for p in d.path]]
        y = d.label.values
        p = softmax(100 * X @ W)
        if bench == "F17K":
            res["zs_F17K_top1"] = accuracy_score(y, p.argmax(1))
            res["zs_F17K_top5"] = top_k_accuracy_score(y, p, k=5, labels=np.arange(len(classes[bench])))
        else:
            res[f"zs_{bench}_acc"] = accuracy_score(y, p.argmax(1))
            res[f"zs_{bench}_auroc"] = roc_auc_score(np.eye(len(classes[bench]))[y], p, multi_class="ovr", average="macro")
    res["zs_mean_acc"] = float(np.mean([res["zs_PAD_acc"], res["zs_HAM_acc"], res["zs_F17K_top1"], res["zs_Daffodil_acc"]]))
    for bench in ["PAD", "HAM", "F17K", "Daffodil"]:
        d = B[(B.bench == bench) & (B.task == "LP")]
        sp = {s: (Fa[[pos[p] for p in d[d.split == s].path]], d[d.split == s].label.values) for s in ["train", "val", "test"]}
        best = max(((accuracy_score(sp["val"][1], LogisticRegression(C=C, max_iter=3000).fit(*sp["train"]).predict(sp["val"][0])), C)
                    for C in [0.3, 1, 3, 10, 30, 100]))
        mdl = LogisticRegression(C=best[1], max_iter=3000).fit(*sp["train"])
        pr = mdl.predict(sp["test"][0])
        res[f"lp_{bench}_acc"] = accuracy_score(sp["test"][1], pr)
        res[f"lp_{bench}_f1"] = f1_score(sp["test"][1], pr, average="macro")
    res["lp_mean_acc"] = float(np.mean([res[f"lp_{b}_acc"] for b in ["PAD", "HAM", "F17K", "Daffodil"]]))
    I = Fa[[pos[p] for p in sc_paths]]
    with torch.no_grad():
        Tt = (m.txt(torch.tensor(sc_txt.astype(np.float32))) if m else F.normalize(torch.tensor(sc_txt.astype(np.float32)), dim=-1)).numpy()
    S = I @ Tt.T
    gt = np.arange(len(I))
    for k in (1, 10):
        res[f"ret_i2t_R@{k}"] = float((np.argsort(-S, 1)[:, :k] == gt[:, None]).any(1).mean())
        res[f"ret_t2i_R@{k}"] = float((np.argsort(-S.T, 1)[:, :k] == gt[:, None]).any(1).mean())
    return res


def main():
    B = pd.read_parquet(WORK / "bench.parquet")
    for name in sys.argv[1:]:
        d = WORK / "backbone" / name
        bi = pd.read_parquet(d / "bench_index.parquet")
        feats = {"paths": bi.path.values, "F": np.load(d / "bench_img.npy")}
        prompts = np.load(d / "prompts.npz")
        sc_txt = np.load(d / "skincap_txt.npy")
        sc_paths = pd.read_parquet(d / "skincap_index.parquet").path.tolist()
        data = {ds: (np.load(d / f"{ds}_pairs_img.npy").astype(np.float32), np.load(d / f"{ds}_pairs_txt.npy").astype(np.float32))
                for ds in ["ours", "derm1m"]}
        data["both"] = tuple(np.concatenate([data["ours"][i], data["derm1m"][i]]) for i in (0, 1))
        out = {"backbone": name, "identity": evaluate(None, feats, prompts, B, sc_txt, sc_paths)}
        print(name, "identity", round(out["identity"]["zs_mean_acc"], 4), flush=True)
        for ds in ["ours", "derm1m", "both"]:
            runs = []
            for seed in range(3):
                m, vl = train(*data[ds], seed)
                r = evaluate(m, feats, prompts, B, sc_txt, sc_paths)
                r["val_loss"] = vl
                runs.append(r)
                print(name, ds, seed, round(r["zs_mean_acc"], 4), round(r["ret_i2t_R@10"], 4), flush=True)
            keys = runs[0].keys()
            out[ds] = {k: {"mean": float(np.mean([r[k] for r in runs])), "std": float(np.std([r[k] for r in runs]))} for k in keys}
        json.dump(out, open(WORK / f"eval/adapter_{name}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
