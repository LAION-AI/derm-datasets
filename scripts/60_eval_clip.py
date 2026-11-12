"""Evaluate a CLIP model on the Derm1M downstream protocol.

Zero-shot (Derm1M protocol, identical class names / 8 templates / metrics):
  PAD-UFES-20, HAM10000, Daffodil: macro OvR AUROC + accuracy; F17K (113+1 classes): top-1/top-5.
Linear probe: L2-regularised logistic regression on frozen image features,
  C selected on the LP val split, accuracy + macro-F1 on the LP test split.
Retrieval: SkinCAP (3,989 images, English captions) image->text / text->image R@1/5/10.

Usage: python 60_eval_clip.py --name <label> --model ViT-B-16-quickgelu --pretrained openai|<ckpt.pt>
Writes work/eval/<label>.json
"""
import argparse
import json
import os
import warnings

import numpy as np
import open_clip
import pandas as pd
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, top_k_accuracy_score
from torch.utils.data import DataLoader, Dataset

from common import WORK, device

warnings.filterwarnings("ignore")
META = json.load(open(os.path.join(os.path.dirname(__file__), "derm1m_zeroshot_meta.json")))
TEMPLATES = [t[2:-1] for t in META["TEMPLATES"]]   # "f'...{c}...'" -> "...{c}..."
CLASSES = {"PAD": META["PAD_CLASSNAMES"], "HAM": META["HAM_CLASSNAMES"],
           "Daffodil": META["DAFFODIL_5_CLASSNAMES"], "F17K": META["F17K_DISEASE_113_CLASSES"]}


class DS(Dataset):
    def __init__(self, paths, tf):
        self.p, self.tf = paths, tf

    def __len__(self):
        return len(self.p)

    def __getitem__(self, i):
        with Image.open(self.p[i]) as im:
            return self.tf(im.convert("RGB"))


class OpenClipBackend:
    def __init__(self, model_name, pretrained, dev):
        self.model, _, self.tf = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        self.model = self.model.to(dev).eval()
        self.tok, self.dev = open_clip.get_tokenizer(model_name), dev

    @torch.no_grad()
    def encode_images(self, paths, bs=128):
        out = []
        for x in DataLoader(DS(paths, self.tf), batch_size=bs, num_workers=6):
            out.append(torch.nn.functional.normalize(self.model.encode_image(x.to(self.dev)).float(), dim=-1).cpu())
        return torch.cat(out).numpy()

    @torch.no_grad()
    def encode_texts(self, texts, bs=256):
        out = []
        for i in range(0, len(texts), bs):
            f = self.model.encode_text(self.tok(texts[i:i + bs]).to(self.dev))
            out.append(torch.nn.functional.normalize(f.float(), dim=-1).cpu())
        return torch.cat(out).numpy()


class _ProcDS(Dataset):
    """Decode + preprocess in DataLoader workers (the HF processor is CPU-bound)."""
    def __init__(self, paths, proc):
        self.p, self.proc = paths, proc

    def __len__(self):
        return len(self.p)

    def __getitem__(self, i):
        with Image.open(self.p[i]) as im:
            return self.proc(images=[im.convert("RGB")], return_tensors="pt")["pixel_values"][0]


class MedSigLipBackend:
    """google/medsiglip-448 (transformers). Text padded to max_length (64 tokens) as SigLIP requires.
    On CUDA the forward pass runs under bf16 autocast."""
    def __init__(self, repo, dev):
        from huggingface_hub import snapshot_download
        from transformers import AutoModel, AutoProcessor
        try:  # gated repo: prefer the local cache (copied from the machine that accepted the terms)
            repo = snapshot_download(repo, local_files_only=True)
        except Exception:
            pass
        self.model = AutoModel.from_pretrained(repo).to(dev).eval()
        self.proc, self.dev = AutoProcessor.from_pretrained(repo), dev
        self.amp = dev == "cuda"

    @torch.no_grad()
    def encode_images(self, paths, bs=64):
        out = []
        for x in DataLoader(_ProcDS(paths, self.proc), batch_size=bs, num_workers=8):
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.amp):
                f = self.model.get_image_features(pixel_values=x.to(self.dev))
            f = f.pooler_output if hasattr(f, "pooler_output") else f
            out.append(torch.nn.functional.normalize(f.float(), dim=-1).cpu())
        return torch.cat(out).numpy()

    @torch.no_grad()
    def encode_texts(self, texts, bs=128):
        out = []
        for i in range(0, len(texts), bs):
            x = self.proc(text=texts[i:i + bs], padding="max_length", truncation=True, max_length=64,
                          return_tensors="pt").to(self.dev)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.amp):
                f = self.model.get_text_features(**x)
            f = f.pooler_output if hasattr(f, "pooler_output") else f
            out.append(torch.nn.functional.normalize(f.float(), dim=-1).cpu())
        return torch.cat(out).numpy()


class EmbeddingGemmaBackend:
    """google/embeddinggemma-2 (sentence-transformers), bf16 (fp16 yields NaNs per model card).
    Images are encoded without prefix; class prompts / captions as search queries."""
    def __init__(self, repo, dev):
        from sentence_transformers import SentenceTransformer
        self.m = SentenceTransformer(repo, device=dev, model_kwargs={"torch_dtype": torch.bfloat16},
                                     config_kwargs={"audio_config": None})

    def encode_images(self, paths, bs=32):
        from concurrent.futures import ThreadPoolExecutor

        def load(p):  # decode + bound the size in threads; the processor rescales to its token budget
            im = Image.open(p).convert("RGB")
            im.thumbnail((1024, 1024))
            return {"image": im}
        out = []
        with ThreadPoolExecutor(12) as ex:
            batches = [paths[i:i + bs] for i in range(0, len(paths), bs)]
            nxt = ex.map(lambda b: [load(p) for p in b], batches)  # one batch per thread, in order
            for ims in nxt:
                out.append(self.m.encode(ims, batch_size=bs, normalize_embeddings=True, convert_to_numpy=True))
        return np.concatenate(out).astype(np.float32)

    def encode_texts(self, texts, bs=64):
        return self.m.encode([f"task: search result | query: {t}" for t in texts], batch_size=bs,
                             normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)


def load_backend(a, dev):
    if a.backend == "medsiglip":
        return MedSigLipBackend(a.pretrained, dev)
    if a.backend == "embeddinggemma":
        return EmbeddingGemmaBackend(a.pretrained, dev)
    return OpenClipBackend(a.model, a.pretrained, dev)


def zs_classifier(be, classes):
    W = []
    for c in classes:
        e = be.encode_texts([t.format(c=c) for t in TEMPLATES]).mean(0)
        W.append(e / np.linalg.norm(e))
    return np.stack(W, 1)


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def main():
    if os.environ.get("DUMP_STACK_AFTER"):  # debugging aid for hangs
        import faulthandler
        faulthandler.dump_traceback_later(int(os.environ["DUMP_STACK_AFTER"]), exit=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--model", default="ViT-B-16-quickgelu")
    ap.add_argument("--pretrained", default="openai", help="open_clip tag / checkpoint path, or HF repo")
    ap.add_argument("--backend", default="openclip", choices=["openclip", "medsiglip", "embeddinggemma"])
    ap.add_argument("--lp-benches", default="PAD,HAM,F17K,Daffodil",
                    help="linear-probe benchmarks (slow backbones: PAD,HAM)")
    a = ap.parse_args()
    dev = device()
    be = load_backend(a, dev)

    B = pd.read_parquet(WORK / "bench.parquet")
    lp = a.lp_benches.split(",")
    need = B[(B.task == "ZS") | (B.bench == "SkinCAP") | ((B.task == "LP") & B.bench.isin(lp))]
    paths = sorted(need.path.unique())
    F = dict(zip(paths, be.encode_images(paths)))
    res = {"name": a.name, "backend": a.backend, "model": a.model, "pretrained": a.pretrained}

    for bench, classes in CLASSES.items():
        d = B[(B.bench == bench) & (B.task == "ZS")]
        X = np.stack([F[p] for p in d.path])
        y = d.label.values
        p = softmax(100 * X @ zs_classifier(be, classes))
        # accuracy per Fitzpatrick group (expert labels; F17K and PAD only) for the bias analysis
        if d.fst.notna().any():
            for gname, lo, hi in [("I-II", 1, 2), ("III-IV", 3, 4), ("V-VI", 5, 6)]:
                m = (d.fst >= lo).values & (d.fst <= hi).values
                if m.sum() >= 20:
                    res[f"zs_{bench}_acc_fst_{gname}"] = accuracy_score(y[m], p[m].argmax(1))
                    res[f"n_{bench}_fst_{gname}"] = int(m.sum())
        if bench == "F17K":
            res["zs_F17K_top1"] = accuracy_score(y, p.argmax(1))
            res["zs_F17K_top5"] = top_k_accuracy_score(y, p, k=5, labels=np.arange(len(classes)))
        else:
            res[f"zs_{bench}_acc"] = accuracy_score(y, p.argmax(1))
            res[f"zs_{bench}_auroc"] = roc_auc_score(np.eye(len(classes))[y], p, multi_class="ovr", average="macro")
    res["zs_mean_acc"] = float(np.mean([res["zs_PAD_acc"], res["zs_HAM_acc"], res["zs_F17K_top1"], res["zs_Daffodil_acc"]]))

    for bench in lp:
        d = B[(B.bench == bench) & (B.task == "LP")]
        sp = {s: (np.stack([F[p] for p in d[d.split == s].path]), d[d.split == s].label.values)
              for s in ["train", "val", "test"]}
        best = None
        for C in [0.1, 0.3, 1, 3, 10, 30]:
            m = LogisticRegression(C=C, max_iter=3000).fit(*sp["train"])
            acc = accuracy_score(sp["val"][1], m.predict(sp["val"][0]))
            if best is None or acc > best[0]:
                best = (acc, C, m)
        pred = best[2].predict(sp["test"][0])
        res[f"lp_{bench}_acc"] = accuracy_score(sp["test"][1], pred)
        res[f"lp_{bench}_f1"] = f1_score(sp["test"][1], pred, average="macro")
        res[f"lp_{bench}_C"] = best[1]
    res["lp_mean_acc"] = float(np.mean([res[f"lp_{b}_acc"] for b in lp]))
    res["lp_benches"] = lp
    np.savez_compressed(WORK / f"eval/feats_{a.name}.npz", paths=np.array(paths),
                        F=np.stack([F[p] for p in paths]).astype(np.float16))

    sc = B[B.bench == "SkinCAP"].dropna(subset=["caption"])
    I = np.stack([F[p] for p in sc.path])
    T = be.encode_texts(sc.caption.tolist())
    S = I @ T.T
    gt = np.arange(len(sc))
    for k in (1, 5, 10):
        res[f"ret_SkinCAP_i2t_R@{k}"] = float((np.argsort(-S, 1)[:, :k] == gt[:, None]).any(1).mean())
        res[f"ret_SkinCAP_t2i_R@{k}"] = float((np.argsort(-S.T, 1)[:, :k] == gt[:, None]).any(1).mean())

    os.makedirs(WORK / "eval", exist_ok=True)
    json.dump(res, open(WORK / f"eval/{a.name}.json", "w"), indent=1, default=float)
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in res.items()}, indent=0))


if __name__ == "__main__":
    main()
