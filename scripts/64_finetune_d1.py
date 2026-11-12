"""Fine-tune the decision model d1-3B (LiquidAI, LFM2.5-VL-3B backbone) with LoRA as an image classifier,
using its own one-pass readout: the prompt is the image + a `choice` question whose options are the
class names; the loss is cross-entropy over the options' answer-token logits (max-pooled per option,
exactly as d1's readout does at inference).

Usage: python 64_finetune_d1.py --bench HAM [--epochs 2 --lr 1e-4 --rank 16 --accum 8 --max-side 384]
Model selection on val (balanced accuracy, evaluated after each epoch);
writes work/eval/d1ft_<bench>.json and the LoRA adapter to work/models/d1ft_<bench>/.
Zero-shot d1 (no fine-tuning) on the same test split is reported as epoch 0.
"""
import argparse
import json
import os
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score

from common import WORK, device

DEV = device()
REPO, REV = "LiquidAI/d1-3B", "051bcc464b01b9f92942b364d9586b0ef5912432"
META = json.load(open(os.path.join(os.path.dirname(__file__), "derm1m_zeroshot_meta.json")))
CLASSES = {"PAD": META["PAD_CLASSNAMES"], "HAM": META["HAM_CLASSNAMES"],
           "Daffodil": META["DAFFODIL_5_CLASSNAMES"], "F17K": META["F17K_DISEASE_113_CLASSES"]}


def metrics(y, P):
    k = P.shape[1]
    out = {"acc": accuracy_score(y, P.argmax(1)), "bal_acc": balanced_accuracy_score(y, P.argmax(1)),
           "macro_f1": f1_score(y, P.argmax(1), average="macro")}
    try:
        out["auroc"] = roc_auc_score(np.eye(k)[y], P, multi_class="ovr", average="macro")
    except ValueError:
        out["auroc"] = float("nan")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--max-side", type=int, default=384)
    ap.add_argument("--max-train", type=int, default=0)
    ap.add_argument("--limit-eval", type=int, default=0, help="smoke tests only")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    torch.manual_seed(0); random.seed(0)
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModel

    model = AutoModel.from_pretrained(REPO, revision=REV, trust_remote_code=True, dtype=torch.bfloat16).to(DEV)
    eng = model.engine                                  # d1's SystemOne runner (prompt rendering + processor)
    import importlib
    prompt = importlib.import_module(type(eng).__module__.rsplit(".", 1)[0] + ".prompt")
    classes = CLASSES[a.bench]
    q = prompt.as_question({"type": "choice", "instructions": "Which skin condition is shown in this image?",
                            "criteria": {c: c for c in classes}})
    groups = prompt.readout_ids(eng.tokenizer, q)       # token ids per option (max-pooled)
    eng._image_markup(1)                                # loads the processor
    prefix = prompt.prefix_text(eng.tokenizer, None, eng.bos, eng.state_style, eng.system, eng._image_markup(1))
    suffix = prompt.suffix_text(eng.tokenizer, q, eng.lead, eng.option_style)
    flat = torch.tensor([t for g in groups for t in g], device=DEV)
    owner = torch.tensor([i for i, g in enumerate(groups) for _ in g], device=DEV)

    def option_logits(img):
        inputs = eng._image_inputs(prefix + suffix, [img])
        logits = model(**inputs, logits_to_keep=1).logits[0, -1].float()
        z = logits - torch.logsumexp(logits, -1)
        sel = z[flat]
        return torch.stack([sel[owner == i].max() for i in range(len(groups))])

    def load(p):
        im = Image.open(p).convert("RGB")
        im.thumbnail((a.max_side, a.max_side), Image.Resampling.BICUBIC)
        return im

    B = pd.read_parquet(WORK / "bench.parquet")
    d = B[(B.bench == a.bench) & (B.task == "LP")]
    tr, va, te = (d[d.split == s] for s in ("train", "val", "test"))
    if a.max_train:
        tr = tr.sample(min(a.max_train, len(tr)), random_state=0)
    if a.limit_eval:
        va, te = va.head(a.limit_eval), te.head(a.limit_eval)

    @torch.no_grad()
    def evaluate(df):
        model.eval()
        P = np.stack([F.softmax(option_logits(load(p)), -1).cpu().numpy() for p in df.path])
        return metrics(df.label.values, P)

    hist = [{"epoch": 0, **{f"val_{k}": v for k, v in evaluate(va).items()},
             **{f"test_{k}": v for k, v in evaluate(te).items()}}]
    print(hist[-1], flush=True)

    cfg = LoraConfig(r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.05,
                     target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "w1", "w2", "w3", "in_proj",
                                     "fc1", "fc2", "linear_1", "linear_2"])
    model = get_peft_model(model, cfg)
    model.print_trainable_parameters()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.0)
    total = a.epochs * len(tr) // a.accum
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / max(1, total // 20)) *
                                              max(0.0, 1 - s / max(1, total)))
    best, t0, step = (-1, None), time.time(), 0
    for ep in range(a.epochs):
        model.train()
        order = tr.sample(frac=1, random_state=ep)
        run = 0.0
        for i, r in enumerate(order.itertuples()):
            img = load(r.path)
            if random.random() < 0.5:
                img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            loss = F.cross_entropy(option_logits(img)[None], torch.tensor([int(r.label)], device=DEV)) / a.accum
            loss.backward()
            run += float(loss) * a.accum
            if (i + 1) % a.accum == 0:
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True); step += 1
                if step % 50 == 0:
                    print({"epoch": ep + 1, "step": step, "loss": round(run / (50 * a.accum), 4),
                           "img_s": round((i + 1) / (time.time() - t0), 2)}, flush=True)
                    run = 0.0
        mv, mt = evaluate(va), evaluate(te)
        hist.append({"epoch": ep + 1, **{f"val_{k}": v for k, v in mv.items()}, **{f"test_{k}": v for k, v in mt.items()},
                     "sec": time.time() - t0})
        print(hist[-1], flush=True)
        if mv["bal_acc"] > best[0]:
            best = (mv["bal_acc"], hist[-1])
            model.save_pretrained(WORK / f"models/d1ft_{a.bench}{a.tag}")
    res = {"model": "d1-3B + LoRA", "bench": a.bench, "zero_shot": hist[0], "selected": best[1],
           "args": vars(a), "history": hist, "n_train": len(tr)}
    json.dump(res, open(WORK / f"eval/d1ft_{a.bench}{a.tag}.json", "w"), indent=1, default=float)
    print("selected:", best[1])


if __name__ == "__main__":
    main()
