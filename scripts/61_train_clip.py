"""Contrastive fine-tuning of an open_clip model on (image, text) pairs (single Apple-silicon GPU).

Usage: python 61_train_clip.py --pairs <pairs.parquet> --out <dir> [--steps 3000 --bs 128 --lr 1e-5]
pairs.parquet needs columns `path`, `text`. Same recipe for every training set so that
datasets can be compared at equal compute: OpenAI ViT-B/16 init, AdamW (wd 0.1),
cosine schedule with 5% warm-up, RandomResizedCrop(0.4-1) + colour jitter, bf16 autocast.
"""
from common import device
import argparse
import json
import math
import os
import random
import time

import numpy as np
import open_clip
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

Image.MAX_IMAGE_PIXELS = None


class Pairs(Dataset):
    def __init__(self, df, tf, tok):
        self.p, self.t, self.tf, self.tok = df.path.tolist(), df.text.tolist(), tf, tok

    def __len__(self):
        return len(self.p)

    def __getitem__(self, i):
        try:
            with Image.open(self.p[i]) as im:
                im.seek(0)
                x = self.tf(im.convert("RGB"))
        except Exception:
            x = torch.zeros(3, 224, 224)
        return x, self.tok([self.t[i]])[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="ViT-B-16-quickgelu")
    ap.add_argument("--pretrained", default="openai")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    torch.manual_seed(a.seed); random.seed(a.seed); np.random.seed(a.seed)
    os.makedirs(a.out, exist_ok=True)
    dev = device()

    model, tf_train, _ = open_clip.create_model_and_transforms(
        a.model, pretrained=a.pretrained,
        aug_cfg=dict(scale=(0.4, 1.0), color_jitter=(0.32, 0.32, 0.32, 0.08), color_jitter_prob=0.8,
                     gray_scale_prob=0.2))
    model = model.to(dev).train()
    tok = open_clip.get_tokenizer(a.model)
    df = pd.read_parquet(a.pairs)
    dl = DataLoader(Pairs(df, tf_train, tok), batch_size=a.bs, shuffle=True, num_workers=a.workers,
                    drop_last=True, persistent_workers=True)
    decay = [p for n, p in model.named_parameters() if p.ndim >= 2 and "logit_scale" not in n]
    no_decay = [p for n, p in model.named_parameters() if p.ndim < 2 or "logit_scale" in n]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": a.wd}, {"params": no_decay, "weight_decay": 0}],
                            lr=a.lr, betas=(0.9, 0.98), eps=1e-6)
    warm = max(1, int(0.05 * a.steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / (a.steps - warm))))

    log, step, t0 = [], 0, time.time()
    while step < a.steps:
        for x, t in dl:
            x, t = x.to(dev, non_blocking=True), t.to(dev, non_blocking=True)
            with torch.autocast(device_type=dev, dtype=torch.bfloat16):
                fi = F.normalize(model.encode_image(x), dim=-1)
                ft = F.normalize(model.encode_text(t), dim=-1)
                logits = model.logit_scale.exp() * fi @ ft.T
            labels = torch.arange(len(x), device=dev)
            loss = (F.cross_entropy(logits.float(), labels) + F.cross_entropy(logits.float().T, labels)) / 2
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            with torch.no_grad():
                model.logit_scale.clamp_(0, math.log(100))
            step += 1
            if step % 50 == 0 or step == 1:
                el = time.time() - t0
                rec = dict(step=step, loss=float(loss), lr=sched.get_last_lr()[0], img_s=step * a.bs / el)
                log.append(rec)
                print(json.dumps(rec), flush=True)
            if step >= a.steps:
                break
    torch.save(model.state_dict(), os.path.join(a.out, "model.pt"))
    json.dump(dict(args=vars(a), log=log, n_pairs=len(df), seconds=time.time() - t0),
              open(os.path.join(a.out, "train_log.json"), "w"), indent=1)
    print("saved", a.out)


if __name__ == "__main__":
    main()
