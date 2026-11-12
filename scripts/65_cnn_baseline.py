"""Conventional CNN baseline: EfficientNetV2-S (ImageNet-21k -> 1k pretrained, timm) fine-tuned on the
benchmark linear-probe splits (train / val / test of the Derm1M benchmark package).

Usage: python 65_cnn_baseline.py --bench HAM [--epochs 15 --img 300 --bs 32 --lr 3e-4]
Model selection on val (balanced accuracy); test metrics: accuracy, balanced accuracy, macro-F1,
macro OvR AUROC. Writes work/eval/cnn_<bench>.json.
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

from common import WORK, device


class DS(Dataset):
    def __init__(self, df, tf):
        self.p, self.y, self.tf = df.path.tolist(), df.label.values, tf

    def __len__(self):
        return len(self.p)

    def __getitem__(self, i):
        with Image.open(self.p[i]) as im:
            return self.tf(im.convert("RGB")), int(self.y[i])


def metrics(y, P):
    k = P.shape[1]
    out = {"acc": accuracy_score(y, P.argmax(1)), "bal_acc": balanced_accuracy_score(y, P.argmax(1)),
           "macro_f1": f1_score(y, P.argmax(1), average="macro")}
    try:
        out["auroc"] = roc_auc_score(np.eye(k)[y], P, multi_class="ovr", average="macro")
    except ValueError:
        out["auroc"] = float("nan")
    return out


@torch.no_grad()
def predict(model, dl, dev):
    model.eval()
    P, Y = [], []
    for x, y in dl:
        P.append(F.softmax(model(x.to(dev)).float(), -1).cpu().numpy()); Y.append(y.numpy())
    return np.concatenate(Y), np.concatenate(P)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--img", type=int, default=300)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--max-train", type=int, default=0, help="subsample train (same seed as 64_finetune_d1)")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    torch.manual_seed(0)
    dev = device()
    B = pd.read_parquet(WORK / "bench.parquet")
    d = B[(B.bench == a.bench) & (B.task == "LP")]
    k = int(d.label.max()) + 1
    if a.max_train:
        tr = d[d.split == "train"].sample(min(a.max_train, int((d.split == "train").sum())), random_state=0)
        d = pd.concat([tr, d[d.split != "train"]])
    mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    tf_tr = T.Compose([T.RandomResizedCrop(a.img, scale=(0.5, 1.0)), T.RandomHorizontalFlip(), T.RandomVerticalFlip(),
                       T.ColorJitter(0.2, 0.2, 0.2, 0.02), T.ToTensor(), T.Normalize(mean, std)])
    tf_te = T.Compose([T.Resize(a.img), T.CenterCrop(a.img), T.ToTensor(), T.Normalize(mean, std)])
    dl = {s: DataLoader(DS(d[d.split == s], tf_tr if s == "train" else tf_te), batch_size=a.bs,
                        shuffle=(s == "train"), num_workers=8, persistent_workers=True) for s in ["train", "val", "test"]}
    model = timm.create_model("tf_efficientnetv2_s.in21k_ft_in1k", pretrained=True, num_classes=k).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.05)
    steps = a.epochs * len(dl["train"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.1)
    best, hist, t0 = (-1, None), [], time.time()
    for ep in range(a.epochs):
        model.train()
        for x, y in dl["train"]:
            loss = F.cross_entropy(model(x.to(dev)), y.to(dev), label_smoothing=0.1)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        yv, Pv = predict(model, dl["val"], dev)
        mv = metrics(yv, Pv)
        hist.append({"epoch": ep + 1, **mv, "sec": time.time() - t0})
        print(hist[-1], flush=True)
        if mv["bal_acc"] > best[0]:
            best = (mv["bal_acc"], {k_: v.detach().clone() for k_, v in model.state_dict().items()})
    model.load_state_dict(best[1])
    yt, Pt = predict(model, dl["test"], dev)
    res = {"model": "EfficientNetV2-S (in21k->1k)", "bench": a.bench, **{f"test_{k_}": v for k_, v in metrics(yt, Pt).items()},
           "best_val_bal_acc": best[0], "epochs": a.epochs, "img": a.img, "history": hist,
           "train_seconds": time.time() - t0, "n_train": int((d.split == "train").sum())}
    os.makedirs(WORK / "eval", exist_ok=True)
    json.dump(res, open(WORK / f"eval/cnn_{a.bench}{a.tag}.json", "w"), indent=1, default=float)
    print({k_: v for k_, v in res.items() if k_ != "history"})


if __name__ == "__main__":
    main()
