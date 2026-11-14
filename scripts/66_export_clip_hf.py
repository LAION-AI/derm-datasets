"""Export a fine-tuned open_clip checkpoint (state dict from 61_train_clip.py) to the Hugging Face layout of open_clip.

Usage: python 66_export_clip_hf.py --ckpt ../work/models/<run>/model.pt --out ../release/dermalit-clip

Writes open_clip_model.safetensors, open_clip_pytorch_model.bin, open_clip_config.json and the tokenizer files, so that
    open_clip.create_model_and_transforms("hf-hub:<org>/<repo>")
loads the model. The export is checked: the exported model must give the same image and text embeddings as the checkpoint.
"""
import argparse
import importlib

import open_clip
import torch

hub = importlib.import_module("open_clip.push_to_hf_hub")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="ViT-B-16-quickgelu")
    ap.add_argument("--safetensors-only", action="store_true", help="do not write the pickle (.bin) copy")
    a = ap.parse_args()

    model, _, _ = open_clip.create_model_and_transforms(a.model, pretrained=None)
    sd = torch.load(a.ckpt, map_location="cpu")
    print(model.load_state_dict(sd, strict=True))
    model.eval()
    tok = hub.HFTokenizer("openai/clip-vit-base-patch16")
    hub.save_for_hf(model, tok, open_clip.get_model_config(a.model), a.out,
                    safe_serialization=True if a.safetensors_only else "both")

    # check: reload from the exported folder and compare embeddings
    m2, _, _ = open_clip.create_model_and_transforms(f"local-dir:{a.out}")
    m2.eval()
    t1 = open_clip.get_tokenizer(a.model)
    t2 = open_clip.get_tokenizer(f"local-dir:{a.out}")
    texts = ["a clinical photograph of psoriasis on the elbow", "dermoscopy of a melanocytic naevus"]
    torch.manual_seed(0)
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        d_img = (model.encode_image(x) - m2.encode_image(x)).abs().max().item()
        d_txt = (model.encode_text(t1(texts)) - m2.encode_text(t2(texts))).abs().max().item()
    print("max abs difference image / text:", d_img, d_txt)
    assert d_img < 1e-5 and d_txt < 1e-5


if __name__ == "__main__":
    main()
