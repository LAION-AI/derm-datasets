"""Zero-shot image-type scores with CLIP prompt ensembles.

Usage: python 31_zeroshot_types.py <emb.npy> <out.parquet>
Writes one probability column per image type (softmax over types, logit scale 100).
"""
import sys

import numpy as np
import open_clip
import pandas as pd
import torch

TYPES = {
    "clinical": ["a clinical photograph of a skin lesion", "a close-up photo of skin with a rash",
                 "a photo of a patient's skin disease", "a clinical photo of a skin condition on the body",
                 "a photo of a nail disease", "a photo of hair loss on the scalp",
                 "a clinical photo of the oral mucosa", "a photo of a skin tumor"],
    "dermoscopy": ["a dermoscopy image of a skin lesion", "a dermatoscopic image of a pigmented mole",
                   "a dermoscopic close-up of a nevus with a circular field of view"],
    "histopathology": ["a histopathology slide stained with hematoxylin and eosin",
                       "a microscopic image of skin tissue histology", "an H&E stained biopsy section",
                       "an immunohistochemistry stained tissue section"],
    "microscopy_other": ["an immunofluorescence microscopy image", "a fluorescence microscopy image of cells",
                         "an electron microscopy image", "a confocal microscopy image",
                         "a microscopy image of fungal hyphae or bacteria"],
    "radiology": ["an X-ray image", "a CT scan", "an MRI scan", "an ultrasound image",
                  "a PET scan image"],
    "chart": ["a bar chart", "a line graph with axes", "a scatter plot", "a Kaplan-Meier survival curve",
              "a forest plot", "a box plot of data", "a statistical chart from a scientific paper"],
    "diagram": ["a schematic diagram", "a flowchart", "a medical illustration drawing",
                "a diagram of a molecular signaling pathway", "a study design flow diagram"],
    "table_text": ["a table with text", "a screenshot of text", "a page of text", "a questionnaire form"],
    "molecular": ["a western blot", "a gel electrophoresis image", "a heatmap of gene expression",
                  "a protein structure model"],
    "other_photo": ["a portrait photo of a person", "a photo of a medical device", "a photo of a laboratory",
                    "a photo of a surgical operation", "a photo of a group of people", "a photo of an animal",
                    "a photo of a pill or cream product", "an endoscopy image"],
}


def main():
    emb, out = sys.argv[1], sys.argv[2]
    model, _, _ = open_clip.create_model_and_transforms("ViT-B-16-quickgelu", pretrained="openai")
    tok = open_clip.get_tokenizer("ViT-B-16-quickgelu")
    model.eval()
    with torch.no_grad():
        W = []
        for k, ps in TYPES.items():
            t = model.encode_text(tok(ps)).float()
            t = t / t.norm(dim=-1, keepdim=True)
            m = t.mean(0)
            W.append(m / m.norm())
        W = torch.stack(W).numpy()
    E = np.load(emb).astype(np.float32)
    logits = 100 * E @ W.T
    logits -= logits.max(1, keepdims=True)
    p = np.exp(logits)
    p /= p.sum(1, keepdims=True)
    df = pd.DataFrame(p, columns=[f"p_{k}" for k in TYPES])
    df["zs_type"] = np.array(list(TYPES))[p.argmax(1)]
    df.to_parquet(out, index=False)
    print(df.zs_type.value_counts())


if __name__ == "__main__":
    main()
