# derm-datasets — code for DermaLit-140k, relaion-derm and the SynthDerm experiments

Code of the project "The future of medical datasets: large-scale analysis and training of medical multimodal models through an iterative ML approach" (DFG Walter Benjamin Programme, R. Kaczmarczyk).

The code builds and analyses three things:

| Result | Where | Scripts |
|---|---|---|
| **DermaLit-140k**, a provenance-tracked dermatology image–text corpus (open tier + metadata release) | https://huggingface.co/datasets/laion/DermaLit-140k | `10`–`42` |
| **relaion-derm**, 189,422 links to likely dermatology images in Re-LAION-2B-en, and an estimate for all of Re-LAION-5B | https://huggingface.co/datasets/laion/relaion-derm | `71`, `105_relaion5b`, `106_laion_index_head`, `107_laion_index_score` |
| Comparison with Derm1M, model studies, bias analyses and synthetic-image experiments | results in the project report and preprint | `50`–`102` |

This repository contains code only. It contains no images, no captions and no scraped content. The raw source material is under copyright and is not distributed; the ingest scripts (`10`–`12`) document how a local, lawfully obtained copy was read.

## Script groups

Scripts are numbered in the order in which they run. All of them read and write below the project folder defined in `scripts/common.py`.

| Numbers | Step |
|---|---|
| `10`–`12` | ingest web atlases, journal figures with captions, textbooks |
| `20`–`23` | image checks, nearest neighbours, near-duplicate clusters, same-patient links |
| `29`–`34` | CLIP / MedSigLIP / EmbeddingGemma embeddings, image-type and Fitzpatrick heads |
| `40`–`42` | build the dataset, statistics, public release (open tier + metadata) |
| `50`–`58` | Derm1M index and comparison, benchmark overlap, diagnosis linking (SapBERT), concept map |
| `60`–`69` | CLIP evaluation and equal-compute fine-tuning, decision models, CNN baseline, adapters, linear probes |
| `70`–`78` | bias analyses: skin type, population references, disease prevalence (GBD 2019), sex, per-skin-type performance |
| `71`, `105_relaion5b`, `106_laion_index_head`, `107_laion_index_score` | dermatology images in Re-LAION: one-shard pilot, caption scan of all 5.35 B pairs, linear head in ViT-L/14 space, scoring of all 966 M index entries on compressed vectors, nudity screen, precision review |
| `79`–`84` | pool of permissively licensed reference photographs (CC BY / CC0) |
| `85`–`102`, `105_realism_prompts`, `106_realism_probe`, `107_pilot_score` | synthetic-image experiments: prompt plan, lesion-lifecycle taxonomy, generation clients, automatic quality gate (`86_eval_synthetic.py`), batch review, LoRA tests, reference-based variants, realism probe |
| `80`, `90`, `95` | figures, result tables, report build |
| `annotator/` | local web app for blinded expert review of synthetic images |

Endpoints for local language and image models are read from environment variables (for example `LLM_BASE_URL`) and default to `localhost`. `donv` in comments is the name of the project's GPU server (2 × RTX 3090).

## Scoring 966 million images without downloading them

`107_laion_index_score.py` applies a linear classifier to the product-quantised vectors of the public Re-LAION-2B-en kNN index. Because the classifier is linear, each logit is one term per inverted list plus a 128 × 256 lookup table per class. The whole index is scored on one CPU in a few minutes. See the docstring for the derivation and the checks.

## Safety rules built into the code

- Images flagged by the nudity screen are never downloaded, viewed or released.
- Synthetic prompts for minors are limited to face, arms and hands; intimate sites are excluded from the open tier.
- Benchmark near-duplicates are removed before any model training.

## Install

```bash
uv venv && uv pip install -r requirements.txt
```

Python 3.13. Package versions are those used in the project.

## Licence

MIT (code). Datasets have their own licences; see the dataset cards.

## Funding

Funded by the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation), Walter Benjamin Programme, project number 526052741.
