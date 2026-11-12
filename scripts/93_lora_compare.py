"""Base vs LoRA comparison for a per-disease FLUX.2-klein-4B LoRA (runs ON donv with Unsloth Studio's Python,
which has diffusers + the cached base model). Same prompts and seeds with and without the LoRA; a control prompt of
another disease checks that the LoRA does not leak into unrelated conditions.
Usage on donv: ~/unsloth-studio/<venv>/bin/python 93_lora_compare.py --lora <dir or .safetensors> --out <dir> --prompts prompts.json
"""
import argparse
import json
from pathlib import Path

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lora", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--base", default="unsloth/FLUX.2-klein-4B")
    ap.add_argument("--size", type=int, default=768)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--guidance", type=float, default=None, help="for the non-distilled base model (e.g. 4.0)")
    a = ap.parse_args()
    from diffusers import Flux2KleinPipeline
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    prompts = json.load(open(a.prompts))
    pipe = Flux2KleinPipeline.from_pretrained(a.base, torch_dtype=torch.bfloat16).to("cuda")
    pipe.set_progress_bar_config(disable=True)

    def run(tag):
        for i, p in enumerate(prompts):
            g = torch.Generator("cuda").manual_seed(1000 + i)
            kw = {} if a.guidance is None else {"guidance_scale": a.guidance}
            im = pipe(prompt=p["prompt"], height=a.size, width=a.size, num_inference_steps=a.steps, generator=g, **kw).images[0]
            im.save(out / f"{i:02d}_{tag}.jpg", quality=92)
    run("base")
    lora = Path(a.lora)
    pipe.load_lora_weights(str(lora.parent if lora.is_file() else lora),
                           weight_name=lora.name if lora.is_file() else "pytorch_lora_weights.safetensors", adapter_name="d")
    pipe.set_adapters(["d"], adapter_weights=[a.scale])
    run("lora")
    json.dump(prompts, open(out / "prompts.json", "w"), indent=1)
    print("done", out)


if __name__ == "__main__":
    main()
