"""Generate plan rows with a diffusers FLUX.2-klein or Z-Image pipeline (runs ON donv with Unsloth Studio's Python, e.g. for the
non-distilled unsloth/FLUX.2-klein-base-4B at 40 steps with CFG, which the INT8 image server cannot run).
Input: a JSONL of {synth_id, prompt, seed}; output: <out>/<synth_id>.jpg (resumable). Same seeds as 96_generate_synthetic.py
for a paired comparison (the noise differs between backends, so pairs share prompt + seed, not the exact sample).
Usage on donv (GPU 0 free: ~/unsloth-studio/train_mode.sh on):
  HF_HOME=~/unsloth-studio/hf HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 ./unsloth_studio/bin/python 97_gen_diffusers.py \
      --rows pilot.jsonl --out gen/pilot_base40 --base unsloth/FLUX.2-klein-base-4B --steps 40 --guidance 4.0
  Z-Image-Turbo (Apache-2.0, 8-step distilled; recommended 9 steps, guidance 0):
      --pipeline zimage --base Tongyi-MAI/Z-Image-Turbo --steps 9 --guidance 0.0
"""
import argparse
import json
import time
from pathlib import Path

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base", default="unsloth/FLUX.2-klein-base-4B")
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--guidance", type=float, default=4.0)
    ap.add_argument("--pipeline", choices=["klein", "zimage"], default="klein")
    ap.add_argument("--lora", default=None)
    ap.add_argument("--lora-scale", type=float, default=1.0)
    a = ap.parse_args()
    import diffusers
    Pipe = diffusers.Flux2KleinPipeline if a.pipeline == "klein" else diffusers.ZImagePipeline
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in open(a.rows)]
    todo = [r for r in rows if not (out / f"{r['synth_id']}.jpg").exists()]
    print(len(todo), "of", len(rows), "to generate", flush=True)
    pipe = Pipe.from_pretrained(a.base, torch_dtype=torch.bfloat16).to("cuda")
    pipe.set_progress_bar_config(disable=True)
    if a.lora:
        lp = Path(a.lora)
        pipe.load_lora_weights(str(lp.parent), weight_name=lp.name, adapter_name="d")
        pipe.set_adapters(["d"], adapter_weights=[a.lora_scale])
    t0 = time.time()
    for i, r in enumerate(todo):
        g = torch.Generator("cuda").manual_seed(int(r["seed"]) % 2**31)
        im = pipe(prompt=r["prompt"], height=a.size, width=a.size, num_inference_steps=a.steps,
                  guidance_scale=a.guidance, generator=g).images[0]
        im.save(out / f"{r['synth_id']}.jpg", quality=93)
        if (i + 1) % 10 == 0:
            print(i + 1, f"{(time.time() - t0) / (i + 1):.1f} s/img", flush=True)
    json.dump(vars(a), open(out / "run_config.json", "w"), indent=1)
    print("done", len(todo), round((time.time() - t0) / 60, 1), "min", flush=True)


if __name__ == "__main__":
    main()
