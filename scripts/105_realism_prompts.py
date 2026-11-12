"""Realism prompt variant for SynthDerm (pilot 2026-11-10, after expert feedback that most images look exaggerated:
geometric lesion patterns, dot lattices, one oversized central lesion, mirror-identical limbs, white studio look).

Changes against the batch-5 prompt (94_lifecycle_prompts.py + prose conversion):
  * lesion text rewritten by the Splash LLM into 2-3 short camera-only sentences: no mm sizes, no differential /
    dermoscopy / time-course terms, no "no X" clauses, no "prominent / high contrast / highly visible"; irregular and
    asymmetric wording ("a few scattered, unevenly sized ..."), count and size relative to the body part, mild to
    moderate severity by default (cache: work/synthetic/realism_lesion_text.json);
  * no negations anywhere (these models have no negative prompt; naming "collage", "text", "exaggerated" primes them);
  * photographic context first and sampled per row: setting, background, camera / flash, light, framing, ordinary
    body details (moles, body hair, sun damage) - instead of "even diffuse clinical lighting" (= white studio).
Usage: python 105_realism_prompts.py --rows ../work/synthetic/realism_pilot_rows.parquet --out ../work/synthetic/realism_pilot_rows_new.parquet
"""
import argparse
import json
import random
import re
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LLM = "http://localhost:4242/v1"
MODEL = "incoai/Qwen3.8-27B-Splash"
CACHE = ROOT / "work/synthetic/realism_lesion_text.json"
SETTING = ["in a dermatology outpatient exam room", "at a hospital bedside", "in a general practice consultation room",
           "in a small community clinic", "in a busy hospital dermatology clinic"]
BACKGROUND = ["a blue paper drape at the edge of the frame", "white couch roll paper and the edge of an exam couch in the background",
              "a beige wall and part of a chair in the background", "a green surgical towel under the body part",
              "the patient's own clothes at the edges of the frame", "a grey hospital blanket in the background"]
CAMERA = ["taken with a smartphone", "taken with a compact camera and its built-in flash", "taken with a DSLR and an on-camera flash",
          "taken with a phone held close to the skin"]
# neutral light only: "warm examination lamp" / "overhead fluorescent" gave a yellow-orange cast in about half the
# arm-D images (pilot 2026-11-11)
LIGHT = ["under neutral white LED examination light", "with slightly harsh direct flash", "in soft window daylight",
         "in bright, neutral daylight-balanced clinic light"]
# paired limb sites in the plural made MOSS (30 steps, CFG 3) splice two photos side by side; one side is named instead
PAIRED = {"arms": "arm", "upper arms": "upper arm", "forearms": "forearm", "legs": "leg", "lower legs": "lower leg",
          "shins": "shin", "thighs": "thigh", "hands": "hand", "feet": "foot", "palms": "palm", "soles": "sole",
          "knees": "knee", "elbows": "elbow", "ankles": "ankle", "calves": "calf", "wrists": "wrist", "axillae": "axilla"}
FRAMING = ["slightly off-centre framing", "the body part fills most of the frame at a slight angle", "photographed from about half a metre away",
           "a little tilted, as taken quickly during the consultation"]
BODY = {"young": ["ordinary skin with a few moles and fine body hair", "normal skin with visible pores and a few freckles",
                  "ordinary skin with a small scar and some body hair"],
        "old": ["ageing skin with fine wrinkles, a few moles and some sun damage", "slightly dry, thin skin with age spots and fine body hair",
                "ordinary older skin with a few seborrhoeic keratoses and sun damage"]}
SCHEMA = {"type": "object", "properties": {"lesion": {"type": "string"}}, "required": ["lesion"]}
RULES = ("Rewrite this dermatology lesion description for a photorealistic image generator. Write 2 to 3 short sentences "
         "describing ONLY what a camera would see on this patient: type of lesion, roughly how many, how large relative to "
         "the body part (e.g. 'coin-sized', 'covering part of the shin'), colour relative to the surrounding skin of this "
         "skin type, surface, and arrangement. Rules: irregular, natural wording - lesions are unevenly sized, unevenly "
         "spaced, clustered in places, asymmetric ('more on one side', 'on the left only'); default severity mild to "
         "moderate; never use the words prominent, high contrast, highly visible, striking, perfect, regular, pattern, "
         "grid, net or lattice; no measurements in mm or cm; no differential diagnosis, dermoscopy terms, causes, symptoms "
         "or time course; no negative statements of the form 'no X' or 'without X'. Output only the description.")


def lesion_part(prompt):
    m = re.search(r"stage\. (.*?) The lesion is realistic", prompt, flags=re.S)
    return m.group(1).strip() if m else ""


def rewrite(key, r):
    txt = (f"{RULES}\n\nDiagnosis: {r.diagnosis} ({r.variant}), stage: {r.stage}. Body site: {r.site}. Skin: Fitzpatrick "
           f"type {r.fst_target}. Original description: {lesion_part(r.prompt)}")
    body = {"model": MODEL, "reasoning_effort": "low", "temperature": 0.3, "max_tokens": 1500,
            "messages": [{"role": "user", "content": txt}],
            "response_format": {"type": "json_schema", "json_schema": {"name": "lesion", "schema": SCHEMA}}}
    req = urllib.request.Request(LLM + "/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    for _ in range(3):
        try:
            with urllib.request.urlopen(req, timeout=600) as h:
                c = json.load(h)["choices"][0]["message"]["content"]
            v = json.loads(re.sub(r"<think>.*?</think>", "", c or "", flags=re.S).strip())["lesion"].strip()
            if 30 < len(v) < 600:
                return key, v
        except (OSError, ValueError, KeyError, TypeError):
            pass
    return key, None


def subject(r):
    """'<site> of a 57-year-old woman of Irish descent with ... skin (Fitzpatrick skin type I)' from the plan prompt."""
    m = re.search(r"(?:Close-up|Regional clinical view) of the (.*?\(Fitzpatrick skin type [IVX]+\))\.", r.prompt)
    s = m.group(1) if m else f"{r.site} of a {r.age}-year-old {r.sex} patient"
    return re.sub(r",? wearing a plain white sports bra and dark shorts|,? wearing dark shorts", "", s), \
        ("wearing a plain sports bra and dark shorts" if "sports bra" in r.prompt else "wearing dark shorts" if "dark shorts" in r.prompt else "")


def one_side(subj, rnd):
    """'proximal upper arms of a ...' -> 'left proximal upper arm of a ...' (paired limb sites only)."""
    for pl, sg in sorted(PAIRED.items(), key=lambda kv: -len(kv[0])):
        m = re.match(rf"^((?:[a-z-]+ )*?){re.escape(pl)}\b", subj)
        if m:
            return f"{rnd.choice(['left', 'right'])} {m.group(1)}{sg}" + subj[m.end():]
    return subj


def build(r, lesion):
    rnd = random.Random(int(r.seed))
    subj, wear = subject(r)
    subj = one_side(subj, rnd)
    body = rnd.choice(BODY["old" if r.age >= 60 else "young"])
    ctx = (f"Amateur clinical photograph {rnd.choice(SETTING)}, {rnd.choice(CAMERA)} {rnd.choice(LIGHT)}, "
           f"{rnd.choice(FRAMING)}, with {rnd.choice(BACKGROUND)}.")
    who = f"Close-up of the {subj}" + (f", {wear}" if wear else "") + f"; {body}."
    return f"{ctx} {who} The patient has {r.diagnosis} ({r.variant}), {r.stage} stage. {lesion} Natural, true-to-life skin texture and colour."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    R = pd.read_parquet(a.rows)
    cache = json.load(open(CACHE)) if CACHE.exists() else {}
    todo = [(r.synth_id, r) for r in R.itertuples() if r.synth_id not in cache]
    with ThreadPoolExecutor(2) as ex:
        for k, v in ex.map(lambda t: rewrite(*t), todo):
            if v: cache[k] = v
    json.dump(cache, open(CACHE, "w"), indent=1, ensure_ascii=False)
    R = R[R.synth_id.isin(cache)].copy()
    R["lesion_old"] = R.prompt.map(lesion_part)
    R["lesion_new"] = R.synth_id.map(cache)
    R["prompt_old"] = R.prompt
    R["prompt"] = [build(r, cache[r.synth_id]) for r in R.itertuples()]
    bad = R.prompt.str.contains(r"\b(?:not|no|without|never)\b", case=False)
    print(len(R), "rows rewritten | prompts still containing a negation:", int(bad.sum()))
    R.to_parquet(a.out, index=False)


if __name__ == "__main__":
    main()
