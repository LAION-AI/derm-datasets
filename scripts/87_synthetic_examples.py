"""First synthetic examples (healthy base photograph -> disease projected by image editing) + automatic checks + HTML.

Generation: Splash image server on localhost:8080 (OpenAI-style /v1/images/generations and /v1/images/edits),
model FLUX.2-klein-4B ("klein", Apache-2.0), 1024x1024. (Prototype round 1 used qwen_image_2.1-turbo - non-commercial
licence, kept only as internal prototypes.) Each base image (healthy skin, given Fitzpatrick type, age group, sex, body site)
is generated once; disease variants are edits of the base ("keep the person, skin tone, lighting unchanged"), so the
healthy/diseased pairs differ only in the lesion. Prompts describe the morphology as it appears on the given skin type
(e.g. erythema violaceous / hyperpigmented on dark skin). Children only appear with arms / hands. Resumable.

Checks per image (same tools as 86_eval_synthetic.py): image-type head, Fitzpatrick head (expected agreement +-1),
nearest real training image (CLIP cos; >= 0.98 = near-duplicate), and for disease images the top-3 diagnoses of the
dermatology CLIP (fine-tuned on ours + Derm1M) over the 979 planned diagnoses + "normal skin".
Writes dataset/synthetic/examples/{img/*.png, manifest.json, checks.json, example_images.html}.
"""
import base64
import importlib
import json
import sys
import time
import types
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from common import DATASET, WORK, device, load_local_embeddings

OUT = DATASET / "synthetic/examples"
MODEL = "klein"   # FLUX.2-klein-4B (Apache-2.0) on the Splash server; the first 37 prototypes used qwen_image_2.1-turbo
                  # (Qwen Research Licence, non-commercial) and are NOT part of SynthDerm-1M
API = None   # resolved once in main(): mDNS lookups of localhost take ~5 s and occasionally time out
STYLE = ("Clinical dermatology photograph, natural skin texture with visible pores and fine vellus hair, neutral grey "
         "background, even diffuse clinical lighting, sharp focus, realistic, no jewelry, no tattoos, no text, no watermark. "
         "Non-sexual medical photograph: no genitals and no nudity beyond the body part shown.")
SKIN = {1: "very fair, pale white skin with freckles that always burns (Fitzpatrick skin type I), Northern European",
        2: "fair white skin (Fitzpatrick skin type II), Central European",
        3: "light olive, light brown skin (Fitzpatrick skin type III), Mediterranean",
        4: "moderate brown skin (Fitzpatrick skin type IV), South Asian",
        5: "dark brown skin (Fitzpatrick skin type V), East African",
        6: "very dark brown, almost black, deeply pigmented skin (Fitzpatrick skin type VI), West African"}
NUDITY_FLAG = 0.30   # open tier: images at or above this CLIP probability of exposed genitals / breasts / buttocks are held back
TIER = "open"        # this script only produces the open tier; the clinician tier is 88_synthetic_clinician.py
KEEP = " Keep the person, body part, framing, skin tone, lighting and background exactly unchanged. Realistic clinical photograph."

BASES = [  # id, fst, age group, sex, site, subject
    ("b01", 1, "elderly", "male", "face", "close-up of the nose, cheek and forehead of a 76-year-old man"),
    ("b02", 2, "adult", "female", "upper back", "close-up of the bare upper back and shoulders of a 35-year-old woman, photographed from behind, hair tied up"),
    ("b03", 3, "elderly", "female", "face", "close-up of the face (nose, cheeks, eyebrows) of a 70-year-old woman"),
    ("b04", 4, "adolescent", "male", "face", "close-up of the forehead and cheeks of a 16-year-old boy"),
    ("b05", 5, "adult", "male", "upper back", "close-up of the bare upper back and shoulders of a 40-year-old man, photographed from behind"),
    ("b06", 6, "adult", "female", "forearm / elbow crease", "close-up of the inner forearm and elbow crease of a 30-year-old woman"),
    ("b07", 6, "child", "male", "hands", "close-up of both hands, palms down, fingers slightly spread, of a 7-year-old boy"),
    ("b08", 5, "adult", "female", "shins", "close-up of both lower legs (shins) of a 45-year-old woman, standing"),
    ("b09", 4, "adult", "male", "flank / trunk", "close-up of the left side of the trunk (flank and lower chest) of a 50-year-old man, "
     "from the armpit down to the waist, wearing dark trousers; the image ends at the waistband"),
    ("b10", 6, "elderly", "male", "hands", "close-up of the backs of both hands of a 62-year-old man"),
    ("b11", 2, "child", "female", "forearm / elbow crease", "close-up of the inner forearm and elbow crease of a 5-year-old girl"),
    ("b12", 5, "elderly", "female", "sole of the foot", "close-up of the sole and heel of the right foot of a 72-year-old woman"),
    ("b13", 3, "adult", "female", "face", "close-up of the face (cheeks, upper lip, forehead) of a 28-year-old woman"),
    ("b14", 6, "adult", "male", "lower leg", "close-up of the right lower leg (shin and calf) of a 38-year-old man"),
    ("b15", 1, "elderly", "male", "nose / cheek", "close-up of the nose and left cheek of a 76-year-old man (only this part of the face fills the frame)"),
]

EDITS = [  # id, base, condition (plan concept name), stage / variant, prompt
    ("e01", "b15", "basal cell carcinoma", "nodular", "Add a clearly visible nodular basal cell carcinoma on the side of the nose: a pearly, translucent pink nodule about 10 mm with a rolled shiny border, fine arborizing telangiectasias on its surface and a small central crusted ulcer."),
    ("e02", "b01", "actinic keratosis", "multiple, thin (Olsen grade I-II)", "Add multiple thin actinic keratoses on the forehead and temple: rough, scaly, pink to skin-coloured macules and thin papules with adherent white scale, on chronically sun-damaged skin."),
    ("e03", "b02", "melanoma", "superficial spreading, early (thin)", "Add an early superficial spreading melanoma on the upper back: a 15 mm asymmetric, flat to slightly raised lesion with irregular notched borders and variegated colours (dark brown, black, tan, areas of pink and blue-grey)."),
    ("e04", "b02", "pityriasis rosea", "herald patch with secondary eruption", "Add pityriasis rosea: one 3 cm oval salmon-pink herald patch with a fine collarette of scale, and multiple smaller oval pink scaly patches aligned along the skin lines in a Christmas-tree pattern."),
    ("e05", "b03", "seborrheic dermatitis", "active flare", "Add an active flare of seborrheic dermatitis: clearly visible greasy, yellowish-white flaky scale on mildly pink skin in both nasolabial folds, along the eyebrows, between the eyebrows and at the sides of the nose. The cheeks stay normal, not diffusely red."),
    ("e06", "b03", "rosacea", "papulopustular", "Add papulopustular rosacea: persistent centrofacial redness of the cheeks and nose with fine telangiectasias and several small inflammatory papules and pustules; no comedones."),
    ("e07", "b04", "acne vulgaris", "inflammatory (papulopustular) with post-inflammatory hyperpigmentation", "Add moderate inflammatory acne vulgaris: open and closed comedones, inflammatory papules and a few pustules on the forehead and cheeks, with several dark brown post-inflammatory hyperpigmented macules; redness is subtle on brown skin."),
    ("e08", "b05", "pityriasis versicolor", "hypopigmented variant", "Add the hypopigmented variant of pityriasis versicolor: multiple round to oval, well-demarcated, lighter macules with very fine powdery scale, partly coalescing over the upper back and shoulders."),
    ("e09", "b06", "atopic dermatitis", "chronic, lichenified", "Add chronic atopic dermatitis in the elbow crease: ill-defined, thickened, lichenified plaques with markedly accentuated skin lines, hyperpigmented grey-brown to violaceous colour (not bright red, as typical on very dark skin), fine scale, a few excoriations, small follicular papules. No oozing."),
    ("e10", "b06", "atopic dermatitis", "acute, oozing with scratch marks", "Add a clearly visible acute flare of atopic dermatitis in the elbow crease: a swollen patch with wet, glistening, oozing erosions, many tiny vesicles, honey-yellow serous crusts and numerous fresh linear scratch marks (excoriations); the inflamed skin looks dark violaceous-brown rather than red on very dark skin. No lichenification."),
    ("e11", "b06", "urticaria", "acute", "Add acute urticaria: several raised, smooth, oedematous wheals of different sizes on the forearm, slightly darker violaceous with paler halos (redness hard to see on very dark skin), some coalescing; no scale."),
    ("e12", "b07", "scabies", "classic (non-crusted)", "Add clearly visible classic scabies: many small red-brown papules, tiny vesicles and short thin wavy greyish burrows between the fingers (interdigital web spaces), on the sides of the fingers, the palms and the wrists, with scratch marks and small crusts."),
    ("e13", "b08", "psoriasis", "untreated plaques with thick silvery scale", "Add untreated plaque psoriasis on both shins: several well-demarcated, thick, raised plaques covered by thick silvery-white scale on a violaceous to dark brown base, as typical on dark skin."),
    ("e14", "b08", "psoriasis", "treated, post-inflammatory hyperpigmentation", "Show treated, cleared plaque psoriasis on both shins: the former plaque sites are now flat, well-demarcated, round to oval dark brown macules (post-inflammatory hyperpigmentation) without any scale, thickening or redness; the rest of the skin is normal."),
    ("e15", "b09", "tinea corporis", "untreated, active border", "Add untreated tinea corporis on the flank: a 6 cm annular plaque with a raised, scaly, slightly darker reddish-brown active border and central clearing, with one smaller ring nearby."),
    ("e16", "b09", "herpes zoster", "vesicular stage (day 2-4)", "Add herpes zoster in the vesicular stage: a band-like, one-sided cluster of grouped clear and cloudy vesicles on a reddish-violaceous base following a dermatome around the flank, a few beginning to crust."),
    ("e17", "b10", "vitiligo", "non-segmental, stable", "Add stable non-segmental vitiligo: sharply demarcated, chalk-white depigmented macules and patches on the backs of both hands and the fingers, roughly symmetric, no scale, a few dark perifollicular dots of repigmentation inside the patches."),
    ("e18", "b11", "molluscum contagiosum", "uncomplicated", "Add uncomplicated molluscum contagiosum: about a dozen 2-4 mm dome-shaped, pearly, skin-coloured papules with a central dimple (umbilication), partly grouped, on the forearm."),
    ("e19", "b11", "atopic dermatitis", "acute, oozing with scratch marks", "Add an acute flare of atopic dermatitis in the elbow crease: ill-defined bright red, swollen patches with tiny vesicles, weeping and yellowish serous crusts, and many linear scratch marks."),
    ("e20", "b12", "acral lentiginous melanoma", "early (in situ / thin)", "Add an early acral lentiginous melanoma on the heel: an irregular, 2 cm, unevenly pigmented dark brown to black macule with blurred, irregular borders and a parallel-ridge pigment pattern."),
    ("e21", "b13", "melasma", "centrofacial", "Add centrofacial melasma: symmetric, irregular but well-defined, light to medium brown macular hyperpigmentation on both cheeks, the upper lip and the forehead, no scale, no redness."),
    ("e22", "b14", "cellulitis", "acute, non-purulent", "Add clearly visible acute non-purulent cellulitis of the shin: the lower leg is markedly swollen; a large, ill-defined area of glossy, stretched, tense skin appears distinctly darker purple-black than the surrounding skin (redness is hard to see on very dark skin), with peau d'orange texture; no blisters."),
]


def resolve_api(host="localhost", port=8080):
    import socket
    ip = socket.getaddrinfo(host, port, socket.AF_INET, socket.SOCK_STREAM)[0][4][0]
    return f"http://{ip}:{port}/v1/images"


def retry(fn, *a, tries=4):
    for k in range(tries):
        try:
            return fn(*a)
        except (OSError, urllib.error.URLError) as e:
            if k == tries - 1:
                raise
            print("retry after", repr(e)[:80], flush=True)
            time.sleep(20 * (k + 1))


def post_json(url, body):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        return json.load(r)


def post_edit(url, image_path, prompt):
    import uuid
    b = uuid.uuid4().hex
    parts = []
    for k, v in [("model", MODEL), ("size", "1024x1024"), ("prompt", prompt)]:
        parts.append(f'--{b}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode())
    parts.append(f'--{b}\r\nContent-Disposition: form-data; name="image"; filename="{Path(image_path).name}"\r\n'
                 f'Content-Type: image/png\r\n\r\n'.encode() + Path(image_path).read_bytes() + b"\r\n")
    parts.append(f"--{b}--\r\n".encode())
    req = urllib.request.Request(url, data=b"".join(parts), headers={"Content-Type": f"multipart/form-data; boundary={b}"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        return json.load(r)


def post_edit_sized(url, image_path, prompt, size):
    """Like post_edit, but with an explicit output size (e.g. to keep a reference image's aspect ratio)."""
    import uuid
    b = uuid.uuid4().hex
    parts = [f'--{b}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode()
             for k, v in [("model", MODEL), ("size", size), ("prompt", prompt)]]
    parts.append(f'--{b}\r\nContent-Disposition: form-data; name="image"; filename="{Path(image_path).name}"\r\n'
                 f'Content-Type: image/png\r\n\r\n'.encode() + Path(image_path).read_bytes() + b"\r\n")
    parts.append(f"--{b}--\r\n".encode())
    req = urllib.request.Request(url, data=b"".join(parts), headers={"Content-Type": f"multipart/form-data; boundary={b}"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        return json.load(r)


def generate():
    (OUT / "img").mkdir(parents=True, exist_ok=True)
    man = []
    for bid, fst, age, sex, site, subj in BASES:
        man.append({"id": bid, "kind": "healthy", "base": bid, "fst": fst, "age": age, "sex": sex, "site": site,
                    "condition": "normal skin", "prompt": f"{STYLE} Healthy skin, no lesions: {subj} with {SKIN[fst]}."})
    meta = {m["id"]: m for m in man}
    for eid, bid, cond, stage, p in EDITS:
        b = meta[bid]
        man.append({**{k: b[k] for k in ["fst", "age", "sex", "site"]}, "id": eid, "kind": "disease", "base": bid,
                    "condition": cond, "stage": stage, "prompt": p + KEEP})

    def run(m):
        f = OUT / f"img/{m['id']}.png"
        if f.exists():
            return m["id"], 0.0
        t = time.time()
        if m["kind"] == "healthy":
            d = retry(post_json, f"{API}/generations", {"model": MODEL, "prompt": m["prompt"], "size": "1024x1024"})
        else:
            d = retry(post_edit, f"{API}/edits", OUT / f"img/{m['base']}.png", m["prompt"])
        f.write_bytes(base64.b64decode(d["data"][0]["b64_json"]))
        return m["id"], time.time() - t
    with ThreadPoolExecutor(2) as ex:                       # bases first, then their edits
        for i, s in ex.map(run, [m for m in man if m["kind"] == "healthy"]):
            print(i, f"{s:.0f}s", flush=True)
        for i, s in ex.map(run, [m for m in man if m["kind"] == "disease"]):
            print(i, f"{s:.0f}s", flush=True)
    for m in man:
        m.update(model=MODEL, file=f"img/{m['id']}.png", tier=TIER)
    json.dump(man, open(OUT / "manifest.json", "w"), indent=1)
    return man


def check(man, out=None):
    out = out or OUT
    sys.argv = [sys.argv[0]]
    ev = importlib.import_module("60_eval_clip")
    import joblib
    dev = device()
    base = ev.load_backend(types.SimpleNamespace(backend="openclip", pretrained="openai", model="ViT-B-16-quickgelu"), dev)
    paths = [str(out / m["file"]) for m in man]
    X = base.encode_images(paths).astype(np.float32)
    H = joblib.load(WORK / "heads_models.joblib")
    prob = H["type"].predict_proba(X); cls = np.array(H["type"].classes_)
    fst = np.clip(np.rint(H["fst"].predict(X)), 1, 6).astype(int)
    P, E = load_local_embeddings()
    train = set(pd.read_parquet(DATASET / "metadata/images.parquet", columns=["sha256", "split"]).query("split == 'train'").sha256)
    keep = P.sha256.isin(train).values
    E = E[keep].astype(np.float32)
    nn = np.max(np.concatenate([X @ E[s:s + 50000].T for s in range(0, len(E), 50000)], axis=1), axis=1)
    dx = ev.load_backend(types.SimpleNamespace(backend="openclip", pretrained=str(WORK / "models/clip_union_3k/model.pt"),
                                               model="ViT-B-16-quickgelu"), dev)
    Xd = dx.encode_images(paths).astype(np.float32)
    vocab = pd.read_csv(DATASET / "synthetic/plan/conditioning_table.csv").concept_name.tolist() + ["normal skin"]
    vocab = list(dict.fromkeys(vocab + [m["condition"] for m in man]))
    S = Xd @ ev.zs_classifier(dx, vocab)
    # nudity screen for the open tier (OpenAI CLIP zero-shot; any flag -> human review, never auto-published)
    labels = ["a clinical close-up photograph of skin on a face, arm, hand, leg, foot, back or chest",
              "a photograph showing exposed genitals or pubic area", "a photograph showing bare female breasts",
              "a photograph showing bare buttocks or the anal region"]
    T = base.encode_texts(labels)
    Z = 100 * X @ T.T
    Pn = np.exp(Z - Z.max(1, keepdims=True)); Pn /= Pn.sum(1, keepdims=True)
    p_exposed = Pn[:, 1:].sum(1)
    Tf = base.encode_texts(["a photograph in which a person's face is visible", "a close-up photograph of skin without any face"])
    Zf = 100 * X @ Tf.T
    p_face = np.exp(Zf[:, 0]) / np.exp(Zf).sum(1)
    res = {}
    for i, m in enumerate(man):
        order = np.argsort(-S[i])
        rank = int((S[i] > S[i, vocab.index(m["condition"])]).sum()) + 1
        res[m["id"]] = {"p_face": float(p_face[i]), "image_type": str(cls[prob[i].argmax()]), "p_clinical": float(prob[i][list(cls).index("clinical")]),
                        "fst_pred": int(fst[i]), "fst_ok": bool(abs(int(fst[i]) - m["fst"]) <= 1),
                        "nn_train_cos": float(nn[i]), "memorised": bool(nn[i] >= 0.98),
                        "dx_top3": [vocab[j] for j in order[:3]], "dx_rank": rank, "vocab_size": len(vocab),
                        "p_exposed": float(p_exposed[i]), "nudity_flag": bool(p_exposed[i] >= NUDITY_FLAG)}
    json.dump(res, open(out / "checks.json", "w"), indent=1)
    return res


ROMAN = {1: "I", 2: "II", 3: "III", 4: "IV", 5: "V", 6: "VI"}


def html(man, chk, out=None, title="Synthetic dermatology examples: healthy skin → projected disease", banner="",
         page="example_images.html", filter_nudity=True):
    out = out or OUT
    esc = lambda s: str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
    rv = json.load(open(out / "review.json")) if (out / "review.json").exists() else {}
    if filter_nudity:   # flagged images are held back unless a reviewer cleared them
        man = [m for m in man if not chk[m["id"]]["nudity_flag"] or str(rv.get(m["id"], {}).get("nudity", "")).startswith("cleared")]
    groups = []
    for b in [m for m in man if m["kind"] == "healthy"]:
        groups.append([b] + [m for m in man if m["kind"] == "disease" and m["base"] == b["id"]])
    chk_all = chk
    chk = {m["id"]: chk_all[m["id"]] for m in man}      # counts refer to the images shown on this page
    n_ok_fst = sum(c["fst_ok"] for c in chk.values()); n_clin = sum(c["image_type"] == "clinical" for c in chk.values())
    top3 = [m for m in man if m["kind"] == "disease" and chk[m["id"]]["dx_rank"] <= 3]
    n_dis = sum(m["kind"] == "disease" for m in man)

    def card(m):
        c = chk[m["id"]]
        badge = "healthy" if m["kind"] == "healthy" else "disease"
        ok = lambda b: '<span class="ok">✓</span>' if b else '<span class="no">✗</span>'
        rank = "" if m["kind"] == "healthy" else f'<li>Diagnosis rank {c["dx_rank"]} of {c["vocab_size"]} {ok(c["dx_rank"] <= 3)}</li>'
        return f'''<figure class="card {badge}">
  <a href="{m['file']}" target="_blank"><img src="{m['file']}" alt="{esc(m['condition'])}, Fitzpatrick {ROMAN[m['fst']]}, {esc(m['age'])} {esc(m['sex'])}, {esc(m['site'])}" loading="lazy"></a>
  <figcaption>
    <div class="cond">{esc(m['condition'].capitalize())}</div>
    {f'<div class="stage">{esc(m["stage"])}</div>' if m.get("stage") else ""}
    <div class="meta">Fitzpatrick {ROMAN[m['fst']]} · {esc(m['age'])} · {esc(m['sex'])} · {esc(m['site'])}</div>
    <ul class="chk">
      <li>Image type: {esc(c['image_type'])} {ok(c['image_type'] == 'clinical')}</li>
      <li>Fitzpatrick estimate {ROMAN[c['fst_pred']]} {ok(c['fst_ok'])}</li>
      {rank}
      <li>Top-3: {esc(', '.join(c['dx_top3']))}</li>
      <li>Closest real image cos {c['nn_train_cos']:.3f} {ok(not c['memorised'])}</li>
      <li>Nudity screen p = {c['p_exposed']:.2f} {ok(not c['nudity_flag']) if not c['nudity_flag'] else '<span class="no">flagged</span> → ' + esc(rv.get(m['id'], {}).get('nudity', 'held back'))}</li>
      {f'<li class="rv rv-{rv[m["id"]]["visual"]}">Preliminary visual review: <b>{esc(rv[m["id"]]["visual"])}</b>{(" – " + esc(rv[m["id"]]["note"])) if rv[m["id"]].get("note") else ""}</li>' if m["id"] in rv and rv[m["id"]].get("visual") else ""}
    </ul>
    <details><summary>Prompt</summary><p>{esc(m['prompt'])}</p></details>
  </figcaption>
</figure>'''
    rows = "\n".join(f'<section class="row"><h2>{esc(g[0]["id"])}: Fitzpatrick {ROMAN[g[0]["fst"]]}, {esc(g[0]["age"])} {esc(g[0]["sex"])}, '
                     f'{esc(g[0]["site"])}</h2><div class="cards">' + "".join(card(m) for m in g) + "</div></section>" for g in groups)
    doc = f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Synthetic Skin Examples</title>
<style>
:root {{ color-scheme: light; --page:#f9f9f7; --surface:#fcfcfb; --ink:#0b0b0b; --ink2:#52514e; --muted:#898781; --border:rgba(11,11,11,.10);
  --healthy:#1baf7a; --disease:#eb6834; --ok:#006300; --no:#b3261e; }}
@media (prefers-color-scheme: dark) {{ :root:where(:not([data-theme="light"])) {{ color-scheme: dark; --page:#0d0d0d; --surface:#1a1a19; --ink:#f0efec;
  --ink2:#c3c2b7; --muted:#898781; --border:rgba(255,255,255,.10); --healthy:#199e70; --disease:#d95926; --ok:#0ca30c; --no:#ff8a80; }} }}
* {{ box-sizing: border-box; }}
body {{ margin:0; background:var(--page); color:var(--ink); font:14px/1.45 system-ui,-apple-system,"Segoe UI",Arial,sans-serif; }}
header, main, footer {{ max-width:1500px; margin:0 auto; padding:16px 24px; }}
h1 {{ font-size:20px; margin:8px 0 4px; }} h2 {{ font-size:15px; margin:18px 0 8px; }}
.sub {{ color:var(--ink2); max-width:980px; margin:0; }}
.tiles {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); gap:10px; margin-top:14px; }}
.tile {{ background:var(--surface); border:1px solid var(--border); border-radius:8px; padding:10px 12px; }}
.tile .v {{ font-size:22px; font-weight:650; font-variant-numeric:tabular-nums; }} .tile .k {{ color:var(--ink2); font-size:12px; }}
.cards {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(260px,1fr)); gap:12px; }}
.card {{ margin:0; background:var(--surface); border:1px solid var(--border); border-radius:8px; overflow:hidden; border-top:4px solid var(--disease); }}
.card.healthy {{ border-top-color:var(--healthy); }}
.card img {{ display:block; width:100%; aspect-ratio:1; object-fit:cover; }}
figcaption {{ padding:8px 10px 10px; }}
.cond {{ font-weight:650; }} .stage {{ font-size:12.5px; color:var(--ink); font-style:italic; }} .meta {{ color:var(--ink2); font-size:12.5px; }}
.chk {{ list-style:none; padding:0; margin:6px 0 0; font-size:12.5px; color:var(--ink2); }} .chk li {{ margin:1px 0; }}
.banner {{ background:#fff3a0; color:#0b0b0b; border-radius:8px; padding:10px 12px; font-weight:600; }}
.rv {{ margin-top:4px !important; }} .rv-good b {{ color:var(--ok); }} .rv-weak b {{ color:#9a6700; }} .rv-fail b {{ color:var(--no); }}
.ok {{ color:var(--ok); font-weight:700; }} .no {{ color:var(--no); font-weight:700; }}
details {{ font-size:12px; color:var(--muted); margin-top:4px; }} details p {{ margin:4px 0 0; }}
footer {{ color:var(--ink2); font-size:12.5px; }}
@media (max-width:700px) {{ header, main, footer {{ padding-left:16px; padding-right:16px; }} }}
</style></head><body>
<header>
  {banner}
  <h1>{esc(title)}</h1>
  <p class="sub">Each row starts with a generated photograph of healthy skin (green bar) with a given Fitzpatrick type, age group, sex and body site.
  The following images (orange bar) are edits of that same photograph that add one skin condition, described as it looks on that skin type.
  Model: {MODEL} on the Splash image server. Below each image: our automatic checks (image-type and Fitzpatrick estimators of the dataset,
  nearest real training image, and the top-3 diagnoses of our dermatology CLIP among {chk[man[0]['id']]['vocab_size']} diagnoses).
  These are unreviewed research samples, not medical reference images.</p>
  <div class="tiles">
    <div class="tile"><div class="v">{len(man)}</div><div class="k">images ({len(groups)} healthy bases, {n_dis} disease edits)</div></div>
    <div class="tile"><div class="v">{n_clin}/{len(man)}</div><div class="k">classified as clinical photograph</div></div>
    <div class="tile"><div class="v">{n_ok_fst}/{len(man)}</div><div class="k">Fitzpatrick estimate within ±1 of target</div></div>
    <div class="tile"><div class="v">{len(top3)}/{n_dis}</div><div class="k">intended diagnosis in CLIP top-3</div></div>
    <div class="tile"><div class="v">{sum(c['memorised'] for c in chk.values())}</div><div class="k">near-duplicates of training images</div></div>
    <div class="tile"><div class="v">{sum(c['nudity_flag'] for c in chk.values())}</div><div class="k">flagged by the automatic nudity screen (shown only after review; see note per image)</div></div>
  </div>
</header>
<main>{rows}</main>
<footer><p>Generated {time.strftime('%Y-%m-%d')} with scripts/87_synthetic_examples.py. Checks are automatic plausibility signals; diagnostic
correctness needs dermatologist review (dataset/docs/SYNTHETIC_PROTOCOL.md).</p></footer>
</body></html>'''
    (out / page).write_text(doc)


def main():
    global API
    API = resolve_api()
    print("API", API, flush=True)
    man = generate()
    chk = check(man)
    html(man, chk)
    print("written", OUT / "example_images.html")


if __name__ == "__main__":
    main()
