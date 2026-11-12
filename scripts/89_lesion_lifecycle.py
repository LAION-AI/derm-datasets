"""'Life of lesions' taxonomy for SynthDerm-1M: diagnosis -> clinical variants (with typical body sites) -> stages,
with morphology per stage and its appearance on Fitzpatrick I-II / III-IV / V-VI. Drafted by an LLM (Splash server,
incoai/Qwen3.8-27B-Splash, JSON-schema constrained, reasoning effort low); every entry is marked
"llm_draft - dermatologist review pending" and is the input for prompt generation, not a reference text.

Diagnosis list (canonical, duplicates merged): Fitzpatrick17k classes, HAM10000 / PAD-UFES-20 / Daffodil classes,
the 15 GBD 2019 skin conditions + keratinocyte carcinomas, conditions mainly seen on skin of colour, and anogenital
conditions (clinician tier). Writes dataset/synthetic/plan/lesion_lifecycle.json (resumable cache).
"""
import json
import os
import re
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from common import DATASET

OUT = DATASET / "synthetic/plan/lesion_lifecycle.json"
BASE = os.environ.get("LLM_BASE_URL", "http://localhost:4242/v1")
MODEL = os.environ.get("LLM_MODEL", "incoai/Qwen3.8-27B-Splash")

F17K = json.load(open(os.path.join(os.path.dirname(__file__), "derm1m_zeroshot_meta.json")))["F17K_DISEASE_113_CLASSES"]
MERGE = {"acne": "acne vulgaris", "malignant melanoma": "melanoma", "superficial spreading melanoma ssm": "melanoma",
         "granuloma pyogenic": "pyogenic granuloma", "basal cell carcinoma morpheiform": "basal cell carcinoma",
         "solid cystic basal cell carcinoma": "basal cell carcinoma", "nevocytic nevus": "melanocytic nevus",
         "acquired autoimmune bullous diseaseherpes gestationis": "pemphigoid gestationis", "dariers disease": "darier disease",
         "behcets disease": "behcet disease", "hidradenitis": "hidradenitis suppurativa", "eczema": "nummular eczema",
         "papilomatosis confluentes and reticulate": "confluent and reticulated papillomatosis",
         "erythema annulare centrifigum": "erythema annulare centrifugum", "sun damaged skin": "chronic actinic damage",
         "stevens johnson syndrome": "stevens-johnson syndrome / toxic epidermal necrolysis", "porokeratosis actinic": "disseminated superficial actinic porokeratosis",
         "disseminated actinic porokeratosis": "disseminated superficial actinic porokeratosis", "lupus subacute": "subacute cutaneous lupus erythematosus",
         "lupus erythematosus": "discoid lupus erythematosus", "fixed eruptions": "fixed drug eruption", "pediculosis lids": "pediculosis",
         "nematode infection": "cutaneous larva migrans", "stasis edema": "stasis dermatitis", "neurodermatitis": "lichen simplex chronicus",
         "lichen simplex": "lichen simplex chronicus"}
EXTRA = [  # benchmark classes, GBD conditions, skin of colour, anogenital (clinician tier)
    "benign keratosis (lichen planus-like keratosis)", "vascular lesions (cherry angioma, angiokeratoma)", "nail psoriasis",
    "hyperpigmentation (post-inflammatory)", "atopic dermatitis", "irritant contact dermatitis", "tinea corporis", "tinea pedis",
    "tinea capitis", "tinea cruris", "onychomycosis", "pityriasis versicolor", "cutaneous candidiasis", "verruca vulgaris",
    "plantar warts", "molluscum contagiosum", "impetigo", "furunculosis", "cellulitis / erysipelas", "pressure ulcer",
    "alopecia areata", "chronic pruritus (no primary skin lesion)", "herpes zoster", "herpes simplex labialis", "varicella",
    "melasma", "dermatosis papulosa nigra", "pseudofolliculitis barbae", "acne keloidalis nuchae", "traction alopecia",
    "central centrifugal cicatricial alopecia", "pityriasis alba", "idiopathic guttate hypomelanosis", "acral lentiginous melanoma",
    "leprosy", "cutaneous leishmaniasis", "mycetoma", "secondary syphilis", "venous leg ulcer", "diabetic foot ulcer",
    "condyloma acuminatum", "primary syphilis (chancre)", "genital herpes", "lichen sclerosus (anogenital)", "balanitis",
    "genital psoriasis", "extramammary paget disease", "pearly penile papules (normal variant)", "vulvar lichen planus",
    "intertrigo", "perianal dermatitis", "anal fissure (perianal skin)", "scabies nodules (genital)"]
CLINICIAN = {"condyloma acuminatum", "primary syphilis (chancre)", "genital herpes", "lichen sclerosus (anogenital)", "balanitis",
             "genital psoriasis", "extramammary paget disease", "pearly penile papules (normal variant)", "vulvar lichen planus",
             "perianal dermatitis", "anal fissure (perianal skin)", "scabies nodules (genital)", "fordyce spots"}

SCHEMA = {"type": "object", "properties": {
    "diagnosis": {"type": "string"}, "category": {"type": "string"},
    "typical_age": {"type": "array", "items": {"type": "string"}},
    "sex_predilection": {"type": "string", "enum": ["none", "female", "male"]},
    "anogenital_or_breast_sites_needed": {"type": "boolean"},
    "variants": {"type": "array", "maxItems": 4, "items": {"type": "object", "properties": {
        "name": {"type": "string"}, "typical_sites": {"type": "array", "items": {"type": "string"}},
        "stages": {"type": "array", "maxItems": 4, "items": {"type": "object", "properties": {
            "stage": {"type": "string"}, "morphology": {"type": "string"},
            "fst_I_II": {"type": "string"}, "fst_III_IV": {"type": "string"}, "fst_V_VI": {"type": "string"}},
            "required": ["stage", "morphology", "fst_I_II", "fst_III_IV", "fst_V_VI"]}}},
        "required": ["name", "typical_sites", "stages"]}},
    "key_differentials": {"type": "array", "items": {"type": "string"}}},
    "required": ["diagnosis", "category", "typical_age", "sex_predilection", "anogenital_or_breast_sites_needed", "variants", "key_differentials"]}

PROMPT = """You are a board-certified dermatologist writing a structured reference for generating realistic clinical photographs.
For the diagnosis "{dx}", describe:
- up to 4 clinically important variants / subtypes (e.g. nodular, superficial, sclerodermiform basal cell carcinoma), each with its typical body sites (specific anatomical sites, e.g. "nasal ala", "ear helix", "antecubital fossa");
- for each variant up to 4 stages of the lesion's natural course and treatment ("life of the lesion"), choosing what applies: early / incipient, acute / active, chronic / established, resolving / healing, treated (residual changes such as post-inflammatory hyperpigmentation or scarring), complicated (e.g. superinfected, ulcerated);
- for each stage the visible morphology (primary and secondary lesions, size, colour, surface, border, distribution) in precise dermatological terms, and how it looks on Fitzpatrick skin types I-II, III-IV and V-VI (e.g. erythema appears violaceous, grey or dark brown on dark skin; post-inflammatory hyperpigmentation is stronger on dark skin).
Be accurate and concise (one or two sentences per field). Use the anatomical site where the condition really occurs."""


def ask(dx):
    body = {"model": MODEL, "reasoning_effort": "low", "temperature": 0, "max_tokens": 6000,
            "messages": [{"role": "user", "content": PROMPT.format(dx=dx)}],
            "response_format": {"type": "json_schema", "json_schema": {"name": "lesion_lifecycle", "schema": SCHEMA}}}
    req = urllib.request.Request(BASE + "/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    for k in range(3):
        try:
            with urllib.request.urlopen(req, timeout=900) as r:
                txt = json.load(r)["choices"][0]["message"]["content"]
            txt = re.sub(r"<think>.*?</think>", "", txt or "", flags=re.S).strip()
            return json.loads(txt)
        except (OSError, ValueError, TypeError) as e:
            err = e
    return {"error": repr(err)[:200]}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--add", help="text file with one extra diagnosis per line (taxonomy extension)")
    ap.add_argument("--out", default=str(OUT), help="output / cache JSON (use a separate file for extensions)")
    a = ap.parse_args()
    out = __import__("pathlib").Path(a.out)
    names = list(dict.fromkeys([MERGE.get(x, x) for x in F17K] + EXTRA)) if not a.add else \
        [l.strip() for l in open(a.add) if l.strip() and not l.startswith("#")]
    cache = json.load(open(out)) if out.exists() else {}
    todo = [n for n in names if n not in cache or "error" in cache[n]]
    print(len(names), "diagnoses,", len(todo), "to draft", flush=True)

    def run(n):
        r = ask(n)
        r["requested_name"] = n
        r["tier"] = "clinician" if n in CLINICIAN or r.get("anogenital_or_breast_sites_needed") else "open"
        r["status"] = "llm_draft - dermatologist review pending"
        r["model"] = MODEL
        return n, r
    with ThreadPoolExecutor(3) as ex:
        for i, (n, r) in enumerate(ex.map(run, todo)):
            cache[n] = r
            print(i + 1, n, "variants", len(r.get("variants", [])), "error" if "error" in r else "", flush=True)
            if (i + 1) % 10 == 0:
                json.dump(cache, open(out, "w"), indent=1, ensure_ascii=False)
    json.dump(cache, open(out, "w"), indent=1, ensure_ascii=False)
    nv = sum(len(r.get("variants", [])) for r in cache.values())
    ns = sum(len(v.get("stages", [])) for r in cache.values() for v in r.get("variants", []))
    print(len(cache), "diagnoses,", nv, "variants,", ns, "variant-stage combinations; errors:",
          [n for n, r in cache.items() if "error" in r])


if __name__ == "__main__":
    main()
