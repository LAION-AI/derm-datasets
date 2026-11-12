"""SynthDerm-1M prompt plan from the lesion-lifecycle taxonomy (89_lesion_lifecycle.py): diagnosis -> variant -> stage,
crossed with Fitzpatrick type (I-VI, equal shares), age, sex and a typical body site of the variant. OPEN TIER ONLY.

Every prompt names diagnosis + subtype + exact site + stage, and the skin colour IN WORDS together with how the lesion
looks on that skin (the taxonomy's fst_I_II / fst_III_IV / fst_V_VI text). FLUX.2-klein ignores a bare Fitzpatrick
number and renders light skin (test 2026-10-14, work/lora_compare/skin_words/).

Open-tier rules (GOALS.md, 87/88): no anogenital, groin, buttock or breast sites (those belong to the gated clinician
tier, 88_synthetic_clinician.py); minors (infant / child / adolescent) only on face, arms or hands; non-sexual framing.
Each row also gets a training caption (for CLIP) that states diagnosis, subtype, stage, site, age group, sex and skin.

Writes dataset/synthetic/plan/lifecycle_prompts.parquet (+ _summary.json) and, with --pilot, a stratified pilot subset
lifecycle_pilot.parquet (diagnoses that map to a planned concept id, so 86_eval_synthetic.py can gate them).
Usage: python 94_lifecycle_prompts.py [--n 1000000] [--pilot-dx 24]
"""
import argparse
import difflib
import hashlib
import json
import re
import sys

import numpy as np
import pandas as pd

from common import DATASET

PLAN = DATASET / "synthetic/plan"
STYLE = ("Clinical dermatology photograph, natural skin texture with visible pores and fine vellus hair, even diffuse "
         "clinical lighting, sharp focus, realistic, no text, no watermark, no jewelry. Non-sexual medical photograph: "
         "no genitals and no nudity beyond the body part shown.")
ROMAN = {1: "I", 2: "II", 3: "III", 4: "IV", 5: "V", 6: "VI"}
SKIN = {1: "very fair, pale white skin with freckles", 2: "fair white skin", 3: "light olive to light brown skin",
        4: "moderate brown skin", 5: "dark brown skin", 6: "very dark brown, almost black, deeply pigmented skin"}
ANCESTRY = {1: ["Northern European", "Irish", "Scandinavian"], 2: ["Central European", "Eastern European", "North American"],
            3: ["Mediterranean", "Middle Eastern", "East Asian", "Latin American"],
            4: ["South Asian", "Southeast Asian", "Latin American", "North African"],
            5: ["East African", "South Asian", "Afro-Caribbean", "Afro-Latin American"],
            6: ["West African", "Central African", "Sudanese", "Melanesian"]}
FST_KEY = {1: "fst_I_II", 2: "fst_I_II", 3: "fst_III_IV", 4: "fst_III_IV", 5: "fst_V_VI", 6: "fst_V_VI"}
AGES = {"infant": (0, 1), "child": (2, 12), "adolescent": (13, 17), "adult": (18, 64), "elderly": (65, 90)}
MINOR = {"infant", "child", "adolescent"}
MINOR_SITES = {"face", "ear", "nose", "lip", "eyelid", "arm", "hand", "finger", "palm"}
# sites never shown in the open tier (gated clinician tier only)
# deliberately broad (drops a few harmless sites such as "labial commissures"); a second, LLM-based layer is in
# site_safety.json (classify_sites); a site is excluded if EITHER layer flags it
INTIMATE = re.compile(r"genit|penis|penile|glans|corona|coronal|coronary sulcus|prepuc|foreskin|frenul|shaft|meatus|"
                      r"urethr|testi|scrot|vulv|labia|labial|clitor|vestib|fourchette|introit|vagin|hymen|mons|pubic|pubis|"
                      r"perian|perine|anal\b|anal canal|anus|groin|inguin|crural|bathing|diaper|napkin|nappy|underwear|"
                      r"butt|glute|natal cleft|breast|nipple|areol|mamm|submammar|bikini|intimate|private", re.I)
# minors (open tier): face / arms / hands only; every body part named in the site text must be one of these
MINOR_BAD = re.compile(r"generali|widespread|diffuse|whole body|entire body|extremit|limb|trunk|torso|body|flexur|scalp|"
                       r"neck|axill|leg|thigh|knee|foot|feet|toe|sole|back|chest|abdom|flank|umbilic|fold|crease|"
                       r"popliteal|shin|calf|ankle|heel|hip|waist|shoulder|sacr|lumb", re.I)
SITE_SAFETY = PLAN / "site_safety.json"
# sites that imply a male subject (androgenetic baldness, beard area)
MALE_SITE = re.compile(r"bald|beard|moustache|mustache|barbae", re.I)
# visual-only lesion descriptions (98_visual_prompts.py), used instead of the taxonomy text where available
_VF = PLAN / "visual_descriptions.json"
VISUAL = {k: v for k, v in (json.load(open(_VF)) if _VF.exists() else {}).items() if "error" not in v}
# internal mucosa: not a skin photograph and rendered grotesquely by klein (pilot 2026-10-14) -> not generated
MUCOSA = re.compile(r"cavity|palat|gingiv|buccal|oral mucosa|tongue|septum|turbinate|conjunctiv|laryn|pharyn|oesoph|esoph|"
                    r"rectal|rectum|cervix|vagin|urethr|mucosa", re.I)
# trunk / proximal-limb sites: clothing is stated in the subject (end-of-prompt drape instructions were ignored)
CLOTHED = {"chest", "trunk", "abdomen", "axilla", "back", "leg"}
REALISM = ("The lesion is realistic and integrated into the surrounding skin with natural skin texture, true-to-life colour "
           "and subtle contrast; not exaggerated, not painted on, no drawn lines.")
NOUN = {("infant", "female"): "baby girl", ("infant", "male"): "baby boy", ("child", "female"): "girl",
        ("child", "male"): "boy", ("adolescent", "female"): "teenage girl", ("adolescent", "male"): "teenage boy",
        ("adult", "female"): "woman", ("adult", "male"): "man", ("elderly", "female"): "woman", ("elderly", "male"): "man"}

sys.path.insert(0, str(DATASET.parent / "scripts"))
_p85 = __import__("85_synthetic_plan")
site_of, SITES85 = _p85.site_of, _p85.SITES   # coarse body-site vocabulary of the plan


_UNSAFE = None


def site_ok(site):
    """Open tier: no intimate / anogenital / breast / buttock site (regex OR LLM verdict) and no internal mucosa."""
    global _UNSAFE
    if _UNSAFE is None:
        v = json.load(open(SITE_SAFETY)) if SITE_SAFETY.exists() else {}
        _UNSAFE = {k for k, x in v.items() if x.get("intimate") is not False}   # unknown / error = unsafe
    return not INTIMATE.search(site) and not MUCOSA.search(site) and site not in _UNSAFE


def site_all(text):
    """All coarse sites (85_synthetic_plan.SITES) named in a site description, not only the first."""
    t = text.lower()
    return {name for name, keys in SITES85 if any(re.search(r"\b" + k, t) for k in keys)}


def minor_ok(site):
    cs = site_all(site)
    return bool(cs) and cs <= MINOR_SITES and not MINOR_BAD.search(site)


def classify_sites(sites, base="http://localhost:4242/v1", model="incoai/Qwen3.8-27B-Splash"):
    """Second safety layer: the Splash LLM marks every unique site string that shows or is adjacent to genitals, the
    anogenital region, female breasts or buttocks (cached in site_safety.json; an error counts as unsafe)."""
    import urllib.request
    from concurrent.futures import ThreadPoolExecutor
    cache = json.load(open(SITE_SAFETY)) if SITE_SAFETY.exists() else {}
    todo = [x for x in sorted(set(sites)) if x not in cache or "error" in cache[x]]
    schema = {"type": "object", "properties": {"intimate": {"type": "boolean"}, "reason": {"type": "string"}},
              "required": ["intimate", "reason"]}

    def ask(site):
        q = (f"Body-site description from a dermatology taxonomy: \"{site}\".\nWould a photograph of this site show, or "
             "necessarily include, the genitals, the anogenital / perianal / perineal region, the groin crease, the pubic "
             "area, female breasts or nipples, or the buttocks / gluteal cleft? Answer intimate=true if yes or if unsure.")
        body = {"model": model, "reasoning_effort": "low", "temperature": 0, "max_tokens": 600,
                "messages": [{"role": "user", "content": q}],
                "response_format": {"type": "json_schema", "json_schema": {"name": "site", "schema": schema}}}
        req = urllib.request.Request(base + "/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                txt = json.load(r)["choices"][0]["message"]["content"]
            return site, json.loads(re.sub(r"<think>.*?</think>", "", txt or "", flags=re.S).strip())
        except (OSError, ValueError, KeyError, TypeError) as e:
            return site, {"intimate": True, "error": repr(e)[:200]}
    with ThreadPoolExecutor(2) as ex:
        for i, (k, v) in enumerate(ex.map(ask, todo)):
            cache[k] = v
            if (i + 1) % 100 == 0:
                print(i + 1, "of", len(todo), flush=True); json.dump(cache, open(SITE_SAFETY, "w"), indent=1, ensure_ascii=False)
    json.dump(cache, open(SITE_SAFETY, "w"), indent=1, ensure_ascii=False)
    return cache


def age_groups(typical):
    """Age groups whose range overlaps the taxonomy's typical ages ('20-40 years', 'childhood', 'over 60', ...)."""
    txt = " ".join(typical or []).lower()
    hit = set()
    if re.search(r"neonat|newborn|infan|birth|congenital", txt): hit.add("infant")
    if re.search(r"child|paediatric|pediatric|school", txt): hit.add("child")
    if re.search(r"adolesc|teen|puberty", txt): hit.add("adolescent")
    if re.search(r"elder|older|old age|senile", txt): hit.add("elderly")
    if re.search(r"adult", txt): hit.add("adult")
    for a, b in re.findall(r"(\d+)\s*[–\-to]+\s*(\d+)", txt):
        lo, hi = int(a), int(b)
        hit |= {g for g, (x, y) in AGES.items() if lo <= y and hi >= x}
    for a in re.findall(r"(?:over|>|above|after)\s*(\d+)", txt):
        hit |= {g for g, (x, y) in AGES.items() if y >= int(a)}
    return sorted(hit) or ["adult"]


def sex_weights(pred):
    p = (pred or "none").lower()
    if "female" in p: return ([1.0, 0.0] if "only" in p or "exclusive" in p else [0.7, 0.3])
    if "male" in p: return ([0.0, 1.0] if "only" in p or "exclusive" in p else [0.3, 0.7])
    return [0.5, 0.5]


def norm(s):
    s = re.sub(r"\(.*?\)", "", s.lower()).replace("/", " ").replace("-", " ")
    return " ".join(w[:-1] if w.endswith("s") and len(w) > 4 else w for w in re.findall(r"[a-z]+", s))


def concept_map(L):
    tab = pd.read_csv(PLAN / "conditioning_table.csv")
    by = {norm(n): int(c) for c, n in zip(tab.concept_id, tab.concept_name)}
    out = {}
    for k, v in L.items():
        for cand in [v.get("requested_name", k), k, v.get("diagnosis", "")] + re.findall(r"\((.*?)\)", v.get("diagnosis", "")):
            n = norm(cand)
            if n in by: out[k] = by[n]; break
        else:
            m = difflib.get_close_matches(norm(v.get("requested_name", k)), list(by), n=1, cutoff=0.88)
            out[k] = by[m[0]] if m else None
    return out


def seed_of(*parts):
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:8], 16)


def rows_for(key, e, n, rng, cid):
    """n prompts for one diagnosis, split equally over variant-stage pairs and Fitzpatrick types."""
    combos = [(va, st) for va in e["variants"] for st in va["stages"]]
    groups = age_groups(e.get("typical_age"))
    sw = sex_weights(e.get("sex_predilection"))
    out, skipped = [], 0
    for i in range(n):
        va, st = combos[i % len(combos)]
        fst = 1 + (i // len(combos)) % 6
        sites = [s for s in va["typical_sites"] if site_ok(s)]
        if not sites:
            skipped += 1; continue
        site = sites[rng.integers(len(sites))]
        coarse = site_of(site) or "other"
        ag = [g for g in groups if g not in MINOR or minor_ok(site)] or ["adult"]
        g = ag[rng.integers(len(ag))]
        lo, hi = AGES[g]; age = int(rng.integers(lo, hi + 1))
        sex = "female" if rng.random() < sw[0] else "male"
        if MALE_SITE.search(site):   # site-sex plausibility (expert feedback 2026-10-16: AK on a "bald scalp" of a woman)
            sex = "male"
        if g in MINOR and not minor_ok(site):   # defensive: the filter above must make this impossible
            raise SystemExit(f"open-tier rule violated: minor on {site!r}")
        anc = ANCESTRY[fst][rng.integers(len(ANCESTRY[fst]))]
        who = f"{age}-month-old {NOUN[(g, sex)]}" if g == "infant" and age == 0 else f"{max(age, 1)}-year-old {NOUN[(g, sex)]}"
        clothed = coarse in CLOTHED or coarse == "other"
        wear = ("" if not clothed else ", wearing a plain white sports bra and dark shorts" if sex == "female"
                else ", wearing dark shorts")
        view = "close-up of" if clothed or rng.random() < 0.7 else "regional clinical view of"
        stage = st["stage"]
        vis = VISUAL.get(f"{e['requested_name']}|{va['name']}|{stage}", {}).get(FST_KEY[fst])
        lesion = (vis if vis else f"{st['morphology']} On this {SKIN[fst].split(',')[0].replace(' skin', '')} skin: "
                                  f"{st[FST_KEY[fst]]}")
        prompt = (f"{STYLE} {view[0].upper() + view[1:]} the {site} of a {who}{wear}, of {anc} descent with {SKIN[fst]} "
                  f"(Fitzpatrick skin type {ROMAN[fst]}). Diagnosis: {e['requested_name']}, {va['name']}, stage: {stage}. "
                  f"{lesion} {REALISM}")
        caption = (f"{e['requested_name']}, {va['name']}, {stage} stage, {site}, {g} {sex}, "
                   f"Fitzpatrick {ROMAN[fst]} ({SKIN[fst]}).")
        out.append(dict(lifecycle_key=key, concept_id=cid, diagnosis=e["requested_name"], variant=va["name"], stage=stage,
                        fst_target=fst, age=age, age_group=g, sex=sex, site=site, site_coarse=coarse, ancestry=anc,
                        view=view.split(" of")[0], prompt=prompt, caption=caption, seed=seed_of(key, i), tier="open",
                        desc_source="visual" if vis else "taxonomy"))
    return out, skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1_000_000)
    ap.add_argument("--pilot-dx", type=int, default=24)
    ap.add_argument("--classify-sites", action="store_true", help="(re)run the LLM site-safety layer first (Splash server)")
    a = ap.parse_args()
    L = {k: v for k, v in json.load(open(PLAN / "lesion_lifecycle.json")).items() if v.get("variants")}
    if a.classify_sites:
        allsites = [s for e in L.values() for va in e["variants"] for s in va["typical_sites"]]
        c = classify_sites([s for s in allsites if not INTIMATE.search(s) and not MUCOSA.search(s)])
        print("LLM-flagged intimate sites:", sorted(k for k, v in c.items() if v.get("intimate") is not False))
    cmap = concept_map(L)
    per = a.n // len(L)
    rng = np.random.default_rng(20261014)
    rows, skipped = [], {}
    for k, e in L.items():
        r, s = rows_for(k, e, per, rng, cmap[k])
        rows += r
        if s: skipped[k] = s
    P = pd.DataFrame(rows)
    P.insert(0, "synth_id", [f"sd_{i:07d}" for i in range(len(P))])
    P["concept_id"] = P.concept_id.astype("Int64")
    bad = P[P.age_group.isin(MINOR) & ~P.site.map(minor_ok)]
    assert bad.empty and P.site.map(site_ok).all(), "open-tier rule violated"
    P.to_parquet(PLAN / "lifecycle_prompts.parquet", index=False)
    summ = dict(rows=len(P), diagnoses=int(P.lifecycle_key.nunique()), per_diagnosis=per,
                variant_stage_pairs=int(P.groupby(["lifecycle_key", "variant", "stage"]).ngroups),
                gate_mapped_diagnoses=int(P.drop_duplicates("lifecycle_key").concept_id.notna().sum()),
                unmapped=sorted(k for k, c in cmap.items() if c is None),
                rows_skipped_intimate_only=skipped, by_fst=P.fst_target.value_counts().sort_index().to_dict(),
                by_age_group=P.age_group.value_counts().to_dict(), by_sex=P.sex.value_counts().to_dict(),
                prompt_chars_median=int(P.prompt.str.len().median()), prompt_chars_max=int(P.prompt.str.len().max()))
    json.dump(summ, open(PLAN / "lifecycle_prompts_summary.json", "w"), indent=1, default=int)
    print(json.dumps({k: v for k, v in summ.items() if k != "unmapped"}, indent=1, default=int))
    print("unmapped (no gate diagnosis check):", len(summ["unmapped"]))

    # pilot: diverse mapped diagnoses, one random variant-stage per Fitzpatrick type
    keys = sorted(P[P.concept_id.notna()].lifecycle_key.unique())
    must = [k for k in ["atopic dermatitis", "psoriasis", "acne vulgaris", "melanoma", "vitiligo", "scabies"] if k in keys]
    rest = [k for k in keys if k not in must]
    pick = must + list(np.random.default_rng(7).choice(rest, size=max(0, min(a.pilot_dx, len(keys)) - len(must)), replace=False))
    pil = (P[P.lifecycle_key.isin(pick)].sample(frac=1.0, random_state=7)
           .drop_duplicates(["lifecycle_key", "fst_target"]).sort_values(["lifecycle_key", "fst_target"]))
    pil.to_parquet(PLAN / "lifecycle_pilot.parquet", index=False)
    print("pilot", len(pil), "rows,", pil.lifecycle_key.nunique(), "diagnoses:", ", ".join(sorted(pick)))


if __name__ == "__main__":
    main()
