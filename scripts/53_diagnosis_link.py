"""Normalise diagnosis labels of our dataset and Derm1M into shared concepts (SapBERT).

1. Vocabulary = all distinct diagnosis strings of
     - our atlases (source label; for non-English atlases the English LLM-structured
       diagnosis from output/meta-opus.jsonl where available),
     - Derm1M disease_label (comma-separated lists split),
   each lower-cased and stripped of trailing "(1 of 3)" counters.
2. SapBERT [CLS] embeddings; leader clustering at cosine >= 0.95 (merges spelling variants only;
   calibrated: distinct diseases such as pemphigus vulgaris/foliaceus reach 0.945) -> concept_id.
   Each concept is also linked to the nearest Derm1M
   ontology node when cosine >= 0.80.
3. English journal captions are scanned for vocabulary phrases (longest match,
   >= 5 characters, word boundaries); matched phrases -> concepts.

Writes work/dx_vocab.parquet (term, concept_id, concept_name, ontology_node, sim),
       work/dx_samples_ours.parquet (src_path, concept_ids),
       work/dx_samples_derm1m.parquet (src_path, concept_ids).
"""
import json
import re

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

from common import EXTERNAL, WORK, device

MODEL = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
GENERIC = {"no definitive diagnosis", "unknown", "normal skin", "skin", "rash", "lesion", "lesions", "figure",
           "dermatitis", "other", "none", "case", "patient", "tumor", "tumour", "disease", "ulcer", "nodule",
           "plaque", "papule", "macule", "erythema", "scar", "keratosis", "no diagnosis"}


def norm(s):
    """Lower-case and strip atlas page artefacts ("... images", "(overview)", "(1 of 3)", "affecting the
    face", "pathology", "--") so that label noise is not counted as distinct conditions."""
    if not isinstance(s, str):
        return None
    s = s.lower().strip()
    s = re.sub(r"\(?\b\d+\s*(of|von|de)\s*\d+\)?", "", s)        # "(1 of 3)"
    s = re.sub(r"\([^)]*\)?", "", s)                               # parenthetical synonyms / notes
    s = re.sub(r"\s+affecting\b.*$", "", s)
    s = re.sub(r"\b(images?|pictures?|photos?|pathology)\b", "", s)
    s = re.sub(r"\s*\(\d+\)$", "", s)
    s = re.sub(r"[_\"'`´’]+", " ", s)
    s = re.sub(r"-{2,}", " ", s)
    s = re.sub(r"\s+", " ", s).strip(" .,;:-()")
    return s or None


def embed(terms, bs=256):
    tok = AutoTokenizer.from_pretrained(MODEL)
    m = AutoModel.from_pretrained(MODEL).eval()
    dev = device()
    m = m.to(dev)
    out = []
    with torch.no_grad():
        for i in range(0, len(terms), bs):
            x = tok(terms[i:i + bs], padding=True, truncation=True, max_length=32, return_tensors="pt").to(dev)
            e = m(**x).last_hidden_state[:, 0]
            out.append(torch.nn.functional.normalize(e, dim=-1).cpu().numpy())
    return np.concatenate(out).astype(np.float32)


def greedy_cluster(E, freq, terms, thr=0.95, typo_ratio=92, typo_sim=0.80):
    """Leader clustering in descending frequency: a term joins the most similar leader if SapBERT
    cos >= thr (calibrated: distinct diseases such as pemphigus vulgaris/foliaceus reach 0.945),
    or if it is a spelling variant (character ratio >= typo_ratio and cos >= typo_sim, e.g.
    'psoriaisis' -> 'psoriasis'); otherwise it becomes a new leader (the concept name)."""
    from rapidfuzz import fuzz
    order = np.argsort(-freq)
    leaders, lab = [], np.full(len(E), -1)
    L = np.zeros((0, E.shape[1]), dtype=np.float32)
    for i in order:
        if len(leaders):
            s = L @ E[i]
            top = np.argsort(-s)[:5]
            j = int(top[0])
            if s[j] >= thr:
                lab[i] = j
                continue
            hit = next((int(t) for t in top if s[t] >= typo_sim and len(terms[i]) >= 6 and
                        fuzz.ratio(terms[i], terms[leaders[t]]) >= typo_ratio), None)
            if hit is not None:
                lab[i] = hit
                continue
        leaders.append(i)
        L = np.vstack([L, E[i][None]])
        lab[i] = len(leaders) - 1
    return lab, np.array(leaders)


def main():
    atl = pd.read_parquet(WORK / "atlases.parquet", columns=["src_path", "source", "diagnosis", "language",
                                                            "llm_text_structured"])

    def atlas_terms(r):
        if r.language != "en" and isinstance(r.llm_text_structured, str):
            d = json.loads(r.llm_text_structured).get("diagnoses") or []
            if d:
                return [norm(x) for x in d if norm(x)]
        t = norm(r.diagnosis)
        return [t] if t else []
    atl["terms"] = [atlas_terms(r) for r in atl.itertuples()]

    d1 = pd.read_parquet(WORK / "derm1m.parquet", columns=["src_path", "disease_label", "source", "caption"])
    d1["terms"] = d1.disease_label.fillna("").map(lambda s: [norm(x) for x in s.split(",") if norm(x)])

    cnt = pd.concat([atl.terms.explode(), d1.terms.explode()]).dropna().value_counts()
    vocab = cnt[(~cnt.index.isin(GENERIC)) & (cnt.index.str.len() >= 3)]
    terms = list(vocab.index)
    print("vocabulary terms:", len(terms))
    E = embed(terms)
    lab, leaders = greedy_cluster(E, vocab.values.astype(float), terms)
    onto = json.load(open(EXTERNAL / "Derm1M-hf/ontology.json"))
    nodes = [k for k in onto if k != "root"]
    NE = embed([n.lower() for n in nodes])
    sim = E @ NE.T
    best = sim.argmax(1)
    V = pd.DataFrame({"term": terms, "freq": vocab.values, "concept_id": lab,
                      "concept_name": [terms[leaders[c]] for c in lab],
                      "ontology_node": [nodes[b] if sim[i, b] >= 0.80 else None for i, b in enumerate(best)],
                      "ontology_sim": sim[np.arange(len(terms)), best]})
    # a concept's ontology node = that of its leader term
    lead_node = dict(zip(V.concept_id[V.term.isin([terms[l] for l in leaders])],
                         V.ontology_node[V.term.isin([terms[l] for l in leaders])]))
    V["concept_ontology_node"] = V.concept_id.map(lead_node)
    V.to_parquet(WORK / "dx_vocab.parquet", index=False)
    np.save(WORK / "dx_vocab_sapbert.npy", E)
    t2c = dict(zip(V.term, V.concept_id))

    atl["concepts"] = atl.terms.map(lambda ts: sorted({t2c[t] for t in ts if t in t2c}))
    d1["concepts"] = d1.terms.map(lambda ts: sorted({t2c[t] for t in ts if t in t2c}))

    # phrase matching in English journal captions (and textbook? German -> skipped)
    jr = pd.read_parquet(WORK / "journals.parquet", columns=["src_path", "source", "caption", "language"])
    jr = jr[jr.language.eq("en") & jr.caption.notna()]
    phrases = {t: c for t, c in t2c.items() if len(t) >= 5 and t not in GENERIC}
    maxw = max(len(t.split()) for t in phrases)
    tokre = re.compile(r"[a-z0-9'\-]+")

    def match(text):
        w = tokre.findall(text.lower())
        found, i = set(), 0
        while i < len(w):
            hit = None
            for n in range(min(maxw, len(w) - i), 0, -1):
                p = " ".join(w[i:i + n])
                if p in phrases:
                    hit = (p, n)
                    break
            if hit:
                found.add(phrases[hit[0]])
                i += hit[1]
            else:
                i += 1
        return sorted(found)
    jr["concepts"] = jr.caption.map(match)
    ours = pd.concat([atl[["src_path", "source", "concepts"]], jr[["src_path", "source", "concepts"]]])
    ours.to_parquet(WORK / "dx_samples_ours.parquet", index=False)
    d1[["src_path", "source", "concepts"]].to_parquet(WORK / "dx_samples_derm1m.parquet", index=False)
    print("concepts:", V.concept_id.nunique(), "| ours with >=1 concept:", (ours.concepts.str.len() > 0).mean().round(3),
          "| derm1m with >=1 concept:", (d1.concepts.str.len() > 0).mean().round(3))


if __name__ == "__main__":
    main()
