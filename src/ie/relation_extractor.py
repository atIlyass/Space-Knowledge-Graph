"""
src/ie/relation_extractor.py
Dependency-parse heuristics to extract (subject, predicate, object) triples.

Strategy
--------
For each sentence, use spaCy's dependency tree to find:
  1. nsubj + ROOT-verb + dobj   →  (subj, verb, obj)
  2. nsubj + ROOT-verb + prep + pobj  → (subj, "verb_prep", pobj)

Filters
-------
- Subject and object must be Named Entities (from NER)
- Verb must not be a copula or auxiliary
- Generic nouns / pronouns are discarded

Usage:
    from src.ie.relation_extractor import extract_relations
    triples = extract_relations(text)

CLI smoke test:
    python -m src.ie.relation_extractor
"""

import re
from typing import NamedTuple

import spacy

# Verbs to skip - too generic to be useful predicates
SKIP_VERBS = {
    "be", "have", "do", "say", "make", "get", "go", "come",
    "know", "think", "see", "use", "find", "give", "take",
    "include", "contain", "become",
}

_NLP = None


def _get_nlp():
    global _NLP
    if _NLP is None:
        _NLP = spacy.load("en_core_web_sm")
    return _NLP


class Triple(NamedTuple):
    subject: str
    predicate: str
    obj: str
    subject_label: str  # NER label of subject
    object_label: str   # NER label of object
    sentence: str


def _ner_map(doc) -> dict[int, str]:
    """Build token-index → NER-label mapping."""
    mapping: dict[int, str] = {}
    for ent in doc.ents:
        for tok in ent:
            mapping[tok.i] = ent.label_
    return mapping


def _entity_span_text(token) -> str:
    """Return the full entity span text for a token if it belongs to one."""
    if token.ent_iob_ in ("B", "I") and token.ent_type_:
        # Walk siblings to get full span
        doc = token.doc
        start = token.i
        end = token.i + 1
        # expand left
        while start > 0 and doc[start - 1].ent_iob_ == "I":
            start -= 1
        # expand right
        while end < len(doc) and doc[end].ent_iob_ == "I":
            end += 1
        return doc[start:end].text.strip()
    return token.text.strip()


def _is_useful_entity(text: str) -> bool:
    """Discard pronouns / very short strings."""
    if len(text) < 2:
        return False
    if text.lower() in {
        "he", "she", "it", "they", "we", "i", "you",
        "this", "that", "these", "those", "who", "which",
    }:
        return False
    # Reject purely numeric
    if re.fullmatch(r"[\d\s,.%-]+", text):
        return False
    return True


def _verb_label(root, prep_child=None) -> str:
    """Build readable predicate string from verb (+ optional preposition)."""
    verb = root.lemma_.lower()
    if prep_child:
        return f"{verb}_{prep_child.lower()}"
    return verb


def extract_relations(text: str) -> list[Triple]:
    """
    Extract (subject, predicate, object) triples via dependency parsing.

    Parameters
    ----------
    text : plain text (may be multi-sentence)

    Returns
    -------
    list of Triple namedtuples
    """
    nlp = _get_nlp()
    doc = nlp(text)
    ner_map = _ner_map(doc)

    triples: list[Triple] = []
    seen: set[tuple] = set()

    for sent in doc.sents:
        # Find the root verb of the sentence
        root = None
        for tok in sent:
            if tok.dep_ == "ROOT" and tok.pos_ == "VERB":
                root = tok
                break
        if root is None:
            continue
        if root.lemma_.lower() in SKIP_VERBS:
            continue

        # Collect nsubj(s)
        subjects = [
            t for t in root.children if t.dep_ in ("nsubj", "nsubjpass")
        ]
        # Collect dobj(s)
        dobjs = [t for t in root.children if t.dep_ == "dobj"]
        # Collect prep + pobj combos
        preps = [t for t in root.children if t.dep_ == "prep"]

        for subj_tok in subjects:
            subj_text = _entity_span_text(subj_tok)
            subj_label = ner_map.get(subj_tok.i, "")
            if not _is_useful_entity(subj_text):
                continue
            if not subj_label:
                continue  # require subject to be a named entity

            # Pattern 1: subj → verb → dobj
            for dobj_tok in dobjs:
                obj_text = _entity_span_text(dobj_tok)
                obj_label = ner_map.get(dobj_tok.i, "")
                if not _is_useful_entity(obj_text):
                    continue
                if not obj_label:
                    continue
                pred = _verb_label(root)
                key = (subj_text, pred, obj_text)
                if key not in seen:
                    seen.add(key)
                    triples.append(
                        Triple(subj_text, pred, obj_text, subj_label, obj_label, sent.text.strip())
                    )

            # Pattern 2: subj → verb → prep → pobj
            for prep_tok in preps:
                for pobj_tok in prep_tok.children:
                    if pobj_tok.dep_ != "pobj":
                        continue
                    obj_text = _entity_span_text(pobj_tok)
                    obj_label = ner_map.get(pobj_tok.i, "")
                    if not _is_useful_entity(obj_text):
                        continue
                    if not obj_label:
                        continue
                    pred = _verb_label(root, prep_tok.text)
                    key = (subj_text, pred, obj_text)
                    if key not in seen:
                        seen.add(key)
                        triples.append(
                            Triple(subj_text, pred, obj_text, subj_label, obj_label, sent.text.strip())
                        )

    return triples


if __name__ == "__main__":
    sample = (
        "Neil Armstrong landed on the Moon during the Apollo 11 mission. "
        "NASA operates the International Space Station with ESA. "
        "Elon Musk founded SpaceX in California."
    )
    for t in extract_relations(sample):
        print(f"({t.subject}) --[{t.predicate}]--> ({t.obj})")
