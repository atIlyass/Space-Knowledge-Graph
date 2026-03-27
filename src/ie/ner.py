"""
src/ie/ner.py
spaCy-based Named Entity Recognition for the space domain.

Extracts entities with labels: PERSON, ORG, GPE, DATE
Filters out stopwords and very short tokens.

Usage (as library):
    from src.ie.ner import extract_entities
    entities = extract_entities(text)

Usage (CLI smoke test):
    python -m src.ie.ner
"""

import re
from typing import NamedTuple

import spacy

# Labels we care about for the space domain
TARGET_LABELS = {"PERSON", "ORG", "GPE", "DATE", "LOC", "FAC", "PRODUCT", "EVENT"}

# Generic tokens to discard even if tagged by spaCy
STOPWORDS = {
    "the", "a", "an", "this", "that", "these", "those",
    "he", "she", "it", "they", "we", "i", "you",
    "mr", "ms", "dr", "prof",
    "one", "two", "three", "many", "some", "other",
    "first", "last", "new", "old", "same", "great", "large", "small",
}

_NLP = None  # lazy-load model


def _get_nlp():
    global _NLP
    if _NLP is None:
        _NLP = spacy.load("en_core_web_sm")
    return _NLP


class Entity(NamedTuple):
    text: str
    label: str
    start_char: int
    end_char: int
    sentence: str


def _clean_text(text: str) -> str:
    """Normalise whitespace; remove problematic Unicode."""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def is_valid_entity(ent_text: str, label: str) -> bool:
    """Return False for noisy or too-generic entities."""
    t = ent_text.strip()
    if len(t) < 2:
        return False
    if t.lower() in STOPWORDS:
        return False
    # Skip purely numeric strings (except dates where digits are expected)
    if label != "DATE" and re.fullmatch(r"[\d\s,.%-]+", t):
        return False
    return True


def extract_entities(text: str) -> list[Entity]:
    """
    Run spaCy NER on text and return a deduplicated list of Entity objects.

    Parameters
    ----------
    text : raw page text

    Returns
    -------
    list of Entity namedtuples
    """
    nlp = _get_nlp()
    doc = nlp(_clean_text(text))

    seen: set[tuple] = set()
    entities: list[Entity] = []

    for sent in doc.sents:
        for ent in sent.ents:
            if ent.label_ not in TARGET_LABELS:
                continue
            if not is_valid_entity(ent.text, ent.label_):
                continue
            key = (ent.text.strip(), ent.label_)
            if key in seen:
                continue
            seen.add(key)
            entities.append(
                Entity(
                    text=ent.text.strip(),
                    label=ent.label_,
                    start_char=ent.start_char,
                    end_char=ent.end_char,
                    sentence=sent.text.strip(),
                )
            )

    return entities


if __name__ == "__main__":
    sample = (
        "Neil Armstrong, an American astronaut, became the first person to walk "
        "on the Moon on July 20, 1969, during NASA's Apollo 11 mission. "
        "SpaceX, founded by Elon Musk, launched Crew Dragon from Cape Canaveral."
    )
    for e in extract_entities(sample):
        print(f"[{e.label:8s}] {e.text}")
