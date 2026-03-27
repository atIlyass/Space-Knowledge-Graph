"""
src/crawl/filter.py
Utility: decide whether a crawled page is "useful" (rich enough in content).
"""

import re


def word_count(text: str) -> int:
    """Return approximate word count for a text string."""
    return len(re.findall(r"\b\w+\b", text))


def is_useful(text: str, min_words: int = 500) -> bool:
    """
    Return True when the page text is long enough to be worth keeping.

    Parameters
    ----------
    text      : cleaned plain-text extracted from a page
    min_words : minimum number of words required (default 500)
    """
    if not text or not text.strip():
        return False
    return word_count(text) >= min_words


if __name__ == "__main__":
    sample = "word " * 600
    print(f"Sample ({word_count(sample)} words) useful:", is_useful(sample))
    short = "just a few words"
    print(f"Short ({word_count(short)} words) useful:", is_useful(short))
