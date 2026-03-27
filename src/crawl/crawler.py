"""
src/crawl/crawler.py
Fetches seed URLs, extracts main content with trafilatura,
filters short pages, and saves crawler_output.jsonl.

Usage:
    python -m src.crawl.crawler
    python -m src.crawl.crawler --seed data/raw/seed_urls.txt --out data/processed/crawler_output.jsonl
"""

import argparse
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
import trafilatura

from src.crawl.filter import is_useful, word_count

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Polite crawl settings
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; SpaceKGBot/1.0; "
        "+https://github.com/student/space-kg; academic research)"
    )
}
REQUEST_TIMEOUT = 15  # seconds
CRAWL_DELAY = 1.5     # seconds between requests (polite)


def fetch_and_extract(url: str) -> dict | None:
    """
    Fetch a URL, extract main text with trafilatura, return a dict or None.

    Returns dict with keys:
        url, title, text, word_count, crawled_at
    """
    try:
        resp = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
    except Exception as exc:
        logger.warning("Failed to fetch %s: %s", url, exc)
        return None

    downloaded = resp.text
    text = trafilatura.extract(
        downloaded,
        include_comments=False,
        include_tables=False,
        no_fallback=False,
    )
    if not text:
        logger.warning("trafilatura returned nothing for %s", url)
        return None

    # Simple title extraction from the URL slug
    title = url.rstrip("/").split("/")[-1].replace("_", " ")

    return {
        "url": url,
        "title": title,
        "text": text,
        "word_count": word_count(text),
        "crawled_at": datetime.now(timezone.utc).isoformat(),
    }


def crawl(seed_path: str, out_path: str, min_words: int = 500) -> list[dict]:
    """
    Main crawl loop.

    Returns list of kept page records.
    """
    seed_file = Path(seed_path)
    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    urls = [
        line.strip()
        for line in seed_file.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    logger.info("Crawling %d seed URLs …", len(urls))

    kept = []
    with out_file.open("w", encoding="utf-8") as fh:
        for url in urls:
            logger.info("Fetching: %s", url)
            record = fetch_and_extract(url)
            if record and is_useful(record["text"], min_words=min_words):
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                kept.append(record)
                logger.info(
                    "   kept %d words", record["word_count"]
                )
            else:
                wc = record["word_count"] if record else 0
                logger.info("  skipped %d words", wc)
            time.sleep(CRAWL_DELAY)

    logger.info(
        "Done. %d / %d pages kept → %s", len(kept), len(urls), out_file
    )
    return kept


def main() -> None:
    parser = argparse.ArgumentParser(description="Space KG web crawler")
    parser.add_argument(
        "--seed",
        default="data/raw/seed_urls.txt",
        help="Path to seed URLs file (one URL per line)",
    )
    parser.add_argument(
        "--out",
        default="data/processed/crawler_output.jsonl",
        help="Output JSONL file path",
    )
    parser.add_argument(
        "--min-words",
        type=int,
        default=500,
        help="Minimum word count per page (default: 500)",
    )
    args = parser.parse_args()
    crawl(args.seed, args.out, args.min_words)


if __name__ == "__main__":
    main()
