"""
PromptAutopsy — Scraper
========================
Scrapes all knowledge base sources and saves
raw documents to disk as .txt files.

Run once before building the index:
    python scraper.py

Output: /raw_docs folder with one .txt per source
"""

import os
import time
from typing import Optional
import requests
from bs4 import BeautifulSoup
import fitz  # PyMuPDF

# ── Config ────────────────────────────────────────────────────────────────────

OUTPUT_DIR = "./raw_docs"
HEADERS    = {"User-Agent": "Mozilla/5.0 (compatible; PromptAutopsy/1.0)"}

# ── Sources ───────────────────────────────────────────────────────────────────

WEBSITE_SOURCES = [
    {
        "url"  : "https://raw.githubusercontent.com/dair-ai/Prompt-Engineering-Guide/main/guides/prompts-intro.md",
        "name" : "dair_prompting_intro",
        "label": "DAIR Prompting Intro"
    },
    {
        "url"  : "https://raw.githubusercontent.com/dair-ai/Prompt-Engineering-Guide/main/guides/prompts-advanced-usage.md",
        "name" : "dair_prompting_advanced",
        "label": "DAIR Prompting Advanced"
    },
    {
        "url"  : "https://raw.githubusercontent.com/dair-ai/Prompt-Engineering-Guide/main/guides/prompts-chatgpt.md",
        "name" : "dair_prompting_chatgpt",
        "label": "DAIR ChatGPT Prompting"
    },
]

PDF_SOURCES = [
    {
        "url"  : "https://arxiv.org/pdf/2307.03172",
        "name" : "lost_in_the_middle",
        "label": "Lost in the Middle (arXiv)"
    },
    {
        "url"  : "https://arxiv.org/pdf/2205.11916",
        "name" : "zero_shot_reasoners",
        "label": "Zero-Shot Reasoners (arXiv)"
    },
    {
        "url"  : "https://arxiv.org/pdf/2201.11903",
        "name" : "chain_of_thought",
        "label": "Chain of Thought Prompting (arXiv)"
    },
]

# ── Scrapers ──────────────────────────────────────────────────────────────────

def scrape_website(source: dict) -> Optional[str]:
    """Fetch raw markdown and return text directly."""
    print(f"  Scraping {source['label']}...")
    try:
        response = requests.get(source["url"], headers=HEADERS, timeout=15)
        response.raise_for_status()
        text = response.text
        print(f"    ✓ {len(text):,} characters extracted")
        return text
    except Exception as e:
        print(f"    ✗ Failed: {e}")
        return None


def scrape_pdf(source: dict) -> Optional[str]:
    """Download a PDF and extract text."""
    print(f"  Downloading {source['label']}...")
    try:
        response = requests.get(source["url"], headers=HEADERS, timeout=30)
        response.raise_for_status()

        pdf   = fitz.open(stream=response.content, filetype="pdf")
        pages = []

        for page_num, page in enumerate(pdf):
            text = page.get_text()
            if text.strip():
                pages.append(f"[Page {page_num + 1}]\n{text}")

        full_text = "\n\n".join(pages)
        full_text = full_text.replace("-\n", "")
        full_text = full_text.replace("\n\n\n", "\n\n")

        print(f"    ✓ {len(pdf)} pages, {len(full_text):,} characters extracted")
        return full_text

    except Exception as e:
        print(f"    ✗ Failed: {e}")
        return None


# ── Save to disk ──────────────────────────────────────────────────────────────

def save(name: str, text: str, label: str):
    """Save scraped text to raw_docs folder with metadata header."""
    filepath = os.path.join(OUTPUT_DIR, f"{name}.txt")
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"SOURCE: {label}\n")
        f.write(f"NAME: {name}\n")
        f.write(f"{'=' * 60}\n\n")
        f.write(text)
    print(f"    💾 Saved to {filepath}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("\n🔬 PromptAutopsy — Scraper")
    print("=" * 50)

    # Create output folder
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    success = 0
    failed  = 0

    # Scrape websites
    print("\n📄 Websites:")
    for source in WEBSITE_SOURCES:
        text = scrape_website(source)
        if text:
            save(source["name"], text, source["label"])
            success += 1
        else:
            failed += 1
        time.sleep(1)  # be polite to servers

    # Scrape PDFs
    print("\n📑 PDFs:")
    for source in PDF_SOURCES:
        text = scrape_pdf(source)
        if text:
            save(source["name"], text, source["label"])
            success += 1
        else:
            failed += 1
        time.sleep(1)

    # Summary
    print(f"\n{'=' * 50}")
    print(f"✅ Scraping complete")
    print(f"   Succeeded : {success}")
    print(f"   Failed    : {failed}")
    print(f"   Saved to  : {OUTPUT_DIR}/")
    print(f"\nNext step: run ingest.py to chunk + embed + index")
    print(f"{'=' * 50}\n")


if __name__ == "__main__":
    main()