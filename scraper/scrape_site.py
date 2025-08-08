# scraper/scrape_site.py
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import time
import os
import re
from collections import deque

ROOT = "https://nub.ac.bd"
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(DATA_DIR, exist_ok=True)

HEADERS = {
    "User-Agent": "NUB-Chatbot-Scraper/1.0 (+https://yourdomain.example)"
}

# don't crawl these extensions
SKIP_EXT = re.compile(r".*\.(jpg|jpeg|png|gif|svg|pdf|mp4|mp3|zip|rar|css|js)$", re.IGNORECASE)

def is_same_domain(url):
    try:
        return urlparse(url).netloc.endswith("nub.ac.bd")
    except Exception:
        return False

def norm_filename(url):
    parsed = urlparse(url)
    path = parsed.path.strip("/").replace("/", "_") or "home"
    if parsed.query:
        path += "_" + re.sub(r'[^0-9a-zA-Z]+', '_', parsed.query)[:50]
    return re.sub(r'[^0-9a-zA-Z_\-\.]', '_', path)[:200] + ".txt"

def extract_text(html):
    soup = BeautifulSoup(html, "html.parser")
    # remove scripts/styles
    for tag in soup(["script", "style", "noscript", "header", "footer", "nav"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    # simple cleanup
    lines = [line.strip() for line in text.splitlines()]
    lines = [l for l in lines if l and len(l) > 10]  # drop tiny lines
    return "\n\n".join(lines)

def crawl(root, max_pages=200, delay=0.5):
    visited = set()
    q = deque([root])
    count = 0
    while q and count < max_pages:
        url = q.popleft()
        if url in visited:
            continue
        visited.add(url)
        if SKIP_EXT.match(url):
            continue
        try:
            resp = requests.get(url, headers=HEADERS, timeout=15)
            if resp.status_code != 200 or "text" not in resp.headers.get("Content-Type", ""):
                continue
            text = extract_text(resp.text)
            if not text:
                continue
            fname = os.path.join(DATA_DIR, norm_filename(url))
            with open(fname, "w", encoding="utf-8") as f:
                f.write(f"URL: {url}\n\n")
                f.write(text)
            print(f"[{count+1}] Saved {url} -> {os.path.basename(fname)}")
            count += 1

            # find internal links
            soup = BeautifulSoup(resp.text, "html.parser")
            for a in soup.find_all("a", href=True):
                href = urljoin(url, a["href"])
                href = href.split("#")[0]  # strip anchors
                if is_same_domain(href) and href not in visited:
                    q.append(href)
        except Exception as e:
            print("Failed", url, e)
        time.sleep(delay)

if __name__ == "__main__":
    # set max_pages to a conservative default; you can increase after testing
    crawl(ROOT, max_pages=400, delay=0.3)
    print("Done scraping. Files in:", DATA_DIR)
