from urllib.parse import urljoin, quote
from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup

BASE_URL = "https://nub.ac.bd"
LIST_URL = f"{BASE_URL}/nub/memberlist/udqnws4i/public-health"

data = []

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()

    # Step 1: Get faculty list
    page.goto(LIST_URL, timeout=60000)
    soup = BeautifulSoup(page.content(), "html.parser")

    faculty_links = []
    main_div = soup.find("div", {"class": "kopa-main-col col-md-9 col-sm-9 col-xs-9"})
    if main_div:
        for article in main_div.find_all("article"):
            a = article.find("a")
            if a and a.get("href") and a["href"] != "#":
                full_link = urljoin(BASE_URL, a["href"])
                faculty_links.append(full_link)

    # Step 2: Visit each faculty page
    for link in faculty_links:
        page.goto(link, timeout=60000)
        detail_soup = BeautifulSoup(page.content(), "html.parser")
        article = detail_soup.find("article", class_="entry-item gallery-post clearfix")

        if not article:
            continue

        # Extract photo URL and encode spaces
        img_tag = article.find("img")
        photo_url = None
        if img_tag and img_tag.get("src"):
            photo_url = img_tag["src"]
            photo_url = photo_url.replace(" ", "%20")  # Encode spaces

        # Extract name
        name_tag = article.find("h4", class_="entry-title")
        name = name_tag.get_text(strip=True) if name_tag else None

        # Extract table data
        info = {}
        table = article.find("table")
        if table:
            for row in table.find_all("tr"):
                th = row.find("th")
                td = row.find("td")
                if th and td:
                    key = th.get_text(strip=True).replace(" :", "").replace(":", "")
                    value = td.get_text(" ", strip=True)
                    info[key] = value

        # Save record
        record = {
            "Name": name,
            "Photo URL": photo_url,
            **info
        }
        data.append(record)

    browser.close()

# Step 3: Save to a text file
with open("faculty_data.txt", "w", encoding="utf-8") as f:
    for person in data:
        for key, value in person.items():
            f.write(f"{key}: {value}\n")
        f.write("\n" + "-"*50 + "\n\n")

print(f"Saved {len(data)} faculty profiles to faculty_data.txt")
