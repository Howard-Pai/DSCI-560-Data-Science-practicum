import os
import csv
from bs4 import BeautifulSoup

RAW_FILE = "../data/raw_data/web_data.html"
PROCESSED_DIR = "../data/processed_data"
MARKET_CSV = os.path.join(PROCESSED_DIR, "market_data.csv")
NEWS_CSV = os.path.join(PROCESSED_DIR, "news_data.csv")

if __name__ == "__main__":
    print("Reading raw HTML file...")
    with open(RAW_FILE, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f, "html.parser")

    # Ensure processed_data folder exists
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # ----------------------------
    # 1 Extract Market Banner Data
    # ----------------------------
    print("Filtering Market banner fields...")

    # Adjust selectors based on actual CNBC page structure
    market_data = []
    market_cards = soup.find_all("div", class_="MarketCard-row")
    for card in market_cards:
        symbol = card.find("span", class_="MarketCard-symbol")
        stock_pos = card.find("span", class_="MarketCard-stockPosition")
        change_data = card.find("div", class_="MarketCard-changeData")
        change_pct = change_data("div", class_="MarketCard-changesPct")

        # Only add if all fields exist
        if symbol and stock_pos and change_pct:
            market_data.append([
                symbol.text.strip(),
                stock_pos.text.strip(),
                change_pct.text.strip()
            ])

    # Save Market CSV
    if market_data:
        print(f"Storing Market data ({len(market_data)} entries)...")
        with open(MARKET_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["symbol", "stockPosition", "changePct"])
            writer.writerows(market_data)
        print(f"CSV created: {MARKET_CSV}")
    else:
        print("No Market data found!")

    # ----------------------------
    # 2 Extract Latest News Data
    # ----------------------------
    print("Filtering Latest News fields...")

    news_data = []
    latest_news = soup.find_all("li", class_="LatestNews-item")
    if latest_news:
        for news in latest_news:
            timestamp = news.find("time")
            title_tag = news.find("a")
            if timestamp and title_tag:
                news_data.append([
                    timestamp.text.strip(),
                    title_tag.text.strip(),
                    title_tag.get("href")
                ])

    # Save News CSV
    if news_data:
        print(f"Storing Latest News data ({len(news_data)} entries)...")
        with open(NEWS_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "title", "link"])
            writer.writerows(news_data)
        print(f"CSV created: {NEWS_CSV}")
    else:
        print("No Latest News data found!")

