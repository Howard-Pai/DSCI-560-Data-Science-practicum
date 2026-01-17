import requests
from bs4 import BeautifulSoup
import os

URL = "https://www.cnbc.com/world/?region=world"

RAW_DATA_DIR = "../data/raw_data"
OUTPUT_FILE = os.path.join(RAW_DATA_DIR, "web_data.html")

HEADERS = {
	"User-Agent":(
		"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"
		"AppleWebKit/537.36 (KHTML, like Gecko)"
	)
}

if __name__ == '__main__':
	response = requests.get(URL, headers = HEADERS)
	response.raise_for_status()

	soup = BeautifulSoup(response.text, "html.parser")

	os.makedirs(RAW_DATA_DIR, exist_ok=True)

	with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
		f.write(soup.prettify())

	print(f"Web data saved to {OUTPUT_FILE}")


