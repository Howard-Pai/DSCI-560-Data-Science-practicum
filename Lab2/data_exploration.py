import os
import json
import requests
import pandas as pd
from bs4 import BeautifulSoup
import pdfplumber
from kaggle.api.kaggle_api_extended import KaggleApi
import zipfile
import argparse


# -----------------------------------
# 1. CSV DATA COLLECTION (RecipeNLG via Kaggle API)
# -----------------------------------

def collect_csv_data():
    print("\n=== CSV DATA COLLECTION (RecipeNLG via Kaggle API) ===\n")

    if not os.path.exists("data/RecipeNLG_dataset.csv"):

        # Load Kaggle API credentials from local file
        with open("kaggle_api_key.json", "r") as f:
            creds = json.load(f)

        os.environ["KAGGLE_USERNAME"] = creds["username"]
        os.environ["KAGGLE_KEY"] = creds["key"]

        api = KaggleApi()
        api.authenticate()

        dataset_name = "paultimothymooney/recipenlg"
        file_name = "RecipeNLG_dataset.csv"
        download_path = "data"
        os.makedirs(download_path, exist_ok=True)
        api.dataset_download_file(
            dataset_name,
            file_name= file_name,
            path=download_path,
            quiet = False
        )

        zip_path = os.path.join(download_path, file_name + ".zip")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(download_path)
        os.remove(zip_path)

    df = pd.read_csv("data/RecipeNLG_dataset.csv")

    # Save a copy for exploration output
    df.to_csv("recipe_data_sample.csv", index=False)

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nDataset shape (rows, columns):")
    print(df.shape)

    print("\nMissing values per column:")
    print(df.isnull().sum())

    return df


# ---------------------------------------------------
# 2. WEB TEXT COLLECTION (USC Data Science Courses)
# ---------------------------------------------------

def collect_web_text():
    print("\n=== WEB TEXT COLLECTION (USC Data Science Courses) ===")

    url = "https://www.cs.usc.edu/academic-programs/data-science/"

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }

    response = requests.get(url, headers=headers)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    text = soup.get_text(separator="\n", strip=True)

    with open("data/usc_data_science_courses.txt", "w", encoding="utf-8") as f:
        f.write(text)

    print("\nSample extracted text:")
    print(text[:600])

    return text


# ---------------------------------------------------
# 3. PDF DATA COLLECTION (Local Course Syllabus)
# ---------------------------------------------------

def collect_pdf_text():
    print("\n=== PDF TEXT EXTRACTION (Local Syllabus) ===")

    pdf_path = "DSCI560_Syllabus.pdf"

    extracted_text = ""

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                extracted_text += page_text + "\n"

    with open("data/syllabus_text.txt", "w", encoding="utf-8") as f:
        f.write(extracted_text)

    print("\nSample extracted PDF text:")
    print(extracted_text[:600])

    return extracted_text

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data collection script")
    parser.add_argument(
        "--task",
        choices=["csv", "web", "pdf", "all"],
        default="all",
        help="Select the type of data to collect"
    )

    args = parser.parse_args()

    if args.task == "csv":
        collect_csv_data()
    elif args.task == "web":
        collect_web_text()
    elif args.task == "pdf":
        collect_pdf_text()
    elif args.task == "all":
        collect_csv_data()
        collect_web_text()
        collect_pdf_text()

