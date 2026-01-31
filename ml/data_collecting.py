import os
import io
import zipfile
import urllib.request
import pandas as pd

UCI_ZIP_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"

def download_and_extract_uci(zip_url: str) -> pd.DataFrame:
    with urllib.request.urlopen(zip_url) as resp:
        data = resp.read()

    with zipfile.ZipFile(io.BytesIO(data)) as z:
        # the main file inside is named as SMSSpamCollection
        with z.open("SMSSpamCollection") as f:
            # Format: label and text
            df = pd.read_csv(f, sep = '\t', header = None, names = ["raw_label", "text"], encoding_errors = "ignore")

    # mapping the label to 0 and 1 instead of scam and ham, respectively
    df['label'] = df['raw_label'].map({"spam": 1, "ham": 0})
    df = df.drop(columns = ["raw_label"])

    df['text'] = df['text'].astype(str).str.strip()
    df = df[df['text'].str.len() > 0]
    df = df.drop_duplicates(subset = ["text"]).reset_index(drop = True)

    if df['label'].isna().any():
        bad = df[df['label'].isna()].head(5)
        raise ValueError(f"Found unknown labels. Sample:\n{bad}")

    return df

def main():
    out_dir = '../data'
    os.makedirs(out_dir, exist_ok = True)
    out_path = os.path.join(out_dir, "messages.csv")

    df = download_and_extract_uci(UCI_ZIP_URL)

    # saving in the exact format of training, i.e. text, label
    df.to_csv(out_path, index = False)
    print(f"Saved: {out_path}")
    print(df['label'].value_counts())
    print(df.head(3))

if __name__ == "__main__":
    main()