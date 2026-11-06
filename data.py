"""Small loader for the competition files."""
import os
from pathlib import Path
import pandas as pd

FILES = {"listings": "listings", "off_market": "test_off_market", "list_aware": "test_list_aware"}
DATES = ("date_listed", "date_sold", "external_avm_estimate_date")

def load(name):
    folder = Path(os.environ.get("DATADASH_DATA", "data"))
    stem = FILES[name]
    for suffix in (".parquet", ".csv"):
        path = folder / f"{stem}{suffix}"
        if path.exists() and not path.read_bytes()[:48].startswith(b"version https://git-lfs.github.com"):
            df = pd.read_parquet(path) if suffix == ".parquet" else pd.read_csv(path, low_memory=False)
            for col in DATES:
                if col in df: df[col] = pd.to_datetime(df[col])
            return df
    raise FileNotFoundError(f"Missing usable {stem} data. Set DATADASH_DATA to the competition data folder.")
