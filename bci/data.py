"""Raw data: download from the public Olist repository, verify checksums, load into DuckDB."""
from __future__ import annotations

import hashlib
import json
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
SAMPLE = ROOT / "data" / "sample"
MANIFEST = ROOT / "data" / "manifest.json"
BASE_URL = "https://raw.githubusercontent.com/olist/work-at-olist-data/master/datasets/"

# Table name in the warehouse -> file name in the Olist repository.
TABLES = {
    "raw_orders": "olist_orders_dataset.csv",
    "raw_items": "olist_order_items_dataset.csv",
    "raw_products": "olist_products_dataset.csv",
    "raw_customers": "olist_customers_dataset.csv",
    "raw_reviews": "olist_order_reviews_dataset.csv",
    "raw_payments": "olist_order_payments_dataset.csv",
    "raw_sellers": "olist_sellers_dataset.csv",
    "raw_category_translation": "product_category_name_translation.csv",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(force: bool = False, retries: int = 3) -> dict:
    """Fetch every file once, then check it against the manifest. Returns {file: 'ok' | 'downloaded'}."""
    RAW.mkdir(parents=True, exist_ok=True)
    expected = json.loads(MANIFEST.read_text())["files"]
    status = {}
    for name in TABLES.values():
        path = RAW / name
        if force or not path.exists():
            for attempt in range(retries):
                try:
                    urllib.request.urlretrieve(BASE_URL + name, path)
                    break
                except OSError:
                    if attempt == retries - 1:
                        raise
                    time.sleep(2 ** attempt)
            status[name] = "downloaded"
        else:
            status[name] = "ok"
        if sha256(path) != expected[name]["sha256"]:
            raise ValueError(f"{name}: checksum differs from data/manifest.json, the source has changed")
    return status


def source_dir(sample: bool = False) -> Path:
    """Full data if present, else the small committed sample (used by tests and CI)."""
    if not sample and all((RAW / f).exists() for f in TABLES.values()):
        return RAW
    return SAMPLE


def load_raw(con, folder: Path) -> None:
    """Create one raw table per file, every column read as text."""
    for table, name in TABLES.items():
        con.execute(f"CREATE OR REPLACE TABLE {table} AS SELECT * FROM read_csv('{(folder / name).as_posix()}', "
                    "header = true, all_varchar = true)")
