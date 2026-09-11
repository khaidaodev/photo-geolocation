"""
Downloads and loads Country211 (OpenAI/CLIP's "which country was this photo taken in" benchmark),
built from geotagged Flickr photos, sampled into a balanced 211-country image dataset: 150
training, 50 validation, and 100 test photos for each country.

The full dataset is huge (~11GB, 63,300 photos across all 211 countries), and 211-way
classification is an extremely hard first problem, even OpenAI's own CLIP model struggles with
it. So on top of downloading the raw dataset, this also picks out a smaller, easier starting
subset of well known, visually distinctive countries to actually build the first model on,
before trying to scale up to more of the 211.

Run it with:
    python src/data_loading.py
"""

import ssl
from pathlib import Path

import certifi
import pycountry

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"

# A first, easier subset to start with: well known countries that tend to look visually distinct
# from each other (different climate, architecture, vegetation, road markings), rather than
# jumping straight into all 211 countries including obscure or visually-similar ones. Codes are
# ISO 3166-1 alpha-2, matching the folder names Country211 uses on disk.
STARTER_COUNTRIES = [
    "US", "GB", "FR", "DE", "IT", "ES", "JP", "IN", "BR", "AU",
    "RU", "EG", "TH", "MX", "ZA", "AR", "TR", "KR", "NL", "GR",
]


def country_name(code: str) -> str:
    """Turns an ISO 3166-1 alpha-2 country code (e.g. "US") into a readable name (e.g.
    "United States"), using the pycountry library rather than hand-typing out 211 country names.
    Falls back to just returning the code itself if it's not a real/recognised one."""
    country = pycountry.countries.get(alpha_2=code)
    return country.name if country else code


def download_country211() -> Path:
    """Downloads the full Country211 dataset (train/valid/test splits) into data/raw/ if it
    isn't already there. This is a ~11GB download across all three splits, so it can take a
    while depending on your internet connection, it only needs to happen once.

    torchvision is imported here rather than at the top of the file, so the pure logic above
    (country_name, list_downloaded_countries) can be imported and tested without needing torch
    and torchvision installed, same idea as demo/webcam_demo.py in the sign language project.

    Also points Python's default https handling at the certifi package's certificate bundle
    before downloading. Macs with Python installed via python.org don't always ship a working
    list of trusted certificates for Python to check https sites against, causing a "self-signed
    certificate in certificate chain" error, same issue and same fix as the phishing detector
    project's dataset download."""
    import torchvision

    ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for split in ("train", "valid", "test"):
        torchvision.datasets.Country211(root=str(RAW_DIR), split=split, download=True)
    return RAW_DIR / "country211"


def list_downloaded_countries(country211_dir: Path, split: str = "train") -> list[str]:
    """Returns the country codes actually present on disk for a given split, sorted
    alphabetically. Useful for checking how much of the dataset has actually finished
    downloading/extracting, rather than assuming all 211 are there."""
    split_dir = country211_dir / split
    if not split_dir.exists():
        return []
    return sorted(p.name for p in split_dir.iterdir() if p.is_dir())


if __name__ == "__main__":
    country211_dir = download_country211()
    downloaded = list_downloaded_countries(country211_dir)
    print(f"{len(downloaded)} countries available in {country211_dir}")
    print("starter subset:", [country_name(c) for c in STARTER_COUNTRIES])
