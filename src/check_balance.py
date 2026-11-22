"""
Counts the photos per country in each split and checks they are even, because an uneven
dataset would make a plain accuracy score misleading (a model could do well by favouring the
big countries).

Run it with:
    python3 src/check_balance.py
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
COUNTRY211_DIR = ROOT / "data" / "raw" / "country211"
SPLITS = ["train", "valid", "test"]


def summarise_counts(counts_by_country):
    """Gives back (countries, smallest count, largest count, countries not at the most common count)."""
    counts = list(counts_by_country.values())
    most_common = max(set(counts), key=counts.count)
    off = sorted(code for code, count in counts_by_country.items() if count != most_common)
    return len(counts), min(counts), max(counts), off


if __name__ == "__main__":
    for split in SPLITS:
        counts = {
            folder.name: sum(1 for f in folder.iterdir() if f.is_file() and not f.name.startswith("."))
            for folder in (COUNTRY211_DIR / split).iterdir()
            if folder.is_dir()
        }
        countries, smallest, largest, off = summarise_counts(counts)
        print(f"{split}: {countries} countries, {smallest} to {largest} photos each, "
              f"{sum(counts.values())} photos in total, uneven countries: {off}")
