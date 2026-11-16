"""
Checks the continent table against the country folders that are really on disk, so a missing
or wrong code shows up before any analysis uses it.

Run it with:
    python3 src/check_continents.py
"""

from collections import Counter

from clip_probe import COUNTRY211_DIR
from continents import CONTINENT_OF


def main():
    codes = sorted(p.name for p in (COUNTRY211_DIR / "train").iterdir() if p.is_dir())
    missing = [code for code in codes if code not in CONTINENT_OF]
    extra = [code for code in CONTINENT_OF if code not in codes]
    print(f"country folders: {len(codes)}")
    print(f"not in the table: {missing}")
    print(f"in the table but not in the data: {len(extra)}")
    counts = Counter(CONTINENT_OF[code] for code in codes if code in CONTINENT_OF)
    for continent, count in counts.most_common():
        print(f"{continent}: {count}")


if __name__ == "__main__":
    main()
