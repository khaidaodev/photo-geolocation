"""
Works out what random guessing would score, so the real scores have something to be compared
with. The countries are perfectly balanced (checked in check_balance.py), so a random guess
is right 1 time in 211, and the right country is in a random top-k list k times in 211.

Run it with:
    python3 src/chance_baseline.py
"""

SCORES_TO_COMPARE = {1: 0.435, 3: 0.619, 5: 0.694}


def chance_at(k, n_countries):
    """Chance that the true country is inside k random guesses out of n_countries."""
    return k / n_countries


if __name__ == "__main__":
    n_countries = 211
    for k, model_score in SCORES_TO_COMPARE.items():
        chance = chance_at(k, n_countries)
        print(f"top-{k}: chance {chance:.2%}, best setup {model_score:.1%}, "
              f"{model_score / chance:.1f} times chance")
