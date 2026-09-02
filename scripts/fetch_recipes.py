"""Build the local recipe corpus from TheMealDB.

TheMealDB has no bulk export, but ``search.php?f=<letter>`` returns every meal
whose name starts with that letter, so walking a-z yields the full free dataset
in 26 requests. Raw responses are cached to ``data/raw/`` so re-normalizing
(after a parser change, say) needs no network access.

Usage::

    python -m scripts.fetch_recipes            # fetch, cache, normalize
    python -m scripts.fetch_recipes --offline  # re-normalize from the cache
"""

from __future__ import annotations

import argparse
import json
import string
import sys
import time

import requests

from agents.recipe import Recipe, from_mealdb, save_recipes
from config import PROCESSED_DATA_DIR, RAW_DATA_DIR, settings

RAW_PATH = RAW_DATA_DIR / "themealdb.json"
CORPUS_PATH = PROCESSED_DATA_DIR / "recipes.json"

#: Courtesy delay between requests to a free, unauthenticated public API.
REQUEST_DELAY_SECONDS = 0.2
REQUEST_TIMEOUT_SECONDS = 20


def fetch_raw_meals() -> list[dict]:
    """Fetch every meal from TheMealDB by walking the a-z name index."""
    meals: dict[str, dict] = {}
    for letter in string.ascii_lowercase:
        url = f"{settings.themealdb_base_url}/search.php?f={letter}"
        response = requests.get(url, timeout=REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()
        batch = response.json().get("meals") or []
        for meal in batch:
            meals[str(meal["idMeal"])] = meal
        print(f"  {letter}: {len(batch):>3} meals (total {len(meals)})")
        time.sleep(REQUEST_DELAY_SECONDS)
    return list(meals.values())


def normalize(raw_meals: list[dict]) -> list[Recipe]:
    """Convert raw meal records into validated recipes, skipping empty ones."""
    recipes = []
    for meal in raw_meals:
        recipe = from_mealdb(meal)
        if recipe.ingredients and recipe.steps:
            recipes.append(recipe)
    return recipes


def main(argv: list[str] | None = None) -> int:
    """Fetch (or reuse) raw meals, normalize them, and write the corpus."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--offline",
        action="store_true",
        help="re-normalize the cached raw response instead of fetching",
    )
    args = parser.parse_args(argv)

    if args.offline:
        if not RAW_PATH.exists():
            print(f"No cache at {RAW_PATH}; run without --offline first.")
            return 1
        raw_meals = json.loads(RAW_PATH.read_text())
        print(f"Loaded {len(raw_meals)} cached meals from {RAW_PATH}")
    else:
        print("Fetching from TheMealDB...")
        raw_meals = fetch_raw_meals()
        RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
        RAW_PATH.write_text(json.dumps(raw_meals, indent=2, ensure_ascii=False))
        print(f"Cached {len(raw_meals)} raw meals to {RAW_PATH}")

    recipes = normalize(raw_meals)
    save_recipes(recipes, CORPUS_PATH)
    skipped = len(raw_meals) - len(recipes)
    print(f"Wrote {len(recipes)} recipes to {CORPUS_PATH} ({skipped} skipped)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
