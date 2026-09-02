"""Merge every fetched source into the single corpus the app loads.

Sources are kept as separate files so each can be rebuilt on its own, and
merged here so the rest of the pipeline sees one list. Recipes carry a
``source`` field, which is what lets the two licences be reported accurately
and lets a source be dropped without re-fetching the others.

Usage::

    python -m scripts.build_corpus
"""

from __future__ import annotations

import sys
from collections import Counter

from agents.recipe import load_recipes, save_recipes
from config import PROCESSED_DATA_DIR

CORPUS_PATH = PROCESSED_DATA_DIR / "recipes.json"

#: Built by scripts/fetch_recipes.py and scripts/fetch_recipenlg.py. A missing
#: source is skipped with a note rather than being an error, so the corpus can
#: be rebuilt from whichever sources have been fetched.
SOURCE_FILES = {
    "themealdb": PROCESSED_DATA_DIR / "themealdb.json",
    "recipenlg": PROCESSED_DATA_DIR / "recipenlg.json",
}


def main(argv: list[str] | None = None) -> int:
    """Merge the fetched sources into ``data/processed/recipes.json``."""
    merged = []
    for name, path in SOURCE_FILES.items():
        if not path.exists():
            print(f"  {name}: not fetched, skipping ({path.name})")
            continue
        recipes = load_recipes(path)
        merged.extend(recipes)
        print(f"  {name}: {len(recipes)} recipes")

    if not merged:
        print("No sources found. Run scripts.fetch_recipes first.")
        return 1

    # Ids are namespaced per source, but assert it rather than assume it: a
    # collision would silently shadow one recipe with another.
    duplicates = [i for i, n in Counter(r.id for r in merged).items() if n > 1]
    if duplicates:
        print(f"Duplicate recipe ids across sources: {duplicates[:5]}")
        return 1

    save_recipes(merged, CORPUS_PATH)
    sizes = sorted(len(r.ingredients) for r in merged)
    print(
        f"\nWrote {len(merged)} recipes to {CORPUS_PATH} "
        f"({CORPUS_PATH.stat().st_size / 1e6:.1f} MB, "
        f"median {sizes[len(sizes) // 2]} ingredients)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
