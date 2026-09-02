"""Build the simple-recipe half of the corpus from RecipeNLG.

TheMealDB is a fine corpus of composed restaurant dishes, and a poor one for
"what can I cook with what I have": median ten ingredients, and for a six-item
pantry only four recipes come within one shop. RecipeNLG is user-submitted home
cooking -- median seven ingredients, half of them six or fewer -- which is the
shape this product needs.

It is large (2.1M recipes, 800MB), so this script streams one parquet shard and
keeps a curated slice: short ingredient lists, a sane number of steps, a
readable title, and an ingredient list its own instructions do not contradict.

**Licence.** RecipeNLG is CC BY-NC-SA 4.0 -- non-commercial, attribution,
share-alike. That covers the recipes this script writes, and is why the corpus
carries a ``source`` field. See DATA_LICENSES.md.

Usage::

    python -m scripts.fetch_recipenlg              # download, cache, curate
    python -m scripts.fetch_recipenlg --offline    # re-curate from the cache
"""

from __future__ import annotations

import argparse
import re
import sys

import requests

from agents.ingredient import parse_phrase
from agents.recipe import Recipe, find_unlisted_in_steps, save_recipes
from config import PROCESSED_DATA_DIR, RAW_DATA_DIR

SHARD_URL = (
    "https://huggingface.co/api/datasets/corbt/all-recipes"
    "/parquet/default/train/0.parquet"
)
RAW_PATH = RAW_DATA_DIR / "recipenlg.parquet"
CORPUS_PATH = PROCESSED_DATA_DIR / "recipenlg.json"

#: Records arrive as one text blob: title, ingredient bullets, direction
#: bullets.
_BLOCK = re.compile(
    r"^(?P<title>.+?)\n\nIngredients:\n(?P<ingredients>.*?)\n\nDirections:\n(?P<steps>.*)$",
    flags=re.S,
)

#: Curation bounds. The upper ingredient bound is the whole point -- anything
#: longer is the kind of project recipe TheMealDB already covers.
MIN_INGREDIENTS, MAX_INGREDIENTS = 3, 9
MIN_STEPS, MAX_STEPS = 2, 12
MAX_TITLE_LENGTH = 90

#: A recipe whose steps reach for more than one unlisted ingredient is not one
#: to recommend when the entire promise is an accurate shopping list.
MAX_UNLISTED = 1

DEFAULT_TARGET = 12000
DOWNLOAD_CHUNK_BYTES = 1 << 20


def download_shard() -> None:
    """Stream the parquet shard to the raw cache."""
    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {SHARD_URL} …")
    with requests.get(SHARD_URL, stream=True, timeout=600) as response:
        response.raise_for_status()
        with RAW_PATH.open("wb") as handle:
            for chunk in response.iter_content(DOWNLOAD_CHUNK_BYTES):
                handle.write(chunk)
    print(f"Cached {RAW_PATH.stat().st_size / 1e6:.0f} MB to {RAW_PATH}")


def parse_record(text: str, record_id: int) -> Recipe | None:
    """Turn one raw RecipeNLG blob into a :class:`Recipe`, or reject it.

    Returns ``None`` for anything that fails the curation bounds, so the caller
    can simply skip it.
    """
    match = _BLOCK.match(text.strip())
    if match is None:
        return None

    title = match.group("title").strip()
    if not title or len(title) > MAX_TITLE_LENGTH:
        return None

    ingredients = [
        parsed
        for line in match.group("ingredients").splitlines()
        if line.startswith("- ") and (parsed := parse_phrase(line[2:]))
    ]
    steps = [
        line[2:].strip()
        for line in match.group("steps").splitlines()
        if line.startswith("- ") and line[2:].strip()
    ]
    if not MIN_INGREDIENTS <= len(ingredients) <= MAX_INGREDIENTS:
        return None
    if not MIN_STEPS <= len(steps) <= MAX_STEPS:
        return None

    unlisted = find_unlisted_in_steps(ingredients, steps)
    if len(unlisted) > MAX_UNLISTED:
        return None

    return Recipe(
        id=f"nlg-{record_id}",
        title=title,
        ingredients=ingredients,
        steps=steps,
        unlisted_in_steps=unlisted,
        source="recipenlg",
    )


def curate(target: int) -> list[Recipe]:
    """Scan the cached shard and keep ``target`` distinct simple recipes."""
    import pyarrow.parquet as pq

    column = pq.read_table(RAW_PATH).column("input")
    kept: list[Recipe] = []
    seen_titles: set[str] = set()

    for index in range(len(column)):
        recipe = parse_record(column[index].as_py(), index)
        if recipe is None:
            continue
        key = recipe.title.casefold()
        if key in seen_titles:
            continue
        seen_titles.add(key)
        kept.append(recipe)
        if len(kept) >= target:
            break
        if len(kept) % 2000 == 0:
            print(f"  kept {len(kept)} of {index + 1} scanned")
    return kept


def main(argv: list[str] | None = None) -> int:
    """Download (or reuse) the shard, curate it, and write the corpus."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--offline", action="store_true", help="re-curate the cached shard"
    )
    parser.add_argument(
        "--target", type=int, default=DEFAULT_TARGET, help="how many to keep"
    )
    args = parser.parse_args(argv)

    if not args.offline or not RAW_PATH.exists():
        if args.offline:
            print(f"No cache at {RAW_PATH}; downloading anyway.")
        download_shard()

    recipes = curate(args.target)
    save_recipes(recipes, CORPUS_PATH)
    median = sorted(len(r.ingredients) for r in recipes)[len(recipes) // 2]
    print(
        f"Wrote {len(recipes)} recipes to {CORPUS_PATH} "
        f"(median {median} ingredients)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
