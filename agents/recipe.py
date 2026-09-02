"""Recipe corpus: schema, normalization, and disk I/O.

Recipes from TheMealDB arrive as flat records with twenty numbered ingredient
and measure fields. They are folded here into a :class:`Recipe` whose ingredient
names go through the *same* :func:`agents.ingredient.parse_phrase` pipeline the
user's pantry does. That symmetry is the whole trick behind the retrieval
agent's overlap score: both sides are canonical, so "scallions" in the pantry
matches "spring onions" in a recipe without any fuzzy matching at query time.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from pydantic import BaseModel, Field

from agents.ingredient import Ingredient, parse_phrase
from agents.normalization import mentioned_ingredients

#: TheMealDB packs ingredients into strIngredient1..strIngredient20.
_MAX_MEALDB_INGREDIENTS = 20

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")
_STEP_PREFIX = re.compile(r"^\s*(?:step\s*)?\d+[.)]\s*", flags=re.IGNORECASE)
#: Some sources put the step number on its own line. A step that is nothing
#: but a number is numbering, not an instruction.
_BARE_NUMBER = re.compile(r"^\s*(?:step\s*)?\d+[.)]?\s*$", flags=re.IGNORECASE)


class Recipe(BaseModel):
    """A single normalized recipe.

    Attributes:
        id: Stable source id, used as the vector-store document key.
        title: Human-readable recipe name.
        category: Coarse type ("Seafood", "Dessert"), when the source gives one.
        area: Cuisine of origin, when the source gives one.
        ingredients: Canonicalized ingredients, parsed from the source measures.
        steps: Instructions split into discrete, numbered-ready steps.
        unlisted_in_steps: Ingredients the instructions name that the
            ingredient list omits. A trust signal, not a repair: roughly half
            of TheMealDB's recipes have some, and they are as often an
            alternative ("pork or chicken") or an optional garnish as a real
            omission, so they are recorded and ranked down rather than added.
        source: Which dataset this came from, for attribution and filtering.
        source_url: Attribution link back to the original recipe.
        thumbnail: Image URL, used by the Streamlit UI.
        tags: Free-form source tags.
    """

    id: str
    title: str = Field(min_length=1)
    category: str | None = None
    area: str | None = None
    ingredients: list[Ingredient] = Field(default_factory=list)
    steps: list[str] = Field(default_factory=list)
    unlisted_in_steps: list[str] = Field(default_factory=list)
    source: str = ""
    source_url: str | None = None
    thumbnail: str | None = None
    tags: list[str] = Field(default_factory=list)

    @property
    def ingredient_names(self) -> set[str]:
        """Canonical ingredient names, for set overlap against a pantry."""
        return {item.name for item in self.ingredients}

    def searchable_text(self) -> str:
        """Flat text representation used to embed this recipe for retrieval."""
        parts = [self.title, self.category or "", self.area or ""]
        parts.extend(sorted(self.ingredient_names))
        return " ".join(part for part in parts if part)


def split_steps(instructions: str) -> list[str]:
    """Split an instruction blob into individual steps.

    Prefers the source's own line breaks; falls back to sentence boundaries when
    the instructions arrive as one unbroken paragraph. Existing "1." / "Step 2)"
    prefixes are stripped so the composer can number steps itself, and lines
    that are *only* a step number are dropped entirely -- several TheMealDB
    entries put the number on its own line, which would otherwise double the
    step count with empty instructions.
    """
    if not instructions or not instructions.strip():
        return []

    lines = [line.strip() for line in instructions.splitlines()]
    steps = [line for line in lines if line]
    if len(steps) <= 1:
        sentences = _SENTENCE_END.split(instructions.strip())
        steps = [sentence.strip() for sentence in sentences if sentence.strip()]

    cleaned = [_STEP_PREFIX.sub("", step).strip() for step in steps]
    return [step for step in cleaned if step and not _BARE_NUMBER.match(step)]


def _mealdb_ingredients(payload: dict) -> list[Ingredient]:
    """Fold TheMealDB's numbered ingredient/measure pairs into ingredients."""
    parsed: list[Ingredient] = []
    for index in range(1, _MAX_MEALDB_INGREDIENTS + 1):
        name = (payload.get(f"strIngredient{index}") or "").strip()
        if not name:
            continue
        measure = (payload.get(f"strMeasure{index}") or "").strip()
        item = parse_phrase(f"{measure} {name}".strip())
        if item:
            parsed.append(item)
    return parsed


#: Words a step may use without the list naming them: things every kitchen
#: has, plus "clove" which in a step almost always means a garlic clove.
_ASSUMED_IN_STEPS: frozenset[str] = frozenset({
    "water", "salt", "black pepper", "olive oil", "vegetable oil", "sugar",
    "butter", "flour", "all-purpose flour", "ice", "clove", "stock", "broth",
})


def find_unlisted_in_steps(
    ingredients: list[Ingredient], steps: list[str]
) -> list[str]:
    """Ingredients the steps name that ``ingredients`` does not list."""
    listed = {item.name for item in ingredients}
    allowed = listed | _ASSUMED_IN_STEPS | {t for name in listed for t in name.split()}
    unlisted: set[str] = set()
    for step in steps:
        for name in mentioned_ingredients(step):
            if name not in allowed and not any(t in allowed for t in name.split()):
                unlisted.add(name)
    return sorted(unlisted)


def from_mealdb(payload: dict) -> Recipe:
    """Build a :class:`Recipe` from one raw TheMealDB meal record."""
    tags = (payload.get("strTags") or "").strip()
    ingredients = _mealdb_ingredients(payload)
    steps = split_steps(payload.get("strInstructions") or "")
    return Recipe(
        id=str(payload["idMeal"]),
        title=(payload.get("strMeal") or "").strip(),
        category=(payload.get("strCategory") or "").strip() or None,
        area=(payload.get("strArea") or "").strip() or None,
        ingredients=ingredients,
        steps=steps,
        unlisted_in_steps=find_unlisted_in_steps(ingredients, steps),
        source_url=(payload.get("strSource") or "").strip() or None,
        thumbnail=(payload.get("strMealThumb") or "").strip() or None,
        tags=[tag.strip() for tag in tags.split(",") if tag.strip()],
    )


def save_recipes(recipes: list[Recipe], path: Path, *, compact: bool = True) -> None:
    """Write ``recipes`` to ``path`` as a JSON array.

    Compact by default: this is derived data, rebuilt by a script, and at
    corpus scale the indentation costs several megabytes in the repository for
    readability nobody uses.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [recipe.model_dump() for recipe in recipes]
    text = (
        json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        if compact
        else json.dumps(payload, indent=2, ensure_ascii=False)
    )
    path.write_text(text)


def load_recipes(path: Path) -> list[Recipe]:
    """Read recipes previously written by :func:`save_recipes`.

    Raises:
        FileNotFoundError: With a pointer to the fetch script, since a missing
            corpus almost always means the dataset was never built.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"No recipe corpus at {path}. Build it with:\n"
            f"    python -m scripts.build_corpus"
        )
    return [Recipe.model_validate(item) for item in json.loads(path.read_text())]
