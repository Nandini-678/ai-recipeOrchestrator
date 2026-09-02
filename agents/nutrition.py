"""Nutrition agent: per-ingredient facts from USDA, totalled and scaled in code.

No language model touches a number here. The model is bad at arithmetic and,
worse, is confidently bad at it -- so the pipeline is: look up facts per 100g
from USDA FoodData Central, convert each ingredient's quantity to grams, scale
linearly, and sum. Every step is ordinary Python that can be unit tested.

Converting to grams is the only genuinely hard part, and it is honest about
which of its three cases it is in:

* **Mass** (``g``, ``oz``, ``lb``) is exact arithmetic, independent of what the
  ingredient is.
* **Volume** (``cup``, ``tbsp``, ``ml``) needs a density: a cup of flour and a
  cup of honey differ by nearly a factor of three. Known densities are tabled;
  unknown ones fall back to water and are marked approximate.
* **Counts** ("2 eggs", "3 cloves garlic") need a typical item weight, which is
  an estimate by nature.

An ingredient that cannot be converted, or that USDA has no match for, is
reported in :attr:`NutritionReport.unestimated` rather than silently counted as
zero. A nutrition panel that quietly omits half the recipe is worse than one
that says what it could not measure.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, fields, replace
from pathlib import Path

from agents.ingredient import Ingredient
from agents.normalization import canonicalize, singularize

# --- Unit conversion ---------------------------------------------------------

#: Exact mass conversions. These are definitions, not measurements: the
#: international pound is exactly 0.45359237 kg.
MASS_TO_GRAMS: dict[str, float] = {
    "mg": 0.001,
    "g": 1.0,
    "kg": 1000.0,
    "oz": 28.349523125,
    "lb": 453.59237,
}

#: Exact volume conversions to millilitres, using US customary measures.
VOLUME_TO_ML: dict[str, float] = {
    "ml": 1.0,
    "cl": 10.0,
    "dl": 100.0,
    "l": 1000.0,
    "tsp": 4.92892159375,
    "tbsp": 14.78676478125,
    "fl oz": 29.5735295625,
    "cup": 236.5882365,
    "pint": 473.176473,
    "quart": 946.352946,
    "qt": 946.352946,
    "gallon": 3785.411784,
}

#: Density in g/ml for ingredients commonly measured by volume. Everything not
#: listed falls back to :data:`DEFAULT_DENSITY`.
DENSITY_G_PER_ML: dict[str, float] = {
    "water": 1.0, "milk": 1.03, "buttermilk": 1.03, "heavy cream": 0.994,
    "yogurt": 1.03, "coconut milk": 0.98, "almond milk": 1.03,
    "olive oil": 0.918, "vegetable oil": 0.92, "sesame oil": 0.92,
    "coconut oil": 0.924, "butter": 0.911, "ghee": 0.9,
    "honey": 1.42, "maple syrup": 1.32, "molasses": 1.4,
    "all-purpose flour": 0.53, "flour": 0.53, "bread flour": 0.53,
    "almond flour": 0.42, "rice flour": 0.58, "cornstarch": 0.63,
    "sugar": 0.845, "brown sugar": 0.93, "powdered sugar": 0.56,
    "salt": 1.217, "baking soda": 0.96, "baking powder": 0.9,
    "rice": 0.85, "oat": 0.41, "breadcrumb": 0.4, "cocoa powder": 0.51,
    "soy sauce": 1.15, "vinegar": 1.01, "wine": 0.99, "broth": 1.0,
    "tomato puree": 1.07, "tomato paste": 1.1, "peanut butter": 1.08,
    "lentil": 0.85, "quinoa": 0.85, "couscous": 0.7, "chickpea": 0.8,
}

#: Water. Wrong for oils and flours, right for the many liquids nobody tabled.
DEFAULT_DENSITY: float = 1.0

#: Grams per item for ingredients counted without a unit ("2 eggs"). USDA
#: reference weights for a medium specimen.
ITEM_WEIGHT_G: dict[str, float] = {
    "egg": 50.0, "egg white": 33.0, "egg yolk": 17.0,
    "onion": 110.0, "red onion": 110.0, "green onion": 15.0, "shallot": 25.0,
    "tomato": 123.0, "potato": 173.0, "sweet potato": 130.0, "carrot": 61.0,
    "celery": 40.0, "cucumber": 301.0, "zucchini": 196.0, "eggplant": 458.0,
    "bell pepper": 119.0, "jalapeno": 14.0, "chili pepper": 45.0,
    "mushroom": 18.0, "avocado": 201.0, "lemon": 58.0, "lime": 67.0,
    "orange": 131.0, "apple": 182.0, "banana": 118.0, "pear": 178.0,
    "peach": 150.0, "garlic": 3.0, "chicken breast": 174.0,
    "chicken thigh": 82.0, "bacon": 28.0, "sausage": 75.0,
    "bay leaf": 0.2, "corn": 90.0,
}

#: Grams per item for a specific (ingredient, unit) pair. Consulted before the
#: generic per-unit weights below, because "a clove of garlic" and "a clove"
#: of anything else are not the same thing.
PAIR_WEIGHT_G: dict[tuple[str, str], float] = {
    ("garlic", "clove"): 3.0,
    ("bread", "slice"): 28.0,
    ("bacon", "slice"): 28.0,
    ("cheese", "slice"): 22.0,
    ("ginger", "cm"): 6.0,
    ("ginger", "inch"): 15.0,
    ("celery", "stalk"): 40.0,
    ("lemongrass", "stalk"): 20.0,
    ("butter", "stick"): 113.0,
    ("cilantro", "bunch"): 25.0,
    ("parsley", "bunch"): 25.0,
    ("basil", "bunch"): 20.0,
    ("mint", "bunch"): 20.0,
    ("garlic", "head"): 45.0,
    ("lettuce", "head"): 500.0,
    ("cabbage", "head"): 900.0,
}

#: Fallback grams for a countable unit when the pair is not tabled. These are
#: rough by nature; anything using them is reported as approximate.
UNIT_WEIGHT_G: dict[str, float] = {
    "clove": 3.0, "slice": 25.0, "piece": 50.0, "pinch": 0.36, "dash": 0.6,
    "knob": 15.0, "splash": 15.0, "drizzle": 10.0,
    "sprig": 2.0, "handful": 30.0, "bunch": 25.0, "stalk": 40.0,
    "head": 500.0, "stick": 113.0, "fillet": 150.0, "can": 400.0,
    "jar": 340.0, "packet": 200.0, "bottle": 500.0, "box": 400.0,
    "bag": 500.0, "cm": 6.0, "inch": 15.0, "mm": 0.6,
}


@dataclass(frozen=True)
class GramsEstimate:
    """A quantity resolved to grams, with how much to trust it.

    Attributes:
        grams: The converted mass.
        exact: True only for mass units, where the conversion is a definition
            and does not depend on what the ingredient is.
        basis: Which rule produced the number, for display and debugging.
    """

    grams: float
    exact: bool
    basis: str


def to_grams(
    quantity: float | None, unit: str | None, name: str
) -> GramsEstimate | None:
    """Convert an ingredient quantity to grams.

    Args:
        quantity: Amount, or ``None`` when the user was vague ("some rice").
        unit: Canonical unit token, or ``None`` for countable items.
        name: Canonical ingredient name, needed for density and item weight.

    Returns:
        The estimate, or ``None`` when no rule applies -- an unknown unit, an
        uncountable ingredient with no unit, or a missing quantity. ``None``
        means "cannot say", never "zero".

    >>> to_grams(1, "lb", "butter").grams
    453.59237
    >>> to_grams(2, None, "egg").grams
    100.0
    """
    if quantity is None or quantity <= 0:
        return None

    if unit in MASS_TO_GRAMS:
        return GramsEstimate(quantity * MASS_TO_GRAMS[unit], True, "mass")

    if unit in VOLUME_TO_ML:
        millilitres = quantity * VOLUME_TO_ML[unit]
        density = DENSITY_G_PER_ML.get(name)
        basis = "volume" if density is not None else "volume (assumed water)"
        return GramsEstimate(millilitres * (density or DEFAULT_DENSITY), False, basis)

    if unit is not None:
        pair = PAIR_WEIGHT_G.get((name, unit))
        if pair is not None:
            return GramsEstimate(quantity * pair, False, "item weight")
        generic = UNIT_WEIGHT_G.get(unit)
        if generic is not None:
            return GramsEstimate(quantity * generic, False, "typical unit weight")
        return None

    item = ITEM_WEIGHT_G.get(name)
    if item is not None:
        return GramsEstimate(quantity * item, False, "item weight")
    return None


# --- Nutrition facts ---------------------------------------------------------


@dataclass(frozen=True)
class NutritionFacts:
    """Nutrients for a given amount of food. All fields are additive."""

    calories: float = 0.0
    protein_g: float = 0.0
    fat_g: float = 0.0
    carbs_g: float = 0.0
    fiber_g: float = 0.0
    sugar_g: float = 0.0
    sodium_mg: float = 0.0

    def scaled(self, factor: float) -> NutritionFacts:
        """Multiply every nutrient by ``factor``.

        Scaling is linear by definition, which is what makes serving math
        checkable: ``facts.scaled(1) == facts``, and scaling by ``n`` then by
        ``1/n`` returns the original.
        """
        return replace(
            self,
            **{f.name: getattr(self, f.name) * factor for f in fields(self)},
        )

    def __add__(self, other: NutritionFacts) -> NutritionFacts:
        """Sum two sets of facts field by field."""
        return replace(
            self,
            **{
                f.name: getattr(self, f.name) + getattr(other, f.name)
                for f in fields(self)
            },
        )

    def rounded(self, digits: int = 1) -> NutritionFacts:
        """Round every nutrient, for display only -- never for further math."""
        return replace(
            self,
            **{f.name: round(getattr(self, f.name), digits) for f in fields(self)},
        )

    def as_dict(self) -> dict[str, float]:
        """Plain dict, for the composer's JSON payload."""
        return {f.name: getattr(self, f.name) for f in fields(self)}


#: USDA nutrient ids. Foundation foods report energy under the Atwater ids
#: (2047/2048) while SR Legacy uses 1008, so all three are accepted in that
#: order of preference.
ENERGY_NUTRIENT_IDS: tuple[int, ...] = (1008, 2048, 2047)
NUTRIENT_IDS: dict[str, int] = {
    "protein_g": 1003,
    "fat_g": 1004,
    "carbs_g": 1005,
    "fiber_g": 1079,
    "sugar_g": 2000,
    "sodium_mg": 1093,
}


@dataclass(frozen=True)
class IngredientNutrition:
    """One ingredient's contribution to the recipe total.

    Attributes:
        name: Canonical ingredient name.
        grams: Mass used for the calculation.
        facts: The ingredient's contribution at that mass.
        source_description: The USDA food actually matched, so a user can see
            that "onion" was priced as "Onions, raw".
        exact_mass: Whether the gram conversion was exact.
        basis: How the mass was derived.
    """

    name: str
    grams: float
    facts: NutritionFacts
    source_description: str
    exact_mass: bool
    basis: str


@dataclass(frozen=True)
class NutritionReport:
    """Nutrition for a whole recipe, plus what could not be measured.

    Attributes:
        servings: Number of servings the recipe yields.
        contributions: Per-ingredient breakdown, in input order.
        unestimated: Ingredients skipped, each with a short reason.
    """

    servings: int
    contributions: tuple[IngredientNutrition, ...] = ()
    unestimated: tuple[tuple[str, str], ...] = ()

    @property
    def total(self) -> NutritionFacts:
        """Sum of every contribution."""
        total = NutritionFacts()
        for contribution in self.contributions:
            total = total + contribution.facts
        return total

    @property
    def per_serving(self) -> NutritionFacts:
        """Total divided evenly across servings."""
        return self.total.scaled(1.0 / self.servings)

    @property
    def coverage(self) -> float:
        """Fraction of ingredients that could be measured, in ``[0, 1]``."""
        counted = len(self.contributions)
        total = counted + len(self.unestimated)
        return counted / total if total else 1.0


# --- USDA client -------------------------------------------------------------

SEARCH_URL = "https://api.nal.usda.gov/fdc/v1/foods/search"


class NutritionLookupError(RuntimeError):
    """A lookup failed for a transient reason: network, rate limit, or 5xx.

    Distinct from "USDA has no match" on purpose. A rate-limited run must not
    be recorded as a permanent absence of data, and must not look to the user
    like a recipe with no nutrition.
    """

#: Words that mark a USDA entry as a prepared product rather than the raw
#: ingredient a recipe means. Used only to break ties, never to exclude.
_PROCESSED_HINTS: frozenset[str] = frozenset({
    "lunchmeat", "breaded", "battered", "canned", "fried", "prepared", "mix",
    "soup", "dehydrated", "infant", "baby", "ring", "roll", "frozen",
    "restaurant", "fast", "sandwich", "pizza", "school", "reduced", "tender",
    "nugget", "patty", "snack", "candy", "dessert", "flavored", "beverage",
})

_WORD = re.compile(r"[^a-z0-9]+")


def _rank_tokens(text: str) -> list[str]:
    """Lowercase and singularize, and nothing else.

    Deliberately *not* :func:`canonicalize`: its typo corrector rewrites USDA
    wording ("breaded" becomes "bread"), and its prep-word stripping deletes
    exactly the qualifiers -- "raw", "canned" -- that make one entry a better
    match than another.
    """
    return [singularize(t) for t in _WORD.split(text.lower()) if t]


def _extract_facts(food: dict) -> NutritionFacts | None:
    """Pull the nutrients we track out of one USDA search result.

    Returns ``None`` when the entry has no energy value at all, which happens
    for a handful of Foundation foods and makes them useless to us.
    """
    values = {
        nutrient.get("nutrientId"): nutrient.get("value")
        for nutrient in food.get("foodNutrients", ())
        if nutrient.get("value") is not None
    }

    calories = next(
        (values[nid] for nid in ENERGY_NUTRIENT_IDS if nid in values), None
    )
    if calories is None:
        return None

    return NutritionFacts(
        calories=float(calories),
        **{
            field: float(values.get(nid, 0.0))
            for field, nid in NUTRIENT_IDS.items()
        },
    )


def _rank_score(query: str, description: str) -> float:
    """Score a candidate description against the query. Lower is better.

    USDA's own relevance ordering is poor for recipe ingredients -- searching
    "olive oil" ranks "Oil, corn, peanut, and olive" above "Oil, olive". This
    prefers descriptions that say little beyond the query, and that say the
    query early.
    """
    query_tokens = _rank_tokens(query)
    description_tokens = _rank_tokens(description)

    score = 0.0
    for token in query_tokens:
        if token in description_tokens:
            score += 0.1 * description_tokens.index(token)
        else:
            score += 25.0  # a missing query term is close to disqualifying

    extra = [t for t in description_tokens if t not in query_tokens]
    score += len(extra)
    score += 2.0 * len(set(extra) & _PROCESSED_HINTS)
    if "raw" in description_tokens:
        score -= 1.5
    return score


class FoodDataCentralClient:
    """Looks up per-100g nutrition facts, with an on-disk cache.

    The cache matters more than it looks: the free tier is rate limited, and
    a 12-ingredient recipe is 12 requests. Cached lookups make repeat runs and
    the Streamlit UI usable, and make the corpus reproducible.
    """

    def __init__(
        self,
        api_key: str | None = None,
        session=None,
        cache_path: Path | None = None,
        timeout: float = 20.0,
    ) -> None:
        """Build a client.

        Args:
            api_key: USDA key. Read from ``USDA_API_KEY`` when omitted.
            session: Anything with ``.get(url, params=, timeout=)``. Defaults
                to ``requests``; tests inject a fake.
            cache_path: JSON cache location. Omit to disable caching.
            timeout: Per-request timeout in seconds.
        """
        from config import require, settings

        self._api_key = api_key or require(settings.usda_api_key, "USDA_API_KEY")
        if session is None:
            import requests

            session = requests.Session()
        self._session = session
        self._timeout = timeout
        self._cache_path = cache_path
        self._cache = self._load_cache()

    @classmethod
    def default(cls) -> FoodDataCentralClient:
        """Build a client wired to the configured key and the standard cache.

        Raises:
            ConfigError: If ``USDA_API_KEY`` is not set.
        """
        from config import NUTRITION_CACHE_PATH

        return cls(cache_path=NUTRITION_CACHE_PATH)

    def _load_cache(self) -> dict[str, dict]:
        """Read the cache file, tolerating absence or corruption."""
        if not self._cache_path or not self._cache_path.exists():
            return {}
        try:
            return json.loads(self._cache_path.read_text())
        except (json.JSONDecodeError, OSError):
            return {}

    def _save_cache(self) -> None:
        """Persist the cache, ignoring write failures."""
        if not self._cache_path:
            return
        try:
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            self._cache_path.write_text(json.dumps(self._cache, indent=2))
        except OSError:
            pass

    def lookup(self, name: str) -> tuple[NutritionFacts, str] | None:
        """Return per-100g facts for ``name`` and the USDA food matched.

        Returns ``None`` when USDA has no usable match, which the agent
        reports rather than treating as zero.
        """
        key = canonicalize(name)
        if not key:
            return None
        if key in self._cache:
            cached = self._cache[key]
            if cached is None:
                return None
            return NutritionFacts(**cached["facts"]), cached["description"]

        # A NutritionLookupError propagates uncached: it means "ask again
        # later", and writing it down would poison the cache permanently.
        result = self._fetch(key)
        self._cache[key] = (
            None
            if result is None
            else {"facts": result[0].as_dict(), "description": result[1]}
        )
        self._save_cache()
        return result

    def _fetch(self, query: str) -> tuple[NutritionFacts, str] | None:
        """Search USDA and return the best usable candidate.

        Returns:
            The best match, or ``None`` when USDA answered but had nothing
            usable -- a definitive result, safe to cache.

        Raises:
            NutritionLookupError: On any transient failure. Deliberately not
                folded into the ``None`` case: a 429 is not an absence of data.
        """
        try:
            response = self._session.get(
                SEARCH_URL,
                params={
                    "query": query,
                    "api_key": self._api_key,
                    "pageSize": 10,
                    "dataType": ["Foundation", "SR Legacy"],
                },
                timeout=self._timeout,
            )
            response.raise_for_status()
            foods = response.json().get("foods") or []
        except Exception as exc:  # noqa: BLE001 - the client library's error
            # taxonomy is not ours to depend on; every failure here is
            # transient from the caller's point of view.
            raise NutritionLookupError(f"USDA lookup for {query!r} failed") from exc

        candidates = []
        for food in foods:
            facts = _extract_facts(food)
            if facts is None:
                continue
            description = food.get("description", "")
            candidates.append((_rank_score(query, description), description, facts))

        if not candidates:
            return None
        _, description, facts = min(candidates, key=lambda c: (c[0], c[1]))
        return facts, description


# --- Agent -------------------------------------------------------------------


class NutritionAgent:
    """Totals and scales recipe nutrition. Arithmetic only, no model calls."""

    def __init__(self, client: FoodDataCentralClient | None = None) -> None:
        """Args: client: Nutrition source. Without one, nothing is estimated."""
        self._client = client

    def analyze(
        self, ingredients: Iterable[Ingredient], *, servings: int = 4
    ) -> NutritionReport:
        """Compute nutrition for a list of ingredients.

        Args:
            ingredients: Parsed ingredients with quantities and units.
            servings: How many servings the recipe yields. Must be positive.

        Returns:
            A report whose ``total`` is the sum of measurable ingredients and
            whose ``unestimated`` names the rest with a reason.

        Raises:
            ValueError: If ``servings`` is not positive.
        """
        if servings <= 0:
            raise ValueError("servings must be positive")

        contributions, unestimated = [], []
        for item in ingredients:
            mass = to_grams(item.quantity, item.unit, item.name)
            if mass is None:
                unestimated.append((item.name, "no quantity or unknown unit"))
                continue
            if self._client is None:
                unestimated.append((item.name, "no nutrition source configured"))
                continue
            try:
                looked_up = self._client.lookup(item.name)
            except NutritionLookupError:
                unestimated.append((item.name, "lookup unavailable"))
                continue
            if looked_up is None:
                unestimated.append((item.name, "no USDA match"))
                continue

            per_100g, description = looked_up
            contributions.append(
                IngredientNutrition(
                    name=item.name,
                    grams=mass.grams,
                    facts=per_100g.scaled(mass.grams / 100.0),
                    source_description=description,
                    exact_mass=mass.exact,
                    basis=mass.basis,
                )
            )

        return NutritionReport(
            servings=servings,
            contributions=tuple(contributions),
            unestimated=tuple(unestimated),
        )

    def scale_to_servings(
        self, report: NutritionReport, servings: int
    ) -> NutritionReport:
        """Re-express a report for a different serving count.

        The totals are unchanged -- only how they are divided. Rescaling never
        recomputes anything, so it cannot drift from the original numbers.
        """
        if servings <= 0:
            raise ValueError("servings must be positive")
        return replace(report, servings=servings)


def scale_ingredients(
    ingredients: Sequence[Ingredient], factor: float
) -> list[Ingredient]:
    """Scale ingredient quantities by ``factor``, e.g. to double a recipe.

    Ingredients with no quantity are passed through untouched: there is nothing
    to scale, and inventing a number would be worse than leaving it vague.
    """
    if factor <= 0:
        raise ValueError("factor must be positive")
    return [
        item
        if item.quantity is None
        else item.model_copy(update={"quantity": item.quantity * factor})
        for item in ingredients
    ]
