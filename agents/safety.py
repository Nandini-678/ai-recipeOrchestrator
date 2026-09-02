"""Safety agent: allergen filtering and ingredient substitution.

Two jobs, kept separate on purpose.

**Allergen filtering is hard-coded.** Not a prompt, not a model call -- a
lookup table over canonical ingredient names. A rule you can unit test is worth
more than one that can be argued out of, and this is the one place in the
pipeline where being wrong hurts someone. The tables here are the whole
mechanism, so they are written to be read and audited.

**Substitution is table-first.** A static table covers the common cases with
exact ratios; the LLM is consulted only when the table has nothing. Whatever
the model suggests is then screened by the same hard-coded allergen check
before it is returned -- a substitution advisor is never trusted on safety.

Matching is token-based over canonical names, which is why "eggplant" is not an
egg and "butternut squash" is not butter: those are single tokens that simply
do not equal the markers. Where a name genuinely does contain a marker token
but is not that allergen ("almond milk", "peanut butter"), it is listed
explicitly in :data:`ALLERGEN_SAFE`.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from pydantic import BaseModel, ValidationError, field_validator

from agents.ingredient import Ingredient
from agents.normalization import canonicalize
from agents.retrieval import RecipeMatch

# --- Allergen tables ---------------------------------------------------------

#: Marker tokens that identify an allergen. A canonical ingredient name belongs
#: to an allergen if any of its *whole tokens* appears here, or if the full name
#: does. Token equality (not substring) is what keeps "nutmeg" out of tree nuts
#: and "eggplant" out of eggs.
ALLERGEN_MARKERS: dict[str, frozenset[str]] = {
    "milk": frozenset({
        "milk", "cheese", "butter", "cream", "yogurt", "ghee", "curd",
        "custard", "mozzarella", "parmesan", "cheddar", "feta", "ricotta",
        "mascarpone", "buttermilk", "whey", "casein", "creme", "paneer",
    }),
    "egg": frozenset({
        "egg", "mayonnaise", "meringue", "albumin", "aioli", "custard",
    }),
    "peanut": frozenset({"peanut", "groundnut"}),
    "tree nut": frozenset({
        "almond", "cashew", "walnut", "pecan", "pistachio", "hazelnut",
        "macadamia", "praline", "marzipan", "nutella", "amaretto", "chestnut",
        "brazil", "pine",
        # The FDA classifies coconut as a tree nut. Most coconut-allergic
        # reactions are unrelated to tree nuts, but over-flagging is the safe
        # error for an allergen filter, so it stays in.
        "coconut",
    }),
    "soy": frozenset({
        "soy", "soya", "soybean", "tofu", "edamame", "miso", "tempeh",
        "tamari",
    }),
    "wheat": frozenset({
        "wheat", "flour", "bread", "breadcrumb", "pasta", "spaghetti",
        "noodle", "couscous", "semolina", "cracker", "biscuit", "pastry",
        "phyllo", "filo", "tortilla", "barley", "rye", "seitan", "bulgur",
        "farro", "orzo", "macaroni", "penne", "lasagne", "baguette", "brioche",
        "udon", "ramen", "wonton", "croissant", "pretzel", "pita", "focaccia",
        "ciabatta", "panko", "matzo", "dumpling", "linguine", "paccheri",
    }),
    "fish": frozenset({
        "fish", "salmon", "tuna", "cod", "anchovy", "sardine", "mackerel",
        "haddock", "halibut", "trout", "tilapia", "bass", "worcestershire",
    }),
    "shellfish": frozenset({
        "shrimp", "prawn", "crab", "lobster", "mussel", "clam", "oyster",
        "scallop", "squid", "calamari", "octopus", "crayfish", "langoustine",
    }),
    "sesame": frozenset({"sesame", "tahini"}),
}

#: Every allergen this agent can screen for.
KNOWN_ALLERGENS: frozenset[str] = frozenset(ALLERGEN_MARKERS)

#: How each category is named to a person. The internal names are chosen for
#: matching ("milk", "wheat"); nobody says "I avoid wheat" when they mean
#: gluten, so anything user-facing shows these instead.
ALLERGEN_LABELS: dict[str, str] = {
    "milk": "Dairy",
    "egg": "Eggs",
    "peanut": "Peanuts",
    "tree nut": "Tree nuts",
    "soy": "Soy",
    "wheat": "Gluten",
    "fish": "Fish",
    "shellfish": "Shellfish",
    "sesame": "Sesame",
}

#: Genuine exceptions to the marker rule: names that carry an allergen's marker
#: token but are not that allergen, and that the generic-marker rule below
#: cannot derive. Deliberately short -- every entry here is load-bearing, and a
#: test asserts as much, so a redundant one cannot quietly accumulate.
ALLERGEN_SAFE: dict[str, frozenset[str]] = {
    # vegan egg *replacements*, which contain no egg at all
    "egg": frozenset({"flax egg", "flaxseed egg", "chia egg"}),
    # an aquatic corm and a mushroom variety, despite their names
    "tree nut": frozenset({"water chestnut", "chestnut mushroom"}),
    "shellfish": frozenset({"oyster mushroom"}),
}


#: Some markers name a *container* rather than the allergen: flour, noodle,
#: butter and cream all describe a form that any number of ingredients take.
#: Listing every safe combination by hand does not scale -- "rice noodle" would
#: not cover "brown rice noodle" or "rice stick noodle" -- so each generic
#: marker is paired with the source tokens that make it allergen-free.
#: Maps allergen -> (generic markers, tokens that negate them).
GENERIC_MARKERS: dict[str, tuple[frozenset[str], frozenset[str]]] = {
    "milk": (
        frozenset({"butter", "cream", "milk", "yogurt"}),
        frozenset({
            "almond", "soy", "soya", "oat", "rice", "cashew", "coconut",
            "peanut", "sunflower", "cocoa", "shea", "apple", "hemp", "bean",
            "tartar", "macadamia", "pistachio", "seed",
        }),
    ),
    "wheat": (
        frozenset({"flour", "noodle", "pasta", "bread", "tortilla", "cracker"}),
        frozenset({
            "rice", "corn", "almond", "coconut", "chickpea", "cassava",
            "tapioca", "buckwheat", "quinoa", "potato", "besan", "oat",
            "millet", "sorghum", "gram", "glass", "soba",
        }),
    ),
}


#: Allergens that cannot be inferred from an ingredient's name at all. These
#: are the ones people miss: standard soy sauce is brewed with wheat,
#: Worcestershire contains anchovy, pesto carries both parmesan and pine nuts.
#: A name-based rule can never find these, so they are stated outright.
HIDDEN_ALLERGENS: dict[str, frozenset[str]] = {
    "soy sauce": frozenset({"soy", "wheat"}),
    "teriyaki sauce": frozenset({"soy", "wheat"}),
    "hoisin sauce": frozenset({"soy", "wheat"}),
    "oyster sauce": frozenset({"shellfish", "soy", "wheat"}),
    "worcestershire sauce": frozenset({"fish"}),
    "pesto": frozenset({"milk", "tree nut"}),
    "caesar dressing": frozenset({"fish", "egg", "milk"}),
    "imitation crab": frozenset({"fish", "shellfish", "wheat"}),
    "surimi": frozenset({"fish", "shellfish", "wheat"}),
    "tempura": frozenset({"wheat", "egg"}),
    "gravy": frozenset({"wheat"}),
    "roux": frozenset({"wheat", "milk"}),
    "bechamel": frozenset({"wheat", "milk"}),
    "nougat": frozenset({"tree nut", "egg"}),
    "marzipan": frozenset({"tree nut"}),
}


def detect_allergens(name: str) -> frozenset[str]:
    """Return the allergens a single ingredient belongs to.

    Args:
        name: An ingredient name. Canonicalized defensively, so this is safe to
            call on raw text as well as on parsed output.

    Returns:
        The matching allergen categories, empty if none.

    >>> sorted(detect_allergens("cheddar cheese"))
    ['milk']
    >>> sorted(detect_allergens("eggplant"))
    []
    >>> sorted(detect_allergens("almond milk"))
    ['tree nut']
    >>> sorted(detect_allergens("soy sauce"))
    ['soy', 'wheat']
    """
    canonical = canonicalize(name)
    if not canonical:
        return frozenset()

    tokens = set(canonical.split())
    found = set()
    for allergen, markers in ALLERGEN_MARKERS.items():
        if canonical in ALLERGEN_SAFE.get(allergen, frozenset()):
            continue
        matched = tokens & markers
        if canonical not in markers and not matched:
            continue
        if _is_negated(allergen, canonical, tokens, matched):
            continue
        found.add(allergen)

    # Stated allergens always apply: they exist precisely because the name
    # gives no clue, so no marker rule could have found them.
    found |= HIDDEN_ALLERGENS.get(canonical, frozenset())
    return frozenset(found)


def _is_negated(
    allergen: str, canonical: str, tokens: set[str], matched: set[str]
) -> bool:
    """True when a generic marker is qualified into being allergen-free.

    "Rice noodle" matches the wheat marker ``noodle``, but ``rice`` says what
    the noodle is made of. This only applies when *every* marker that matched
    was generic: "egg noodle" is still wheat, and "bread flour" still wheat
    however it is qualified.
    """
    generic, negating = GENERIC_MARKERS.get(allergen, (frozenset(), frozenset()))
    if not matched or canonical in ALLERGEN_MARKERS[allergen]:
        return False
    return bool(matched <= generic and tokens & negating)


def normalize_allergens(avoid: Iterable[str]) -> frozenset[str]:
    """Canonicalize a user's allergen list onto :data:`KNOWN_ALLERGENS`.

    Accepts the common spellings people actually type ("dairy", "nuts",
    "gluten", "shellfish") and maps them onto the internal category names.
    Unrecognized entries are dropped rather than silently ignored downstream.
    """
    synonyms = {
        "dairy": "milk", "lactose": "milk", "eggs": "egg", "peanuts": "peanut",
        "nuts": "tree nut", "tree nuts": "tree nut", "treenut": "tree nut",
        "gluten": "wheat", "soya": "soy", "shell fish": "shellfish",
        "seafood": "shellfish",
    }
    resolved = set()
    for entry in avoid:
        key = " ".join(str(entry).lower().split())
        key = synonyms.get(key, key)
        if key in KNOWN_ALLERGENS:
            resolved.add(key)
    return frozenset(resolved)


# --- Substitution table ------------------------------------------------------


@dataclass(frozen=True)
class Substitution:
    """A proposed replacement for one ingredient.

    Attributes:
        original: The ingredient being replaced.
        replacement: What to use instead, as a canonical name.
        ratio: Multiply the original quantity by this to get the replacement's.
        unit: Canonical unit the replacement is measured in, when it differs
            from the original's. Four eggs become four *tablespoons* of ground
            flaxseed, not four flaxseeds; without this the quantity is
            nonsense and the nutrition agent cannot weigh it.
        note: Short human-readable caveat, shown in the final recipe.
        source: ``"table"`` for a curated entry, ``"llm"`` for a model
            suggestion that passed the allergen screen.
        in_pantry: True when the user already has the replacement.
    """

    original: str
    replacement: str
    ratio: float = 1.0
    unit: str | None = None
    note: str = ""
    source: str = "table"
    in_pantry: bool = False


#: Curated substitutions, best candidate first, as
#: ``(replacement, ratio, unit, note)``. A unit of ``None`` keeps the
#: original's. Ratios are by volume unless the note says otherwise.
SUBSTITUTIONS: dict[str, tuple[tuple[str, float, str | None, str], ...]] = {
    "butter": (
        ("olive oil", 0.75, None, "use three quarters as much"),
        ("coconut oil", 1.0, None, "adds a faint coconut note"),
    ),
    "milk": (
        ("oat milk", 1.0, None, ""),
        ("almond milk", 1.0, None, ""),
        ("soy milk", 1.0, None, ""),
        ("coconut milk", 1.0, None, "richer, slightly sweet"),
    ),
    "heavy cream": (
        ("coconut cream", 1.0, None, "dairy free"),
        ("evaporated milk", 1.0, None, "less rich"),
    ),
    "sour cream": (("yogurt", 1.0, None, "thicker if strained"),),
    "yogurt": (
        ("sour cream", 1.0, None, ""),
        ("coconut yogurt", 1.0, None, "dairy free"),
    ),
    "buttermilk": (("milk", 1.0, None, "add 1 tbsp lemon juice per cup, rest 5 min"),),
    "parmesan cheese": (
        ("pecorino", 1.0, None, "sharper"),
        ("nutritional yeast", 0.5, None, "vegan, use half as much"),
    ),
    "egg": (
        ("flaxseed", 1.0, "tbsp", "1 tbsp ground flaxseed + 3 tbsp water per egg"),
        ("applesauce", 60.0, "ml", "60ml per egg; baking only"),
        ("banana", 0.5, None, "half a mashed banana per egg; baking only"),
    ),
    "all-purpose flour": (
        ("oat flour", 1.0, None, "gluten free; slightly denser"),
        ("rice flour", 1.0, None, "gluten free"),
        ("almond flour", 1.0, None, "gluten free; do not use for bread"),
    ),
    "breadcrumb": (
        ("oat", 1.0, None, "pulse first"),
        ("almond flour", 1.0, None, "gluten free"),
    ),
    "pasta": (("rice noodle", 1.0, None, "gluten free"),),
    "soy sauce": (
        ("tamari", 1.0, None, "gluten free"),
        ("coconut amino", 1.0, None, "soy free, milder and sweeter"),
    ),
    "peanut butter": (
        ("sunflower seed butter", 1.0, None, "nut free"),
        ("almond butter", 1.0, None, "still a tree nut"),
    ),
    "honey": (
        ("maple syrup", 1.0, None, "vegan"),
        ("sugar", 0.75, None, "add a splash of water"),
    ),
    "sugar": (("honey", 0.75, None, ""), ("maple syrup", 0.75, None, "")),
    "lemon juice": (("lime juice", 1.0, None, ""), ("vinegar", 0.5, None, "sharper")),
    "cilantro": (("parsley", 1.0, None, "milder, no citrus note"),),
    "shallot": (("onion", 0.5, None, "use half; stronger"),),
    "green onion": (("chive", 1.0, None, ""), ("onion", 0.5, None, "cook it first")),
    "cornstarch": (("all-purpose flour", 2.0, None, "use twice as much"),),
    "wine": (("broth", 1.0, None, "add 1 tsp vinegar for acidity"),),
    "broth": (
        ("bouillon", 1.0, None, "dissolve one cube in the same volume of water"),
    ),
    "shrimp": (("chicken breast", 1.0, None, ""), ("tofu", 1.0, None, "vegetarian")),
    "rice": (("quinoa", 1.0, None, ""), ("couscous", 1.0, None, "contains wheat")),
    "vegetable oil": (("olive oil", 1.0, None, ""), ("coconut oil", 1.0, None, "")),
    "ginger": (("ginger paste", 1.0, None, "1 tsp paste per 1 tsp fresh"),),
    "ginger paste": (("ginger", 1.0, None, "1 tsp fresh grated per 1 tsp paste"),),
    # Coverage for what real pantries most often lack, added after watching
    # the advisor propose dairy milk for coconut milk. A curated answer that
    # keeps the dish intact beats a plausible one that changes it.
    "coconut milk": (
        ("coconut cream", 0.75, None, "thin with water to taste"),
        ("heavy cream", 1.0, None, "loses the coconut note; not dairy free"),
    ),
    "bay leaf": (("thyme", 0.5, None, "different but works in a braise"),),
    "turmeric": (("curry powder", 1.0, None, "already contains turmeric"),),
    "paprika": (("chili powder", 0.5, None, "hotter; use less"),),
    "sea salt": (("salt", 1.0, None, ""),),
    "kosher salt": (("salt", 0.75, None, "finer grain, so use less"),),
    "vegetable stock": (
        ("broth", 1.0, None, ""),
        ("bouillon", 1.0, None, "dissolve in water"),
    ),
    "beef stock": (
        ("broth", 1.0, None, ""),
        ("bouillon", 1.0, None, "dissolve in water"),
    ),
    "chicken stock": (
        ("broth", 1.0, None, ""),
        ("bouillon", 1.0, None, "dissolve in water"),
    ),
    "baguette": (("bread", 1.0, None, ""),),
    "sunflower oil": (("vegetable oil", 1.0, None, ""), ("olive oil", 1.0, None, "")),
    "basil leaf": (("basil", 1.0, None, ""), ("parsley", 1.0, None, "milder")),
    "tomato puree": (("tomato paste", 0.5, None, "thicker; dilute with water"),),
    "lime": (("lemon", 1.0, None, "sharper, less floral"),),
    "red chili": (("chili powder", 0.25, None, "use a quarter as much"),),
    "sesame seed": (("sunflower seed", 1.0, None, "less aromatic"),),
    "red pepper": (("bell pepper", 1.0, None, ""),),
    "cabbage": (("lettuce", 1.0, None, "raw only; will not hold up to cooking"),),
    "bread": (("bread roll", 1.0, None, ""),),
}

#: Substituting a handful of things is help; substituting most of the list
#: quietly turns the recipe into a different dish. Past this count the
#: remaining gaps are reported as unresolved so the user can decide.
MAX_SUBSTITUTIONS_PER_RECIPE = 3


class SubstitutionAdvisor:
    """Protocol for the LLM fallback: name in, replacement name out or None."""

    def __call__(self, missing: str, avoid: frozenset[str]) -> Substitution | None:
        """Suggest a replacement for ``missing``, or ``None`` if it has none."""
        raise NotImplementedError


def suggest_substitution(
    missing: str,
    *,
    pantry: Iterable[str] = (),
    avoid: Iterable[str] = (),
    advisor: SubstitutionAdvisor | None = None,
) -> Substitution | None:
    """Find a safe replacement for one missing ingredient.

    Preference order: something the user already has, then the best curated
    candidate, then an advisor suggestion. Every candidate -- curated or
    model-supplied -- is screened against ``avoid`` before it can be returned.

    Args:
        missing: Canonical name of the ingredient to replace.
        pantry: Canonical names the user has, used to prefer a candidate they
            can act on immediately.
        avoid: Allergens to screen against, in any accepted spelling.
        advisor: Optional LLM fallback, consulted only when the table has no
            safe candidate.

    Returns:
        The chosen :class:`Substitution`, or ``None`` when nothing safe exists.
    """
    allergens = normalize_allergens(avoid)
    have = set(pantry)

    safe = [
        Substitution(
            original=missing,
            replacement=replacement,
            ratio=ratio,
            unit=unit,
            note=note,
            source="table",
            in_pantry=replacement in have,
        )
        for replacement, ratio, unit, note in SUBSTITUTIONS.get(missing, ())
        if not (detect_allergens(replacement) & allergens)
    ]
    if safe:
        # A candidate the user already has beats a better one they do not.
        return next((s for s in safe if s.in_pantry), safe[0])

    if advisor is None:
        return None

    suggestion = advisor(missing, allergens)
    if suggestion is None:
        return None
    # The advisor is never trusted on safety: re-screen its answer.
    if detect_allergens(suggestion.replacement) & allergens:
        return None
    return Substitution(
        original=missing,
        replacement=suggestion.replacement,
        ratio=suggestion.ratio,
        unit=suggestion.unit,
        note=suggestion.note,
        source="llm",
        in_pantry=suggestion.replacement in have,
    )


# --- Screening ---------------------------------------------------------------


@dataclass(frozen=True)
class ScreenedMatch:
    """A retrieval match after allergen screening and substitution planning.

    Attributes:
        match: The underlying retrieval result.
        violations: Ingredient name -> the allergens it triggers. Empty when
            the recipe is safe as written.
        substitutions: Replacements for the ingredients the pantry lacks.
        unresolved: Missing ingredients with no safe substitute.
    """

    match: RecipeMatch
    violations: dict[str, frozenset[str]]
    substitutions: tuple[Substitution, ...] = ()
    unresolved: tuple[str, ...] = ()

    @property
    def is_safe(self) -> bool:
        """True when no ingredient triggers an avoided allergen."""
        return not self.violations

    @property
    def recipe(self):
        """Convenience passthrough to the underlying recipe."""
        return self.match.recipe


def apply_substitutions(screened: ScreenedMatch) -> list[Ingredient]:
    """Return a recipe's ingredients with the planned substitutions applied.

    The nutrition agent must be given *this* list, not the recipe's original
    one: a recipe whose eggs became flaxseed has different nutrition, and
    computing it from the pre-substitution ingredients would describe a dish
    nobody is going to cook.

    Quantities are scaled by each substitution's ratio; ingredients with no
    quantity stay unquantified.
    """
    by_original = {s.original: s for s in screened.substitutions}
    resolved = []
    for item in screened.recipe.ingredients:
        replacement = by_original.get(item.name)
        if replacement is None:
            resolved.append(item)
            continue
        quantity = (
            item.quantity * replacement.ratio if item.quantity is not None else None
        )
        resolved.append(
            item.model_copy(
                update={
                    "name": replacement.replacement,
                    "quantity": quantity,
                    "unit": replacement.unit or item.unit,
                }
            )
        )
    return resolved


class SafetyAgent:
    """Screens retrieval matches for allergens and plans substitutions."""

    def __init__(
        self,
        advisor: SubstitutionAdvisor | None = None,
        max_substitutions: int = MAX_SUBSTITUTIONS_PER_RECIPE,
    ) -> None:
        """Build the agent.

        Args:
            advisor: Optional LLM fallback for uncovered substitutions.
            max_substitutions: Most replacements to apply to one recipe;
                beyond this, gaps are reported as unresolved.
        """
        self._advisor = advisor
        self._max_substitutions = max_substitutions

    def find_violations(
        self, recipe, avoid: frozenset[str]
    ) -> dict[str, frozenset[str]]:
        """Map each of ``recipe``'s ingredients that triggers ``avoid``."""
        violations = {}
        for ingredient in recipe.ingredients:
            triggered = detect_allergens(ingredient.name) & avoid
            if triggered:
                violations[ingredient.name] = triggered
        return violations

    def screen(
        self,
        matches: Sequence[RecipeMatch],
        *,
        pantry: Iterable[str] = (),
        avoid: Iterable[str] = (),
        drop_unsafe: bool = True,
    ) -> list[ScreenedMatch]:
        """Screen matches and plan substitutions for what the pantry lacks.

        Args:
            matches: Retrieval output, best-first.
            pantry: Canonical names the user has.
            avoid: Allergens to screen against.
            drop_unsafe: When true (the default) recipes containing an avoided
                allergen are removed entirely. Set false to keep them with
                their ``violations`` populated, for a UI that wants to explain
                why a recipe was excluded.

        Returns:
            Screened matches in the order given.
        """
        allergens = normalize_allergens(avoid)
        have = set(pantry)

        screened = []
        for match in matches:
            violations = self.find_violations(match.recipe, allergens)
            if violations and drop_unsafe:
                continue

            substitutions, unresolved = [], []
            for missing in match.missing:
                if len(substitutions) >= self._max_substitutions:
                    # Enough already changed; report the rest honestly rather
                    # than rewriting the dish out from under the user.
                    unresolved.append(missing)
                    continue
                found = suggest_substitution(
                    missing, pantry=have, avoid=allergens, advisor=self._advisor
                )
                if found:
                    substitutions.append(found)
                else:
                    unresolved.append(missing)

            screened.append(
                ScreenedMatch(
                    match=match,
                    violations=violations,
                    substitutions=tuple(substitutions),
                    unresolved=tuple(unresolved),
                )
            )
        return screened


# --- LLM fallback ------------------------------------------------------------

ADVISOR_PROMPT = """\
Suggest one common cooking substitute for the ingredient the user names.

Rules:
- Give a single, widely available replacement ingredient.
- Give the conversion ratio as a number: replacement = original * ratio.
- If you know of no reasonable substitute, set "replacement" to null.
- Never suggest an ingredient containing any of the listed allergens.

Respond with JSON only:
{"replacement": "<name>" | null, "ratio": <number>, "note": "<short note>"}\
"""


class _AdvisorResponse(BaseModel):
    """Schema an advisor response must satisfy before it is trusted."""

    replacement: str | None = None
    ratio: float = 1.0
    note: str = ""

    @field_validator("ratio")
    @classmethod
    def _positive(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("ratio must be positive")
        return value


class GroqSubstitutionAdvisor:
    """LLM fallback, used only when the curated table has no safe candidate.

    Its output is schema-validated here and allergen-screened by
    :func:`suggest_substitution` afterwards. Both checks are in code: the
    prompt's own allergen instruction is a courtesy, not the safety mechanism.
    """

    def __init__(self, client=None, model: str | None = None) -> None:
        """Build an advisor.

        Args:
            client: A ``groq.Groq`` instance. Built from config if omitted.
            model: Model id. Falls back to ``GROQ_MODEL`` in the environment.
        """
        from config import require, settings

        if client is None:
            from groq import Groq

            client = Groq(api_key=require(settings.groq_api_key, "GROQ_API_KEY"))
        self._client = client
        self._model = model or settings.groq_model

    def __call__(self, missing: str, avoid: frozenset[str]) -> Substitution | None:
        """Ask the model for a replacement, returning ``None`` on any doubt."""
        avoided = ", ".join(sorted(avoid)) or "none"
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": ADVISOR_PROMPT},
                    {
                        "role": "user",
                        "content": f"Ingredient: {missing}\nAllergens: {avoided}",
                    },
                ],
                response_format={"type": "json_object"},
                temperature=0,
            )
            parsed = _AdvisorResponse.model_validate_json(
                response.choices[0].message.content
            )
        except (ValidationError, json.JSONDecodeError, KeyError, IndexError):
            return None
        except Exception:  # noqa: BLE001 - a failed advisor means "no
            # suggestion", never a failed request. The caller degrades to
            # reporting the ingredient as unresolved.
            return None

        if not parsed.replacement:
            return None
        replacement = canonicalize(parsed.replacement)
        if not replacement or replacement == missing:
            return None
        return Substitution(
            original=missing,
            replacement=replacement,
            ratio=parsed.ratio,
            note=parsed.note.strip(),
            source="llm",
        )
