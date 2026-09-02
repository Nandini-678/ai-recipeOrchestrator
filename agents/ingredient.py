"""Ingredient agent: messy free text in, structured ingredients out.

Design note -- the split of responsibility here is deliberate. A language model
is good at *segmenting* rambling prose ("uhh I've got some chicken, maybe two
eggs and half a bag of rice") into discrete ingredient phrases. It is needlessly
unreliable at turning "1 1/2" into ``1.5`` or "scallions" into "green onion".

So the LLM, when configured, only segments. Every phrase is then structured by
pure Python in :func:`parse_phrase`, and its output is normalized by
:mod:`agents.normalization`. That keeps the quantity math exact, the behaviour
reproducible, and the whole agent unit testable with no API key.

Typical use::

    >>> parse_ingredients("2 chicken breasts, 1/2 cup rice, some brocolli")
    [Ingredient(name='chicken breast', quantity=2.0, unit=None, ...),
     Ingredient(name='rice', quantity=0.5, unit='cup', ...),
     Ingredient(name='broccoli', quantity=None, unit=None, ...)]
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Callable, Iterable

from pydantic import BaseModel, Field, ValidationError, field_validator

from agents.normalization import PREP_MODIFIERS, canonicalize, normalize_unit

#: A callable that splits raw user text into discrete ingredient phrases.
Extractor = Callable[[str], list[str]]


class Ingredient(BaseModel):
    """One parsed ingredient.

    Attributes:
        name: Canonical, matchable ingredient name (lowercase, singular).
        quantity: Amount as a number, or ``None`` when the user was vague
            ("some rice"). Downstream agents must handle ``None``.
        unit: Canonical unit token, or ``None`` for countable items ("2 eggs").
        raw: The original phrase, kept so the critic agent can explain itself.
    """

    name: str = Field(min_length=1)
    quantity: float | None = Field(default=None, gt=0)
    unit: str | None = None
    raw: str = ""

    @field_validator("name")
    @classmethod
    def _non_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("ingredient name cannot be blank")
        return value


class IngredientExtractionError(RuntimeError):
    """Raised when an LLM extractor returns something unusable."""


# --- Quantity parsing --------------------------------------------------------

WORD_NUMBERS: dict[str, float] = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
    "twelve": 12, "dozen": 12, "half": 0.5, "quarter": 0.25,
}

#: Two-word quantity phrases, checked before single words so "a couple" does
#: not get read as the "a" in :data:`WORD_NUMBERS`.
PHRASE_NUMBERS: dict[str, float] = {
    "a couple": 2, "a few": 3, "a dozen": 12, "half a": 0.5, "a half": 0.5,
    "a quarter": 0.25,
}

#: Two-word hedges. Checked before :data:`PHRASE_NUMBERS` and
#: :data:`WORD_NUMBERS` so "a bit of butter" is not read as "a" == 1.
VAGUE_PHRASES: frozenset[str] = frozenset({
    "a bit", "a little", "a lot", "a load", "a splash", "a touch",
})

#: Hedges that signal "I have this but haven't measured it". They are consumed
#: so they never leak into the name, and leave ``quantity`` as ``None``.
VAGUE_QUANTIFIERS: frozenset[str] = frozenset({
    "some", "several", "plenty", "lots", "loads", "any", "bit", "little",
    "leftover", "spare", "maybe", "like", "about", "around", "roughly",
})

_FRACTION = re.compile(r"^(\d+)\s*/\s*(\d+)$")
#: "2-1/2 cups" is two and a half, not a range from 2 down to 1/2.
_MIXED_SLASH = re.compile(r"^(\d+)-(\d+)\s*/\s*(\d+)$")
_MIXED_VULGAR = re.compile(r"^(\d+)([¼-¾⅐-⅞])$")
#: The upper bound may carry a fraction ("1-1½ cups"); it is
#: discarded either way, since ranges resolve to their lower bound.
_RANGE = re.compile(
    r"^(\d+(?:\.\d+)?)\s*[-–]\s*[\d./¼-¾⅐-⅞]+$"
)
_DIGIT_LETTER = re.compile(r"(\d)\s*([a-z])")
#: "8-ounce", "1/2-inch" -> split so the unit can be read positionally.
_COMPOUND_MEASURE = re.compile(r"(\d)-([a-z])")
#: Dual metric/imperial measures ("175g/6oz", "50ml/2fl oz"): the source
#: gives the same amount twice, so keep the first and drop the alternative.
_DUAL_MEASURE = re.compile(
    r"(?<=\d)\s*([a-z]{1,4})\s*/\s*[\d¼-¾⅐-⅞./]+"
    r"\s*(?:fl\s*oz|floz|fl|oz|lb|g|kg|ml|l)\b",
    flags=re.IGNORECASE,
)
_NUMBER = re.compile(r"^\d+(?:\.\d+)?$")


def _vulgar_value(token: str) -> float | None:
    """Return the numeric value of a single vulgar fraction like ``'½'``."""
    if len(token) != 1:
        return None
    try:
        return unicodedata.numeric(token)
    except (TypeError, ValueError):
        return None


def _token_value(token: str) -> float | None:
    """Numeric value of one quantity token, or ``None`` if it isn't one."""
    if _NUMBER.match(token):
        return float(token)
    mixed_slash = _MIXED_SLASH.match(token)
    if mixed_slash:
        denominator = int(mixed_slash.group(3))
        if not denominator:
            return None
        whole, numerator = int(mixed_slash.group(1)), int(mixed_slash.group(2))
        return whole + numerator / denominator
        return float(token)
    fraction = _FRACTION.match(token)
    if fraction:
        denominator = int(fraction.group(2))
        return int(fraction.group(1)) / denominator if denominator else None
    mixed = _MIXED_VULGAR.match(token)
    if mixed:
        part = _vulgar_value(mixed.group(2))
        return int(mixed.group(1)) + part if part is not None else None
    vulgar = _vulgar_value(token)
    if vulgar is not None:
        return vulgar
    return WORD_NUMBERS.get(token)


def _leading_quantity(tokens: list[str]) -> tuple[float | None, int]:
    """Read a quantity off the front of ``tokens``.

    Returns:
        ``(quantity, consumed)`` -- the value (``None`` if the user was vague or
        gave no number) and how many tokens to drop before reading the unit.

    Ranges resolve to their **lower** bound: "2-3 cloves" means you can count on
    two, and over-promising an inventory is the more damaging error here.
    """
    if not tokens:
        return None, 0

    if len(tokens) >= 2:
        phrase = f"{tokens[0]} {tokens[1]}"
        if phrase in VAGUE_PHRASES:
            consumed = 3 if len(tokens) > 2 and tokens[2] == "of" else 2
            return None, consumed
        if phrase in PHRASE_NUMBERS:
            consumed = 3 if len(tokens) > 2 and tokens[2] == "of" else 2
            return PHRASE_NUMBERS[phrase], consumed

    if tokens[0] in VAGUE_QUANTIFIERS:
        consumed = 2 if len(tokens) > 1 and tokens[1] == "of" else 1
        return None, consumed

    if _MIXED_SLASH.match(tokens[0]):
        return _token_value(tokens[0]), 1
    ranged = _RANGE.match(tokens[0])
    if ranged:
        return float(ranged.group(1)), 1
    if (
        len(tokens) >= 3
        and tokens[1] == "to"
        and _NUMBER.match(tokens[0])
        and _NUMBER.match(tokens[2])
    ):
        return float(tokens[0]), 3

    first = _token_value(tokens[0])
    if first is None:
        return None, 0

    # Mixed numbers: "1 1/2 cups" -> 1.5
    if len(tokens) >= 2 and float(first).is_integer():
        second = _token_value(tokens[1])
        if second is not None and 0 < second < 1:
            return first + second, 2

    consumed = 2 if len(tokens) > 1 and tokens[1] == "of" else 1
    return first, consumed


def _leading_unit(tokens: list[str]) -> tuple[str | None, int]:
    """Read a canonical unit off the front of ``tokens``, longest match first."""
    if not tokens:
        return None, 0
    if len(tokens) >= 2:
        two_word = normalize_unit(f"{tokens[0]} {tokens[1]}")
        if two_word:
            consumed = 3 if len(tokens) > 2 and tokens[2] == "of" else 2
            return two_word, consumed
    one_word = normalize_unit(tokens[0])
    if one_word:
        consumed = 2 if len(tokens) > 1 and tokens[1] == "of" else 1
        return one_word, consumed
    return None, 0


def _skip_modifiers(tokens: list[str]) -> list[str]:
    """Drop leading preparation words so the unit reader sees the real unit.

    "Small bunch coriander" hides ``bunch`` behind ``small``; without this the
    unit is never found and "bunch" leaks into the ingredient name.
    """
    index = 0
    while (
        index < len(tokens) - 1
        and tokens[index] in PREP_MODIFIERS
        # Some words are prep noise in a *name* but a real unit in *position*
        # ("cm", "piece", "slice"). Never skip past one of those.
        and not normalize_unit(tokens[index])
    ):
        index += 1
    return tokens[index:]


def _trailing_unit(tokens: list[str]) -> tuple[str | None, list[str]]:
    """Pull a unit off the *end* of a phrase: "3 garlic cloves".

    Only fires when something would remain as the name, so "2 cloves" is left
    for the caller to reject rather than being reduced to a nameless quantity.
    """
    if len(tokens) < 2:
        return None, tokens
    unit = normalize_unit(tokens[-1])
    return (unit, tokens[:-1]) if unit else (None, tokens)


def parse_phrase(phrase: str) -> Ingredient | None:
    """Structure a single ingredient phrase.

    Args:
        phrase: One ingredient as written by the user, e.g. ``"1 1/2 cups of
            plain flour"``.

    Returns:
        The parsed :class:`Ingredient`, or ``None`` if nothing nameable is left
        after stripping quantities and units (e.g. the phrase was ``"and"``).

    >>> parse_phrase("500g maida")
    Ingredient(name='all-purpose flour', quantity=500.0, unit='g', raw='500g maida')
    """
    cleaned = phrase.strip().lower()
    cleaned = _DUAL_MEASURE.sub(r"\1", cleaned)
    cleaned = _COMPOUND_MEASURE.sub(r"\1 \2", cleaned)
    cleaned = _DIGIT_LETTER.sub(r"\1 \2", cleaned)
    tokens = [t for t in re.split(r"\s+", cleaned) if t]
    if not tokens:
        return None

    tokens = _skip_modifiers(tokens)
    quantity, consumed = _leading_quantity(tokens)
    tokens = _skip_modifiers(tokens[consumed:])

    # Pack sizes multiply: "3 400g cans of tomatoes" is 1200g of tomatoes.
    if quantity is not None:
        pack_size, pack_consumed = _leading_quantity(tokens)
        if pack_size is not None and normalize_unit(
            tokens[pack_consumed] if len(tokens) > pack_consumed else ""
        ):
            quantity *= pack_size
            tokens = tokens[pack_consumed:]

    unit, consumed = _leading_unit(tokens)
    tokens = tokens[consumed:]
    if unit is None:
        unit, tokens = _trailing_unit(tokens)
    else:
        # Container words stack on a measure, sometimes more than once:
        # "400g can chickpeas", "2 400g cans can of chickpeas". Keep the
        # measure and peel off every redundant container behind it.
        while True:
            extra, extra_consumed = _leading_unit(tokens)
            if extra is None or len(tokens) <= extra_consumed:
                break
            tokens = tokens[extra_consumed:]

    name = canonicalize(" ".join(tokens))
    if not name:
        return None
    return Ingredient(name=name, quantity=quantity, unit=unit, raw=phrase.strip())


# --- Segmentation ------------------------------------------------------------

_SEPARATORS = re.compile(
    r"[,;\n•]|\band\b|\bplus\b|\balso\b|\bwith\b", flags=re.IGNORECASE
)
_LEAD_IN = re.compile(
    r"^\s*(?:i(?:'ve| have)?\s+(?:got|have)|there(?:'s| is)|we(?:'ve| have)"
    r"|my fridge has|in the fridge|got)\b[:,]?\s*",
    flags=re.IGNORECASE,
)


def split_phrases(text: str) -> list[str]:
    """Rule-based segmentation: the default extractor.

    Strips a conversational lead-in ("I've got ...") then splits on commas,
    newlines, bullets, and connective words.
    """
    without_lead_in = _LEAD_IN.sub("", text.strip())
    parts = _SEPARATORS.split(without_lead_in)
    return [part.strip(" .\t") for part in parts if part and part.strip(" .\t")]


class _SegmentationResponse(BaseModel):
    """Schema the LLM segmentation response must satisfy before we trust it."""

    ingredients: list[str]

    @field_validator("ingredients")
    @classmethod
    def _drop_blanks(cls, values: list[str]) -> list[str]:
        return [v.strip() for v in values if isinstance(v, str) and v.strip()]


SEGMENTATION_PROMPT = """\
Split the user's message into individual ingredient phrases.

Rules:
- Copy each phrase VERBATIM from the message, including any quantity and unit.
- Do not convert, translate, correct, or expand anything.
- Do not invent ingredients that are not mentioned.
- Exclude equipment, cooking steps, and preferences.

Respond with JSON only, in the form:
{"ingredients": ["<phrase>", "<phrase>"]}\
"""


class GroqExtractor:
    """LLM-backed segmentation for genuinely messy prose.

    Only splits the text into phrases -- structuring and normalization stay in
    :func:`parse_phrase`. The response is validated against
    :class:`_SegmentationResponse`, and any phrase the model did not copy from
    the source text is discarded, which is a cheap guard against hallucinated
    ingredients.
    """

    def __init__(self, client=None, model: str | None = None) -> None:
        """Build an extractor.

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

    def __call__(self, text: str) -> list[str]:
        """Segment ``text``, falling back to :func:`split_phrases` on failure."""
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": SEGMENTATION_PROMPT},
                {"role": "user", "content": text},
            ],
            response_format={"type": "json_object"},
            temperature=0,
        )
        content = response.choices[0].message.content
        try:
            parsed = _SegmentationResponse.model_validate_json(content)
        except (ValidationError, json.JSONDecodeError) as exc:
            raise IngredientExtractionError(
                f"segmentation response failed validation: {exc}"
            ) from exc

        haystack = text.lower()
        return [p for p in parsed.ingredients if p.lower() in haystack]


# --- Agent entry point -------------------------------------------------------


def merge_duplicates(items: Iterable[Ingredient]) -> list[Ingredient]:
    """Combine repeats of the same ingredient, preserving first-seen order.

    Quantities are summed only when the unit matches, since "1 cup rice" and
    "200g rice" cannot be added without a density table. Mismatched units are
    kept as separate entries for the safety and composer agents to reconcile.
    """
    merged: dict[tuple[str, str | None], Ingredient] = {}
    for item in items:
        key = (item.name, item.unit)
        existing = merged.get(key)
        if existing is None:
            merged[key] = item
        elif existing.quantity is not None and item.quantity is not None:
            merged[key] = existing.model_copy(
                update={"quantity": existing.quantity + item.quantity}
            )
        elif existing.quantity is None:
            merged[key] = existing.model_copy(update={"quantity": item.quantity})
    return list(merged.values())


def parse_ingredients(
    text: str, *, extractor: Extractor | None = None
) -> list[Ingredient]:
    """Parse raw user text into a list of canonical :class:`Ingredient` objects.

    Args:
        text: Whatever the user typed, however messy.
        extractor: Segmentation strategy. Defaults to the deterministic
            :func:`split_phrases`; pass a :class:`GroqExtractor` for LLM
            segmentation. If the extractor raises, parsing falls back to the
            rule-based split rather than failing the request.

    Returns:
        Parsed ingredients, deduplicated, in the order first mentioned. Returns
        an empty list for empty or unparseable input -- never raises on content.
    """
    if not text or not text.strip():
        return []

    segment = extractor or split_phrases
    try:
        phrases = segment(text)
    except Exception:  # noqa: BLE001 - any extractor failure (network, rate
        # limit, auth, bad schema) must degrade to the rule-based split rather
        # than fail the user's request.
        phrases = split_phrases(text)

    parsed = [item for item in (parse_phrase(p) for p in phrases) if item]
    return merge_duplicates(parsed)
