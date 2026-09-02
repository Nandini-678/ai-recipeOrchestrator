"""Deterministic text normalization shared across agents.

Ingredient names arrive misspelled, pluralized, and full of regional slang
("aubergine", "scallions", "corriander"). Every agent that has to *match* an
ingredient -- retrieval against the recipe index, safety against the allergen
table -- needs the same canonical form, so that logic lives here rather than
inside any one agent.

Nothing in this module calls a language model. It is pure and deterministic,
which is what lets the ingredient agent be unit tested without an API key.
"""

from __future__ import annotations

import re
import unicodedata
from difflib import get_close_matches

# --- Units -------------------------------------------------------------------

#: Maps every spelling we accept onto a canonical unit token.
UNIT_ALIASES: dict[str, str] = {
    # volume
    "tsp": "tsp", "tsps": "tsp", "teaspoon": "tsp", "teaspoons": "tsp",
    "tbsp": "tbsp", "tbsps": "tbsp", "tbs": "tbsp", "tablespoon": "tbsp",
    "tblsp": "tbsp", "tblsps": "tbsp", "tspn": "tsp", "tsps.": "tsp",
    "tablespoons": "tbsp",
    "cup": "cup", "cups": "cup", "c": "cup",
    "ml": "ml", "milliliter": "ml", "milliliters": "ml", "millilitre": "ml",
    "millilitres": "ml",
    "l": "l", "liter": "l", "liters": "l", "litre": "l", "litres": "l",
    "floz": "fl oz", "fl oz": "fl oz", "fluid ounce": "fl oz",
    "fluid ounces": "fl oz",
    "pint": "pint", "pints": "pint", "pt": "pint", "pts": "pint",
    "quart": "quart", "quarts": "quart", "qt": "qt", "qts": "qt",
    "gallon": "gallon", "gallons": "gallon", "gal": "gallon",
    "cl": "cl", "dl": "dl",
    # weight
    "g": "g", "gram": "g", "grams": "g", "gramme": "g", "grammes": "g",
    "kg": "kg", "kilogram": "kg", "kilograms": "kg", "kilo": "kg",
    "kilos": "kg",
    "mg": "mg", "milligram": "mg", "milligrams": "mg",
    "oz": "oz", "ounce": "oz", "ounces": "oz",
    "lb": "lb", "lbs": "lb", "pound": "lb", "pounds": "lb",
    # countable / informal
    "clove": "clove", "cloves": "clove",
    "slice": "slice", "slices": "slice",
    "piece": "piece", "pieces": "piece",
    "pinch": "pinch", "pinches": "pinch",
    "dash": "dash", "dashes": "dash",
    "can": "can", "cans": "can", "tin": "can", "tins": "can",
    "jar": "jar", "jars": "jar",
    "packet": "packet", "packets": "packet", "pack": "packet",
    "bunch": "bunch", "bunches": "bunch",
    "head": "head", "heads": "head",
    "stalk": "stalk", "stalks": "stalk",
    "sprig": "sprig", "sprigs": "sprig",
    "handful": "handful", "handfuls": "handful",
    "knob": "knob", "knobs": "knob", "splash": "splash", "drizzle": "drizzle",
    "stick": "stick", "sticks": "stick",
    "fillet": "fillet", "fillets": "fillet",
    "package": "packet", "pkg": "packet", "box": "box", "boxes": "box",
    "bottle": "bottle", "bottles": "bottle", "bag": "bag", "bags": "bag",
    # size descriptors that appear where a unit does ("2cm piece ginger")
    "cm": "cm", "mm": "mm", "inch": "inch", "inches": "inch",
}

# --- Names -------------------------------------------------------------------

#: Regional names, slang, and brand-ish terms mapped to one canonical name.
INGREDIENT_ALIASES: dict[str, str] = {
    "aubergine": "eggplant", "brinjal": "eggplant",
    "courgette": "zucchini",
    "capsicum": "bell pepper", "sweet pepper": "bell pepper",
    "scallion": "green onion", "spring onion": "green onion",
    "rocket": "arugula",
    "coriander": "cilantro", "dhania": "cilantro",
    # NB: "coriander seed" is the spice, not the herb, and is left alone by
    # the KNOWN_INGREDIENTS short-circuit in normalize_name.
    "prawn": "shrimp",
    "mince": "ground beef",
    "garbanzo": "chickpea", "garbanzo bean": "chickpea",
    "passata": "tomato puree",
    "double cream": "heavy cream", "thickened cream": "heavy cream",
    "caster sugar": "sugar", "castor sugar": "sugar",
    "icing sugar": "powdered sugar",
    "confectioner sugar": "powdered sugar",
    "maida": "all-purpose flour", "plain flour": "all-purpose flour",
    "curd": "yogurt", "dahi": "yogurt",
    "ladyfinger": "okra", "bhindi": "okra",
    "spud": "potato", "tater": "potato",
    "soda water": "sparkling water",
    "beetroot": "beet",
    "chickpea flour": "besan",
    "corn flour": "cornstarch", "cornflour": "cornstarch",
    "bicarbonate of soda": "baking soda", "bicarb": "baking soda",
    # hyphens are split before alias lookup, so the spaced form must map too
    "all purpose flour": "all-purpose flour",
}

#: Preparation words that describe *handling*, not identity, so they are safe to
#: drop when matching. Words that change what the ingredient *is* -- "ground",
#: "whole", "green", "smoked" -- are deliberately absent.
PREP_MODIFIERS: frozenset[str] = frozenset({
    "chopped", "diced", "minced", "sliced", "grated", "shredded", "crushed",
    "fresh", "freshly", "frozen", "canned", "tinned", "dried", "peeled",
    "cubed", "halved", "quartered", "large", "small", "medium", "ripe",
    "raw", "cooked", "boneless", "skinless", "finely", "roughly", "thinly",
    "optional", "packed", "softened", "melted", "beaten", "washed", "rinsed",
    "trimmed", "stemmed", "seeded", "pitted", "torn", "cut", "leftover",
    "into", "cube", "chunk", "strip", "wedge", "batons", "baton", "approx",
    "thick", "thin", "long", "square", "round", "whole", "slice", "sliver",
    # "salt to taste" is a quantity hedge, not part of the ingredient's name
    "to", "for", "taste", "needed", "serving", "garnish", "dusting",
    "drizzling", "extra", "virgin", "unsalted", "salted", "unsweetened",
    "sweetened", "free", "range", "organic",
    # TheMealDB puts the *purpose* in the measure field ("For brushing",
    # "For frying"), which otherwise prefixes it onto the ingredient name and
    # splits "olive oil" into three different ingredients.
    "brushing", "frying", "greasing", "glaze", "glazing", "topping",
    "sprinkling", "serve", "coating", "dredging", "rolling", "deep",
    # size words are units positionally; if one reaches the *name* it is noise
    "inch", "cm", "mm", "piece",
})

#: Words that merely look plural. Stripping the trailing "s" would corrupt them.
NEVER_SINGULARIZE: frozenset[str] = frozenset({
    "asparagus", "couscous", "hummus", "molasses", "watercress", "cress",
    "lemongrass", "brussels", "swiss", "bass", "gas", "grits", "oats",
    "greens", "chives", "capers", "sprouts",
})

#: Plurals that do not follow the regular rules.
IRREGULAR_PLURALS: dict[str, str] = {
    "leaves": "leaf", "loaves": "loaf", "knives": "knife", "halves": "half",
    "shelves": "shelf", "wolves": "wolf", "geese": "goose", "mice": "mouse",
    "feet": "foot", "teeth": "tooth", "children": "child", "people": "person",
    "roes": "roe", "anchovies": "anchovy",
}

#: Canonical vocabulary. Doubles as the target list for typo correction, so it
#: is worth keeping broad but strictly singular and canonical.
KNOWN_INGREDIENTS: frozenset[str] = frozenset({
    # produce
    "onion", "green onion", "garlic", "ginger", "tomato", "potato", "carrot",
    "celery", "bell pepper", "chili pepper", "jalapeno", "cucumber", "zucchini",
    "eggplant", "mushroom", "spinach", "kale", "lettuce", "arugula", "cabbage",
    "broccoli", "cauliflower", "pea", "green bean", "corn", "beet", "radish",
    "pumpkin", "squash", "sweet potato", "okra", "leek", "shallot", "avocado",
    "lemon", "lime", "orange", "apple", "banana", "strawberry", "blueberry",
    "raspberry", "mango", "pineapple", "grape", "peach", "pear", "cherry",
    "coconut", "date", "raisin", "olive",
    # herbs and spices
    "basil", "cilantro", "parsley", "mint", "rosemary", "thyme", "oregano",
    "sage", "dill", "bay leaf", "cumin", "turmeric", "paprika", "cinnamon",
    "nutmeg", "clove", "cardamom", "chili powder", "curry powder",
    "black pepper",
    "salt", "cayenne", "coriander seed", "mustard seed", "saffron", "vanilla",
    # protein
    "chicken", "chicken breast", "chicken thigh", "beef", "ground beef",
    "steak", "pork", "bacon", "ham", "sausage", "lamb", "turkey", "duck",
    "fish", "salmon", "tuna", "cod", "shrimp", "crab", "lobster", "mussel",
    "clam", "anchovy", "egg", "tofu", "tempeh", "seitan",
    # dairy
    "milk", "butter", "cheese", "cheddar cheese", "parmesan cheese",
    "mozzarella cheese", "feta cheese", "cream cheese", "heavy cream",
    "sour cream", "yogurt", "ghee",
    # pantry
    "rice", "pasta", "spaghetti", "noodle", "bread", "flour",
    "all-purpose flour", "besan", "cornstarch", "sugar", "powdered sugar",
    "brown sugar", "honey", "maple syrup", "molasses", "baking soda",
    "baking powder", "yeast", "oat", "quinoa", "couscous", "lentil",
    "chickpea", "black bean", "kidney bean", "white bean", "peanut",
    "almond", "cashew", "walnut", "pecan", "pistachio", "sesame seed",
    "sunflower seed", "chia seed", "olive oil", "vegetable oil", "sesame oil",
    "coconut oil", "vinegar", "balsamic vinegar", "soy sauce", "fish sauce",
    "hot sauce", "ketchup", "mustard", "mayonnaise", "tomato paste",
    "tomato puree", "coconut milk", "broth", "bouillon", "stock",
    "peanut butter", "chocolate", "cocoa powder", "wine", "beer",
})

_PUNCTUATION = re.compile(r"[^\w\s/.-]+")
_WHITESPACE = re.compile(r"\s+")
_PARENTHETICAL = re.compile(r"\([^)]*\)")

#: A token carrying no name information: bare numbers, fractions, and the
#: leftovers of compound measures like "1/2-inch".
_NUMERIC_TOKEN = re.compile(r"^[\d.,/\-\u00bc-\u00be\u2150-\u215e]+$")

#: "juice of 2 lemons" -> "lemon juice"; same for zest, rind, and peel.
_EXTRACT_OF = re.compile(r"^(juice|zest|rind|peel)\s+of\s+(.+)$")


def strip_accents(text: str) -> str:
    """Fold accented characters to ASCII so "jalapeño" matches "jalapeno"."""
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def singularize(word: str) -> str:
    """Return the singular form of a single lowercase ``word``.

    Handles the regular English patterns plus a small irregular table. Words in
    :data:`NEVER_SINGULARIZE` are returned untouched.
    """
    if word in NEVER_SINGULARIZE or len(word) <= 3:
        return word
    if word in IRREGULAR_PLURALS:
        return IRREGULAR_PLURALS[word]
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith("oes"):
        return word[:-2]
    if re.search(r"(ch|sh|ss|x|z)es$", word):
        return word[:-2]
    if word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def normalize_unit(raw: str | None) -> str | None:
    """Map a unit spelling onto its canonical token, or ``None`` if unknown."""
    if not raw:
        return None
    key = _WHITESPACE.sub(" ", strip_accents(raw).lower().strip(" ."))
    return UNIT_ALIASES.get(key)


def normalize_name(raw: str) -> str:
    """Reduce an ingredient phrase to a canonical, matchable name.

    Lowercases, strips accents, drops parentheticals and preparation words,
    singularizes each remaining word, then applies the alias table.

    >>> normalize_name("2 finely chopped Scallions (green parts)")
    'green onion'
    """
    text = strip_accents(raw).lower()
    text = _PARENTHETICAL.sub(" ", text)
    # Split hyphenated compounds ("1.5cm-thick", "2-inch") so each part can be
    # judged on its own. Genuinely hyphenated names are restored by the alias
    # table, which is consulted after this.
    text = text.replace("-", " ")
    text = _PUNCTUATION.sub(" ", text)
    text = _WHITESPACE.sub(" ", text).strip()

    words = [w for w in text.split() if w and not _NUMERIC_TOKEN.match(w)]
    words = [w for w in words if w not in PREP_MODIFIERS]
    words = [singularize(w) for w in words]
    # Filter again: the set holds singular forms, so plural prep words
    # ("cubes", "strips") only become matchable after singularization.
    words = [w for w in words if w not in PREP_MODIFIERS]

    name = " ".join(words).strip()
    if not name:
        return ""

    # "juice of lemon" is the same ingredient as "lemon juice".
    extracted = _EXTRACT_OF.match(name)
    if extracted:
        name = f"{extracted.group(2).strip()} {extracted.group(1)}".strip()

    # Aliases are checked on the whole phrase first, then word by word, so both
    # "spring onion" and "scallions" land on "green onion".
    if name in INGREDIENT_ALIASES:
        return INGREDIENT_ALIASES[name]
    # A name already in the canonical vocabulary is never rewritten. Without
    # this, the word-wise pass below turns "coriander seed" (the spice) into
    # "cilantro seed" (the herb) via the coriander -> cilantro alias.
    if name in KNOWN_INGREDIENTS:
        return name
    return " ".join(INGREDIENT_ALIASES.get(w, w) for w in name.split())


#: Individual words drawn from :data:`KNOWN_INGREDIENTS`, used to fix a typo
#: inside a multi-word phrase ("brocolli florets") that no whole-phrase match
#: would ever catch.
VOCABULARY_WORDS: frozenset[str] = frozenset(
    word for entry in KNOWN_INGREDIENTS for word in entry.split()
)

#: Below this length a fuzzy match is more likely to corrupt a good word than
#: to rescue a bad one ("bag" -> "bay"), so short words are left alone.
_MIN_CORRECTABLE_LENGTH = 5


def correct_spelling(name: str, cutoff: float = 0.82) -> str:
    """Snap a near-miss name onto the known vocabulary.

    Tries the whole phrase first, then falls back to correcting each word
    against :data:`VOCABULARY_WORDS`. Only matches above ``cutoff`` similarity
    are accepted and only words already absent from the vocabulary are touched,
    so unfamiliar-but-real ingredients survive rather than being silently
    rewritten into something else.

    >>> correct_spelling("brocolli")
    'broccoli'
    >>> correct_spelling("brocolli floret")
    'broccoli floret'
    >>> correct_spelling("gochujang")
    'gochujang'
    """
    if not name or name in KNOWN_INGREDIENTS:
        return name

    whole = get_close_matches(name, KNOWN_INGREDIENTS, n=1, cutoff=cutoff)
    if whole:
        return whole[0]

    corrected = [_correct_word(word, cutoff) for word in name.split()]
    return " ".join(corrected)


def _correct_word(word: str, cutoff: float) -> str:
    """Correct a single word against :data:`VOCABULARY_WORDS`, or return it."""
    if word in VOCABULARY_WORDS or len(word) < _MIN_CORRECTABLE_LENGTH:
        return word
    match = get_close_matches(word, VOCABULARY_WORDS, n=1, cutoff=cutoff)
    return match[0] if match else word


def canonicalize(raw: str) -> str:
    """Full name pipeline: :func:`normalize_name` then :func:`correct_spelling`."""
    return correct_spelling(normalize_name(raw))


#: The longest ingredient name, in tokens, that step text is scanned for.
_MAX_NAME_TOKENS = 3
_WORD_BOUNDARY = re.compile(r"[^a-z]+")

def mentioned_ingredients(text: str) -> set[str]:
    """Find canonical ingredient names mentioned anywhere in ``text``.

    Scans longest-first over 1-3 token windows so "olive oil" is recognised as
    itself rather than as two unrelated words. Only names in the project's
    canonical vocabulary are recognised, which keeps the check conservative:
    an ingredient we do not know about cannot be reported as invented.

    >>> sorted(mentioned_ingredients("Fry the onions, then add the chicken."))
    ['chicken', 'onion']
    """
    tokens = [singularize(t) for t in _WORD_BOUNDARY.split(text.lower()) if t]
    found: set[str] = set()
    claimed: set[int] = set()
    # Longest first, and a matched window consumes its positions, so "olive
    # oil" is one ingredient rather than also counting as "olive".
    for width in range(_MAX_NAME_TOKENS, 0, -1):
        for start in range(len(tokens) - width + 1):
            span = range(start, start + width)
            if any(position in claimed for position in span):
                continue
            phrase = " ".join(tokens[start : start + width])
            canonical = INGREDIENT_ALIASES.get(phrase, phrase)
            if canonical in KNOWN_INGREDIENTS:
                found.add(canonical)
                claimed.update(span)
    return found


