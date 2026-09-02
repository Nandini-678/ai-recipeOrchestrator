"""Unit tests for the shared normalization layer.

Pure functions, no network, no API key.
"""

import pytest

from agents.normalization import (
    canonicalize,
    correct_spelling,
    normalize_name,
    normalize_unit,
    singularize,
)


class TestSingularize:
    @pytest.mark.parametrize(
        ("plural", "expected"),
        [
            ("tomatoes", "tomato"),
            ("potatoes", "potato"),
            ("eggs", "egg"),
            ("berries", "berry"),
            ("leaves", "leaf"),
            ("knives", "knife"),
            ("peaches", "peach"),
            ("dishes", "dish"),
            ("boxes", "box"),
            ("anchovies", "anchovy"),
        ],
    )
    def test_regular_and_irregular_plurals(self, plural, expected):
        assert singularize(plural) == expected

    @pytest.mark.parametrize(
        "word", ["asparagus", "couscous", "hummus", "molasses", "lemongrass"]
    )
    def test_words_that_only_look_plural_are_untouched(self, word):
        assert singularize(word) == word

    def test_already_singular_is_unchanged(self):
        assert singularize("onion") == "onion"


class TestNormalizeUnit:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("tablespoons", "tbsp"),
            ("Tbsp.", "tbsp"),
            ("teaspoon", "tsp"),
            ("CUPS", "cup"),
            ("grams", "g"),
            ("kilos", "kg"),
            ("pounds", "lb"),
            ("cloves", "clove"),
            ("fl oz", "fl oz"),
            ("tins", "can"),
        ],
    )
    def test_aliases_collapse_to_canonical(self, raw, expected):
        assert normalize_unit(raw) == expected

    @pytest.mark.parametrize("raw", [None, "", "chicken", "handfulish"])
    def test_non_units_return_none(self, raw):
        assert normalize_unit(raw) is None


class TestNormalizeName:
    def test_strips_prep_words_and_singularizes(self):
        assert normalize_name("finely chopped Onions") == "onion"

    def test_keeps_identity_changing_modifiers(self):
        """'ground' and 'green' change what the ingredient is, so they stay."""
        assert normalize_name("ground beef") == "ground beef"
        assert normalize_name("green onions") == "green onion"

    def test_drops_parentheticals(self):
        assert normalize_name("cilantro (leaves only)") == "cilantro"

    def test_folds_accents(self):
        assert normalize_name("jalapeños") == "jalapeno"

    @pytest.mark.parametrize(
        ("slang", "expected"),
        [
            ("aubergine", "eggplant"),
            ("courgettes", "zucchini"),
            ("scallions", "green onion"),
            ("spring onions", "green onion"),
            ("capsicum", "bell pepper"),
            ("prawns", "shrimp"),
            ("maida", "all-purpose flour"),
            ("plain flour", "all-purpose flour"),
            ("bhindi", "okra"),
            ("spuds", "potato"),
            ("bicarb", "baking soda"),
        ],
    )
    def test_regional_slang_maps_to_canonical(self, slang, expected):
        assert normalize_name(slang) == expected

    def test_empty_input_yields_empty_string(self):
        assert normalize_name("   ") == ""


class TestCorrectSpelling:
    @pytest.mark.parametrize(
        ("typo", "expected"),
        [
            ("brocolli", "broccoli"),
            ("chikcen", "chicken"),
            ("tomatoe", "tomato"),
            ("garlick", "garlic"),
            ("cinnammon", "cinnamon"),
        ],
    )
    def test_near_misses_snap_to_vocabulary(self, typo, expected):
        assert correct_spelling(typo) == expected

    def test_known_ingredients_are_never_rewritten(self):
        assert correct_spelling("chicken") == "chicken"

    def test_unknown_but_plausible_ingredients_survive(self):
        """A real ingredient we simply don't list must not be mangled."""
        assert correct_spelling("dragonfruit") == "dragonfruit"
        assert correct_spelling("gochujang") == "gochujang"


class TestCanonicalize:
    def test_combines_normalization_and_spelling_correction(self):
        assert canonicalize("2 finely chopped Brocolli florets") == "broccoli floret"
        assert canonicalize("Scallions") == "green onion"
