"""Unit tests for the RecipeNLG curation script.

The curation rules are the reason this source improves the product, so they
are tested directly: a corpus of twenty-ingredient project recipes would leave
the pipeline no better off than the one it replaced.
"""

import pytest

from scripts.fetch_recipenlg import (
    MAX_INGREDIENTS,
    MAX_TITLE_LENGTH,
    MIN_INGREDIENTS,
    parse_record,
)


def _blob(
    title="Test Dish",
    ingredients=("1 c. rice", "2 eggs"),
    steps=("Cook.", "Serve."),
):
    return (
        f"{title}\n\nIngredients:\n"
        + "\n".join(f"- {i}" for i in ingredients)
        + "\n\nDirections:\n"
        + "\n".join(f"- {s}" for s in steps)
    )


class TestParseRecord:
    def test_parses_a_well_formed_record(self):
        recipe = parse_record(
            _blob(ingredients=("1 c. rice", "2 eggs", "1 tbsp. butter")), 7
        )
        assert recipe.title == "Test Dish"
        assert recipe.id == "nlg-7"
        assert recipe.source == "recipenlg"
        assert [i.name for i in recipe.ingredients] == ["rice", "egg", "butter"]
        assert recipe.steps == ["Cook.", "Serve."]

    def test_measures_are_parsed_not_just_stored(self):
        recipe = parse_record(
            _blob(ingredients=("1 c. rice", "2 eggs", "1 lb. beef")), 1
        )
        rice = recipe.ingredients[0]
        assert (rice.quantity, rice.unit) == (1.0, "cup")

    def test_a_parenthesised_pack_size_does_not_hide_the_container(self):
        """"1 (10 oz.) can soup" is one can, not one of something unnamed."""
        recipe = parse_record(
            _blob(ingredients=("1 (10 1/2 oz.) can cream of mushroom soup",
                               "2 eggs", "1 c. rice")), 1
        )
        soup = recipe.ingredients[0]
        assert soup.name == "cream of mushroom soup"
        assert (soup.quantity, soup.unit) == (1.0, "can")

    def test_an_alternative_resolves_to_the_first_option(self):
        recipe = parse_record(
            _blob(ingredients=("1 c. butter or margarine", "2 eggs", "1 c. rice")), 1
        )
        assert recipe.ingredients[0].name == "butter"

    @pytest.mark.parametrize(
        "text",
        [
            "",
            "no structure at all",
            "Title\n\nIngredients:\n- rice",
            "Title\n\nDirections:\n- Cook.",
        ],
    )
    def test_malformed_records_are_rejected(self, text):
        assert parse_record(text, 1) is None


class TestCurationBounds:
    """The bounds are the point: this source exists to supply simple recipes."""

    def test_too_few_ingredients_is_rejected(self):
        assert parse_record(_blob(ingredients=("1 c. rice",)), 1) is None

    def test_too_many_ingredients_is_rejected(self):
        many = tuple(f"1 c. thing{n}" for n in range(MAX_INGREDIENTS + 1))
        assert parse_record(_blob(ingredients=many), 1) is None

    def test_the_bounds_admit_a_normal_recipe(self):
        ok = tuple(f"1 c. thing{n}" for n in range(MIN_INGREDIENTS, MAX_INGREDIENTS))
        assert parse_record(_blob(ingredients=ok), 1) is not None

    def test_a_single_step_is_rejected(self):
        assert parse_record(_blob(steps=("Do everything.",)), 1) is None

    def test_an_overlong_title_is_rejected(self):
        assert parse_record(_blob(title="x" * (MAX_TITLE_LENGTH + 1)), 1) is None

    def test_a_recipe_leaning_on_unlisted_ingredients_is_rejected(self):
        """The promise is an accurate shopping list; these break it."""
        blob = _blob(
            ingredients=("1 c. rice", "2 eggs", "1 tbsp. butter"),
            steps=("Serve over bread with cheese and shrimp.", "Enjoy."),
        )
        assert parse_record(blob, 1) is None
