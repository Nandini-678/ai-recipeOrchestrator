"""Unit tests for the recipe corpus schema and TheMealDB normalization."""

import json

import pytest

from agents.recipe import (
    Recipe,
    find_unlisted_in_steps,
    from_mealdb,
    load_recipes,
    save_recipes,
    split_steps,
)


class TestSplitSteps:
    def test_prefers_the_sources_own_line_breaks(self):
        assert split_steps("Chop it.\r\nCook it.\r\nServe it.") == [
            "Chop it.",
            "Cook it.",
            "Serve it.",
        ]

    def test_falls_back_to_sentences_for_one_paragraph(self):
        steps = split_steps("Chop the onion. Fry it gently. Serve hot.")
        assert steps == ["Chop the onion.", "Fry it gently.", "Serve hot."]

    def test_strips_existing_step_numbering(self):
        assert split_steps("1. Chop it.\n2) Cook it.\nStep 3. Serve.") == [
            "Chop it.",
            "Cook it.",
            "Serve.",
        ]

    def test_lines_that_are_only_a_step_number_are_dropped(self):
        """Several TheMealDB entries put the number on its own line."""
        blob = "1\r\nFry the garlic.\r\n2\r\nAdd oil.\r\n3\r\n"
        assert split_steps(blob) == ["Fry the garlic.", "Add oil."]

    def test_a_step_that_merely_starts_with_a_number_is_kept(self):
        assert split_steps("350 degrees is the target.") == [
            "350 degrees is the target."
        ]

    @pytest.mark.parametrize("blob", ["", "   ", "\n\n"])
    def test_empty_instructions_give_no_steps(self, blob):
        assert split_steps(blob) == []


class TestFromMealDB:
    @pytest.fixture
    def payload(self):
        return {
            "idMeal": "52772",
            "strMeal": "Teriyaki Chicken Casserole",
            "strCategory": "Chicken",
            "strArea": "Japanese",
            "strInstructions": "Preheat oven.\r\nCombine and bake.",
            "strMealThumb": "https://example.test/img.jpg",
            "strTags": "Meat,Casserole",
            "strSource": "https://example.test/recipe",
            "strIngredient1": "soy sauce",
            "strMeasure1": "3/4 cup",
            "strIngredient2": "Plain Flour",
            "strMeasure2": "175g/6oz",
            "strIngredient3": "Spring Onions",
            "strMeasure3": "Small bunch",
            "strIngredient4": "",
            "strMeasure4": "",
        }

    def test_maps_core_fields(self, payload):
        recipe = from_mealdb(payload)
        assert recipe.id == "52772"
        assert recipe.title == "Teriyaki Chicken Casserole"
        assert recipe.category == "Chicken"
        assert recipe.tags == ["Meat", "Casserole"]
        assert recipe.steps == ["Preheat oven.", "Combine and bake."]

    def test_ingredients_go_through_the_same_pipeline_as_user_input(self, payload):
        """Corpus and pantry must canonicalize identically or overlap fails."""
        recipe = from_mealdb(payload)
        assert recipe.ingredient_names == {
            "soy sauce",
            "all-purpose flour",
            "green onion",
        }

    def test_measures_are_parsed_not_just_stored(self, payload):
        by_name = {i.name: i for i in from_mealdb(payload).ingredients}
        assert (by_name["soy sauce"].quantity, by_name["soy sauce"].unit) == (
            0.75,
            "cup",
        )
        # dual metric/imperial measure keeps the metric side
        assert (by_name["all-purpose flour"].quantity,
                by_name["all-purpose flour"].unit) == (175.0, "g")

    def test_blank_ingredient_slots_are_skipped(self, payload):
        assert len(from_mealdb(payload).ingredients) == 3

    def test_missing_optional_fields_become_none(self, payload):
        payload["strArea"] = ""
        payload["strSource"] = None
        recipe = from_mealdb(payload)
        assert recipe.area is None and recipe.source_url is None


class TestSearchableText:
    def test_includes_title_category_and_ingredients(self):
        recipe = Recipe(
            id="1",
            title="Fried Rice",
            category="Chinese",
            ingredients=[{"name": "rice"}, {"name": "egg"}],
        )
        text = recipe.searchable_text()
        assert "Fried Rice" in text and "Chinese" in text
        assert "rice" in text and "egg" in text


class TestRoundTrip:
    def test_save_then_load_preserves_recipes(self, tmp_path, sample_recipes):
        path = tmp_path / "nested" / "corpus.json"
        save_recipes(sample_recipes, path)
        assert load_recipes(path) == sample_recipes

    def test_load_points_at_the_build_script_when_missing(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="scripts.build_corpus"):
            load_recipes(tmp_path / "absent.json")

    def test_saved_file_is_valid_readable_json(self, tmp_path, sample_recipes):
        path = tmp_path / "corpus.json"
        save_recipes(sample_recipes, path)
        assert len(json.loads(path.read_text())) == len(sample_recipes)


class TestUnlistedInSteps:
    """A trust signal: nearly half of TheMealDB's recipes have one."""

    def _ingredients(self, names):
        from agents.ingredient import Ingredient

        return [Ingredient(name=n) for n in names]

    def test_finds_an_ingredient_the_list_omits(self):
        found = find_unlisted_in_steps(
            self._ingredients(["garlic"]), ["Tip the bread into a bowl."]
        )
        assert found == ["bread"]

    def test_listed_ingredients_are_not_reported(self):
        found = find_unlisted_in_steps(
            self._ingredients(["bread", "garlic"]), ["Tip the bread into a bowl."]
        )
        assert found == []

    def test_kitchen_staples_are_assumed(self):
        """Every recipe uses water and salt without listing them."""
        found = find_unlisted_in_steps(
            self._ingredients(["rice"]), ["Boil in salted water."]
        )
        assert found == []

    def test_a_token_of_a_listed_ingredient_is_allowed(self):
        found = find_unlisted_in_steps(
            self._ingredients(["chicken breast"]), ["Sear the chicken."]
        )
        assert found == []

    def test_it_is_recorded_not_repaired(self):
        """Recorded, never added to the ingredient list.

        Half are alternatives or optional garnishes, so adding them would make
        the shopping list wrong rather than right.
        """
        recipe = from_mealdb({
            "idMeal": "1", "strMeal": "T", "strInstructions": "Serve with bread.",
            "strIngredient1": "garlic", "strMeasure1": "1 clove",
        })
        assert recipe.unlisted_in_steps == ["bread"]
        assert "bread" not in recipe.ingredient_names
