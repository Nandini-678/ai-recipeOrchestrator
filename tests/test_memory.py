"""Unit tests for the SQLite memory store.

Every test uses an in-memory database, so nothing touches the user's real
store and the suite stays order-independent.
"""

import pytest

from agents.composer import ComposedIngredient, ComposedRecipe, NutritionSummary
from orchestrator.memory import MemoryStore


@pytest.fixture
def store():
    with MemoryStore(":memory:") as memory:
        yield memory


def _recipe(title="Test Dish", recipe_id="42"):
    return ComposedRecipe(
        title=title,
        servings=4,
        ingredients=[
            ComposedIngredient(name="rice", quantity=2, unit="cup", have=True)
        ],
        steps=["Cook the rice."],
        nutrition_per_serving=NutritionSummary(
            calories=200, protein_g=4, fat_g=1, carbs_g=44,
            fiber_g=1, sugar_g=0, sodium_mg=2,
        ),
        source_recipe_id=recipe_id,
    )


class TestPreferences:
    def test_round_trips_a_value(self, store):
        store.set_preference("servings", 6)
        assert store.get_preference("servings") == 6

    def test_types_survive(self, store):
        """JSON encoding means a list comes back a list, not its repr."""
        store.set_preference("avoid", ["dairy", "gluten"])
        assert store.get_preference("avoid") == ["dairy", "gluten"]

    def test_missing_keys_return_the_default(self, store):
        assert store.get_preference("nope", default=4) == 4

    def test_setting_twice_replaces(self, store):
        store.set_preference("servings", 2)
        store.set_preference("servings", 8)
        assert store.get_preference("servings") == 8
        assert len(store.all_preferences()) == 1

    def test_all_preferences(self, store):
        store.set_preference("a", 1)
        store.set_preference("b", "two")
        assert store.all_preferences() == {"a": 1, "b": "two"}

    def test_empty_store(self, store):
        assert store.all_preferences() == {}


class TestHistory:
    def test_records_a_run(self, store):
        row_id = store.record_run(
            "rice and eggs", avoid=["dairy"], servings=2, accepted=True,
            recipe=_recipe(),
        )
        assert row_id > 0
        entry = store.recent_runs()[0]
        assert entry.pantry_text == "rice and eggs"
        assert entry.avoid == ("dairy",)
        assert entry.servings == 2
        assert entry.accepted is True

    def test_the_recipe_round_trips(self, store):
        store.record_run("rice", recipe=_recipe("Fried Rice"), accepted=True)
        entry = store.recent_runs()[0]
        assert entry.recipe.title == "Fried Rice"
        assert entry.recipe.ingredients[0].name == "rice"

    def test_a_run_with_no_recipe_is_still_recorded(self, store):
        store.record_run("qwerty", accepted=False)
        entry = store.recent_runs()[0]
        assert entry.recipe is None and entry.recipe_title is None

    def test_newest_first(self, store):
        for title in ("first", "second", "third"):
            store.record_run(title)
        assert [e.pantry_text for e in store.recent_runs()] == [
            "third", "second", "first",
        ]

    def test_limit_is_respected(self, store):
        for index in range(10):
            store.record_run(f"run {index}")
        assert len(store.recent_runs(limit=3)) == 3

    def test_clear_history(self, store):
        store.record_run("rice")
        store.clear_history()
        assert store.recent_runs() == []

    def test_a_recipe_that_no_longer_parses_does_not_lose_the_row(self, store):
        """An older schema is history, not an error."""
        store.record_run("rice", recipe=_recipe("Old"), accepted=True)
        store._connection.execute(
            "UPDATE history SET recipe_json = ?", ('{"bogus": true}',)
        )
        store._connection.commit()
        entry = store.recent_runs()[0]
        assert entry.recipe is None
        assert entry.recipe_title == "Old"


class TestFavourites:
    def test_save_and_list(self, store):
        store.save_favourite(_recipe("Fried Rice"))
        assert [r.title for r in store.list_favourites()] == ["Fried Rice"]

    def test_saving_the_same_recipe_twice_does_not_duplicate(self, store):
        store.save_favourite(_recipe("Fried Rice", recipe_id="7"))
        store.save_favourite(_recipe("Fried Rice (edited)", recipe_id="7"))
        favourites = store.list_favourites()
        assert len(favourites) == 1
        assert favourites[0].title == "Fried Rice (edited)"

    def test_is_favourite(self, store):
        recipe = _recipe(recipe_id="9")
        assert store.is_favourite(recipe) is False
        store.save_favourite(recipe)
        assert store.is_favourite(recipe) is True

    def test_remove(self, store):
        store.save_favourite(_recipe(recipe_id="9"))
        store.remove_favourite("9")
        assert store.list_favourites() == []

    def test_recipes_without_a_source_id_key_on_title(self, store):
        recipe = _recipe("Improvised", recipe_id="")
        store.save_favourite(recipe)
        assert store.is_favourite(recipe) is True


class TestPersistence:
    def test_data_survives_reopening_the_file(self, tmp_path):
        path = tmp_path / "nested" / "memory.sqlite3"
        with MemoryStore(path) as store:
            store.set_preference("servings", 6)
            store.record_run("rice", recipe=_recipe(), accepted=True)
            store.save_favourite(_recipe())

        with MemoryStore(path) as reopened:
            assert reopened.get_preference("servings") == 6
            assert len(reopened.recent_runs()) == 1
            assert len(reopened.list_favourites()) == 1

    def test_the_parent_directory_is_created(self, tmp_path):
        path = tmp_path / "a" / "b" / "memory.sqlite3"
        MemoryStore(path).close()
        assert path.exists()

    def test_opening_an_existing_store_is_not_destructive(self, tmp_path):
        path = tmp_path / "memory.sqlite3"
        MemoryStore(path).close()
        with MemoryStore(path) as store:
            store.set_preference("k", 1)
        with MemoryStore(path) as store:
            assert store.get_preference("k") == 1
