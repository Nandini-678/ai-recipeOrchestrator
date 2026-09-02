"""Smoke tests for the Streamlit app.

These render the app through Streamlit's own AppTest harness and assert it
comes up clean. They deliberately stop short of running a search, which would
make live API calls -- the pipeline behind that button is covered by
test_orchestrator.py.

Each test points ``RECIPE_DB_PATH`` at a temporary file so the user's real
store is never touched.
"""

import pytest

from agents.safety import ALLERGEN_LABELS
from config import PROCESSED_DATA_DIR, PROJECT_ROOT

pytest.importorskip("streamlit.testing.v1")
from streamlit.testing.v1 import AppTest  # noqa: E402

#: Absolute: AppTest resolves relative paths against the *calling* file.
APP = str(PROJECT_ROOT / "ui" / "app.py")


@pytest.fixture
def app(tmp_path, monkeypatch):
    """An app instance backed by a throwaway database."""
    if not (PROCESSED_DATA_DIR / "recipes.json").exists():
        pytest.skip("recipe corpus not built")
    monkeypatch.setenv("RECIPE_DB_PATH", str(tmp_path / "memory.sqlite3"))
    instance = AppTest.from_file(APP, default_timeout=120)
    instance.run()
    return instance


class TestRendering:
    def test_the_app_renders_without_error(self, app):
        assert app.exception == []
        assert app.title[0].value == "What can I cook?"

    def test_the_search_controls_are_present(self, app):
        assert [t.label for t in app.text_area] == ["Your ingredients"]
        assert any(b.label == "Find a recipe" for b in app.button)

    def test_the_sidebar_offers_preferences(self, app):
        assert [m.label for m in app.multiselect] == ["Avoid"]
        assert [n.label for n in app.number_input] == ["Servings"]


class TestAllergenLabels:
    def test_users_see_labels_not_internal_category_names(self, app):
        """Nobody says they avoid "wheat" when they mean gluten."""
        shown = app.multiselect[0].options
        assert "Gluten" in shown and "Dairy" in shown
        assert "wheat" not in shown and "milk" not in shown

    def test_every_category_is_offered(self, app):
        assert set(app.multiselect[0].options) == set(ALLERGEN_LABELS.values())

    def test_selecting_a_label_stores_the_category(self, app):
        app.multiselect[0].select("Dairy").run()
        assert app.multiselect[0].value == ["milk"]


class TestPreferencePersistence:
    def test_preferences_survive_a_restart(self, tmp_path, monkeypatch):
        """A widget keyed off a mutable default silently loses its state."""
        monkeypatch.setenv("RECIPE_DB_PATH", str(tmp_path / "memory.sqlite3"))

        first = AppTest.from_file(APP, default_timeout=120)
        first.run()
        first.multiselect[0].select("Dairy").run()
        first.number_input[0].set_value(6).run()

        second = AppTest.from_file(APP, default_timeout=120)
        second.run()
        assert second.multiselect[0].value == ["milk"]
        assert second.number_input[0].value == 6


class TestNavigation:
    @pytest.mark.parametrize(
        ("page", "heading"),
        [("History", "History"), ("Saved", "Saved recipes")],
    )
    def test_each_page_renders(self, app, page, heading):
        app.radio[0].set_value(page).run()
        assert app.exception == []
        assert app.title[0].value == heading

    def test_empty_pages_explain_themselves(self, app):
        app.radio[0].set_value("History").run()
        assert any("Nothing yet" in i.value for i in app.info)


class TestDegradedMode:
    def test_an_empty_search_asks_for_input_rather_than_running(self, app):
        app.text_area[0].set_value("   ")
        next(b for b in app.button if b.label == "Find a recipe").click().run()
        assert app.exception == []
        assert any("Add a few ingredients" in i.value for i in app.info)
