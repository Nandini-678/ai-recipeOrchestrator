"""Unit tests for the composer agent.

Two properties carry most of the weight here:

1. A model response is validated strictly before any part of it is used.
2. Nothing factual comes from the model. Quantities, have/missing flags and
   nutrition are assembled in code, so a model that lies about them cannot
   change the output.

Offline throughout: the Groq client is faked.
"""

import pytest
from pydantic import ValidationError

from agents.composer import (
    ComposedRecipe,
    ComposerAgent,
    ComposerError,
    _ComposerResponse,
)
from agents.nutrition import IngredientNutrition, NutritionFacts, NutritionReport
from agents.recipe import Recipe
from agents.retrieval import RecipeMatch
from agents.safety import ScreenedMatch, Substitution


def _screened(
    ingredients,
    *,
    title="Test Dish",
    steps=("Chop things.", "Cook things."),
    matched=(),
    missing=(),
    staples=(),
    substitutions=(),
    unresolved=(),
    violations=None,
):
    recipe = Recipe(
        id="42",
        title=title,
        ingredients=[
            dict(i) if isinstance(i, dict) else {"name": i} for i in ingredients
        ],
        steps=list(steps),
    )
    match = RecipeMatch(
        recipe=recipe,
        score=1.0,
        matched=tuple(matched),
        missing=tuple(missing),
        staples_assumed=tuple(staples),
    )
    return ScreenedMatch(
        match=match,
        violations=violations or {},
        substitutions=tuple(substitutions),
        unresolved=tuple(unresolved),
    )


def _report(servings=4, measured=(), unestimated=()):
    contributions = tuple(
        IngredientNutrition(
            name=name,
            grams=100.0,
            facts=facts,
            source_description=f"{name}, raw",
            exact_mass=True,
            basis="mass",
        )
        for name, facts in measured
    )
    return NutritionReport(
        servings=servings, contributions=contributions, unestimated=tuple(unestimated)
    )


class TestComposerResponseSchema:
    def test_a_well_formed_response_validates(self):
        parsed = _ComposerResponse.model_validate_json(
            '{"title": "Soup", "summary": "Warm.", "prep_time_minutes": 10,'
            ' "cook_time_minutes": 20, "steps": ["Chop.", "Boil."]}'
        )
        assert parsed.title == "Soup"
        assert parsed.steps == ["Chop.", "Boil."]

    def test_unexpected_fields_are_rejected(self):
        """A response carrying fields we never asked for ignored instructions."""
        with pytest.raises(ValidationError):
            _ComposerResponse.model_validate(
                {"title": "Soup", "steps": ["Boil."], "calories": 900}
            )

    @pytest.mark.parametrize(
        "payload",
        [
            {"title": "Soup"},                                  # no steps
            {"title": "Soup", "steps": []},                     # empty steps
            {"title": "Soup", "steps": ["", "  "]},             # blank steps
            {"title": "", "steps": ["Boil."]},                  # empty title
            {"title": "S", "steps": ["Boil."], "prep_time_minutes": -5},
            {"title": "S", "steps": ["Boil."], "cook_time_minutes": 99999},
        ],
    )
    def test_malformed_responses_are_rejected(self, payload):
        with pytest.raises(ValidationError):
            _ComposerResponse.model_validate(payload)

    def test_blank_steps_are_stripped_but_real_ones_kept(self):
        parsed = _ComposerResponse.model_validate(
            {"title": "Soup", "steps": ["  Chop.  ", "", "Boil."]}
        )
        assert parsed.steps == ["Chop.", "Boil."]

    def test_times_are_optional(self):
        parsed = _ComposerResponse.model_validate({"title": "S", "steps": ["Boil."]})
        assert parsed.prep_time_minutes is None


class TestFactualAssembly:
    """Everything in this class must hold with no model involved at all."""

    def test_have_flags_come_from_the_pantry(self):
        screened = _screened(["rice", "tofu"])
        composed = ComposerAgent().compose(screened, _report(), pantry={"rice"})
        flags = {i.name: i.have for i in composed.ingredients}
        assert flags == {"rice": True, "tofu": False}

    def test_quantities_are_carried_through_unchanged(self):
        screened = _screened([{"name": "rice", "quantity": 2, "unit": "cup"}])
        composed = ComposerAgent().compose(screened, _report(), pantry={"rice"})
        assert composed.ingredients[0].quantity == 2.0
        assert composed.ingredients[0].unit == "cup"

    def test_assumed_staples_are_flagged_separately(self):
        screened = _screened(["rice", "salt"], staples=("salt",))
        composed = ComposerAgent().compose(screened, _report(), pantry={"rice"})
        salt = next(i for i in composed.ingredients if i.name == "salt")
        assert salt.assumed_staple is True

    def test_substitutions_replace_the_ingredient_and_scale_the_quantity(self):
        screened = _screened(
            [{"name": "butter", "quantity": 100, "unit": "g"}],
            substitutions=(
                Substitution(
                    original="butter", replacement="olive oil", ratio=0.75,
                    note="use three quarters", in_pantry=True,
                ),
            ),
        )
        composed = ComposerAgent().compose(screened, _report(), pantry=set())
        line = composed.ingredients[0]
        assert line.name == "olive oil"
        assert line.quantity == 75.0
        assert line.substituted_for == "butter"
        assert line.note == "use three quarters"
        assert line.have is True

    def test_substitutions_are_also_listed_separately(self):
        screened = _screened(
            ["butter"],
            substitutions=(
                Substitution(original="butter", replacement="olive oil", ratio=0.75),
            ),
        )
        composed = ComposerAgent().compose(screened, _report(), pantry=set())
        assert [s.substituted_for for s in composed.substitutions] == ["butter"]

    def test_a_substitution_with_no_quantity_stays_unquantified(self):
        screened = _screened(
            ["butter"],
            substitutions=(
                Substitution(original="butter", replacement="olive oil", ratio=0.75),
            ),
        )
        composed = ComposerAgent().compose(screened, _report(), pantry=set())
        assert composed.ingredients[0].quantity is None

    def test_ingredient_order_follows_the_source_recipe(self):
        screened = _screened(["rice", "tofu", "soy sauce"])
        composed = ComposerAgent().compose(screened, _report(), pantry=set())
        assert [i.name for i in composed.ingredients] == ["rice", "tofu", "soy sauce"]

    def test_missing_ingredients_are_derivable(self):
        screened = _screened(["rice", "tofu", "salt"], staples=("salt",))
        composed = ComposerAgent().compose(screened, _report(), pantry={"rice"})
        assert composed.missing_ingredients == ["tofu"]


class TestNutritionSummary:
    def test_comes_from_the_nutrition_agent_not_the_model(self):
        report = _report(
            servings=4,
            measured=[("rice", NutritionFacts(calories=800.0, protein_g=20.0))],
        )
        composed = ComposerAgent().compose(_screened(["rice"]), report)
        assert composed.nutrition_per_serving.calories == 200.0
        assert composed.nutrition_per_serving.protein_g == 5.0

    def test_reports_incomplete_coverage(self):
        report = _report(
            measured=[("rice", NutritionFacts(calories=400.0))],
            unestimated=[("saffron", "no USDA match")],
        )
        composed = ComposerAgent().compose(_screened(["rice", "saffron"]), report)
        summary = composed.nutrition_per_serving
        assert summary.complete is False
        assert (summary.measured_ingredients, summary.total_ingredients) == (1, 2)

    def test_complete_coverage_is_marked_complete(self):
        report = _report(measured=[("rice", NutritionFacts(calories=400.0))])
        composed = ComposerAgent().compose(_screened(["rice"]), report)
        assert composed.nutrition_per_serving.complete is True

    def test_servings_default_to_the_nutrition_report(self):
        composed = ComposerAgent().compose(_screened(["rice"]), _report(servings=6))
        assert composed.servings == 6

    def test_servings_can_be_overridden(self):
        composed = ComposerAgent().compose(
            _screened(["rice"]), _report(servings=6), servings=2
        )
        assert composed.servings == 2


class TestWarnings:
    def test_unresolved_missing_ingredients_are_warned_about(self):
        screened = _screened(["saffron"], unresolved=("saffron",))
        composed = ComposerAgent().compose(screened, _report(), pantry=set())
        assert any("No substitute found" in w for w in composed.warnings)

    def test_allergen_violations_are_warned_about(self):
        screened = _screened(["butter"], violations={"butter": frozenset({"milk"})})
        composed = ComposerAgent().compose(screened, _report(), pantry=set())
        assert any("avoided allergens" in w for w in composed.warnings)

    def test_incomplete_nutrition_is_warned_about(self):
        report = _report(
            measured=[("rice", NutritionFacts(calories=400.0))],
            unestimated=[("saffron", "no USDA match")],
        )
        composed = ComposerAgent().compose(_screened(["rice", "saffron"]), report)
        assert any("saffron" in w for w in composed.warnings)

    def test_a_clean_recipe_has_no_warnings(self):
        report = _report(measured=[("rice", NutritionFacts(calories=400.0))])
        composed = ComposerAgent().compose(_screened(["rice"]), report, pantry={"rice"})
        assert composed.warnings == []


class TestFallback:
    def test_without_a_client_the_source_instructions_are_used(self):
        screened = _screened(["rice"], steps=("Rinse rice.", "Simmer 20 minutes."))
        composed = ComposerAgent().compose(screened, _report())
        assert composed.steps == ["Rinse rice.", "Simmer 20 minutes."]
        assert composed.composed_by == "fallback"

    def test_the_source_title_is_used(self):
        composed = ComposerAgent().compose(
            _screened(["rice"], title="Plain Rice"), _report()
        )
        assert composed.title == "Plain Rice"

    def test_a_recipe_with_no_instructions_still_composes(self):
        """ComposedRecipe requires at least one step, so one is supplied."""
        composed = ComposerAgent().compose(_screened(["rice"], steps=()), _report())
        assert len(composed.steps) == 1
        assert any("no instructions" in w for w in composed.warnings)

    def test_source_traceability_is_recorded(self):
        composed = ComposerAgent().compose(
            _screened(["rice"], title="Plain Rice"), _report()
        )
        assert composed.source_recipe_id == "42"
        assert composed.source_title == "Plain Rice"


class _FakeCompletion:
    def __init__(self, content):
        self.choices = [
            type("Choice", (), {"message": type("Msg", (), {"content": content})()})()
        ]


class _FakeGroqClient:
    def __init__(self, content, raises=None):
        self._content, self._raises = content, raises
        self.prompts = []
        self.chat = type(
            "Chat", (), {"completions": type("C", (), {"create": self._create})()}
        )()

    def _create(self, **kwargs):
        self.prompts.append(kwargs)
        if self._raises:
            raise self._raises
        return _FakeCompletion(self._content)


GOOD = (
    '{"title": "Weeknight Rice", "summary": "Quick and simple.",'
    ' "prep_time_minutes": 5, "cook_time_minutes": 20,'
    ' "steps": ["Rinse the rice.", "Simmer until tender."]}'
)


class TestModelComposition:
    def test_prose_comes_from_the_model(self):
        agent = ComposerAgent(client=_FakeGroqClient(GOOD), model="fake")
        composed = agent.compose(_screened(["rice"]), _report())
        assert composed.title == "Weeknight Rice"
        assert composed.summary == "Quick and simple."
        assert composed.prep_time_minutes == 5
        assert composed.steps == ["Rinse the rice.", "Simmer until tender."]
        assert composed.composed_by == "llm"

    def test_facts_still_come_from_code_when_a_model_is_used(self):
        """The model writes prose; it cannot move a number."""
        report = _report(measured=[("rice", NutritionFacts(calories=800.0))])
        agent = ComposerAgent(client=_FakeGroqClient(GOOD), model="fake")
        composed = agent.compose(
            _screened([{"name": "rice", "quantity": 2, "unit": "cup"}]),
            report,
            pantry={"rice"},
        )
        assert composed.nutrition_per_serving.calories == 200.0
        assert composed.ingredients[0].quantity == 2.0
        assert composed.ingredients[0].have is True

    def test_the_prompt_carries_the_resolved_ingredient_list(self):
        client = _FakeGroqClient(GOOD)
        agent = ComposerAgent(client=client, model="fake")
        agent.compose(
            _screened(
                ["butter"],
                substitutions=(
                    Substitution(
                        original="butter", replacement="olive oil", ratio=0.75
                    ),
                ),
            ),
            _report(),
        )
        sent = client.prompts[0]["messages"][1]["content"]
        assert "olive oil" in sent
        assert "replaces butter" in sent

    def test_retry_feedback_reaches_the_prompt(self):
        """A retry must say what was wrong, not just ask again."""
        client = _FakeGroqClient(GOOD)
        agent = ComposerAgent(client=client, model="fake")
        agent.compose(
            _screened(["rice"]),
            _report(),
            feedback=["Step 5 still says 'egg'; it was replaced with 'flaxseed'."],
        )
        sent = client.prompts[0]["messages"][1]["content"]
        assert "Step 5 still says" in sent
        assert "rejected" in sent.lower()

    def test_no_feedback_means_no_retry_section(self):
        client = _FakeGroqClient(GOOD)
        ComposerAgent(client=client, model="fake").compose(
            _screened(["rice"]), _report()
        )
        assert "rejected" not in client.prompts[0]["messages"][1]["content"].lower()

    @pytest.mark.parametrize(
        "content",
        ["not json", "{}", '{"title": "X"}', '{"title": "X", "steps": []}',
         '{"title": "X", "steps": ["Go."], "nutrition": {"calories": 900}}'],
    )
    def test_invalid_responses_fall_back_to_the_source(self, content):
        screened = _screened(["rice"], steps=("Original step.",))
        agent = ComposerAgent(client=_FakeGroqClient(content), model="fake")
        composed = agent.compose(screened, _report())
        assert composed.steps == ["Original step."]
        assert composed.composed_by == "fallback"
        assert any("original instructions" in w for w in composed.warnings)

    def test_a_failing_client_falls_back_rather_than_raising(self):
        screened = _screened(["rice"], steps=("Original step.",))
        client = _FakeGroqClient(None, raises=RuntimeError("network down"))
        agent = ComposerAgent(client=client, model="fake")
        composed = agent.compose(screened, _report())
        assert composed.composed_by == "fallback"
        assert composed.steps == ["Original step."]

    def test_composer_error_names_the_cause(self):
        agent = ComposerAgent(client=_FakeGroqClient("garbage"), model="fake")
        with pytest.raises(ComposerError):
            agent._write_prose(_screened(["rice"]), [])


class TestComposedRecipeContract:
    def test_the_result_is_always_a_valid_composed_recipe(self):
        composed = ComposerAgent().compose(_screened(["rice"]), _report())
        assert ComposedRecipe.model_validate(composed.model_dump()) == composed

    def test_it_serializes_to_json(self):
        composed = ComposerAgent().compose(_screened(["rice"]), _report())
        assert '"title"' in composed.model_dump_json()

    @pytest.mark.parametrize("servings", [0, -3])
    def test_servings_must_be_positive(self, servings):
        with pytest.raises(ValidationError):
            ComposerAgent().compose(_screened(["rice"]), _report(), servings=servings)
