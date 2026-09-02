"""End-to-end tests for the sequential orchestrator.

These exercise the whole pipeline against the small fixture corpus, with every
stage either real or a deliberately misbehaving stub. Nothing here touches the
network.
"""

import pytest

from agents.composer import ComposerAgent
from agents.critic import CriticAgent, Finding, Verdict
from agents.nutrition import NutritionAgent, NutritionFacts
from agents.retrieval import RetrievalAgent
from agents.safety import SafetyAgent
from orchestrator import RecipeOrchestrator


class _StubNutritionSource:
    def __init__(self, table=None):
        self.table = table or {
            "egg": NutritionFacts(calories=143.0, protein_g=12.6),
            "butter": NutritionFacts(calories=717.0, fat_g=81.0),
            "rice": NutritionFacts(calories=360.0, protein_g=6.6),
        }

    def lookup(self, name):
        facts = self.table.get(name)
        return None if facts is None else (facts, f"{name}, raw")


@pytest.fixture
def orchestrator(sample_recipes):
    return RecipeOrchestrator(
        sample_recipes, nutrition=NutritionAgent(_StubNutritionSource())
    )


class TestHappyPath:
    def test_produces_an_accepted_recipe(self, orchestrator):
        result = orchestrator.run("3 eggs, butter and salt")
        assert result.accepted
        assert result.recipe is not None
        assert result.recipe.title == "Simple Omelette"

    def test_reports_the_parsed_pantry(self, orchestrator):
        result = orchestrator.run("3 eggs and a knob of butter")
        assert {i.name for i in result.pantry} == {"egg", "butter"}

    def test_have_flags_reflect_the_pantry(self, orchestrator):
        result = orchestrator.run("3 eggs, butter and salt")
        have = {i.name for i in result.recipe.ingredients if i.have}
        assert "egg" in have and "butter" in have

    def test_nutrition_is_attached_and_scaled(self, orchestrator):
        result = orchestrator.run("3 eggs, butter and salt", servings=2)
        assert result.recipe.servings == 2
        assert result.recipe.nutrition_per_serving.calories > 0

    def test_one_attempt_is_enough_when_nothing_is_wrong(self, orchestrator):
        assert orchestrator.run("3 eggs, butter and salt").attempt_count == 1

    def test_counts_are_reported(self, orchestrator):
        result = orchestrator.run("3 eggs, butter and salt")
        assert result.candidates_retrieved > 0
        assert result.candidates_safe > 0


class TestAllergenHandling:
    def test_unsafe_recipes_are_excluded(self, orchestrator):
        result = orchestrator.run("3 eggs, butter and salt", avoid=["dairy"])
        assert result.recipe is None or "butter" not in {
            i.name for i in result.recipe.ingredients
        }

    def test_a_recipe_is_still_found_when_a_safe_one_exists(self, orchestrator):
        result = orchestrator.run("rice, water and salt", avoid=["dairy"])
        assert result.accepted
        assert result.recipe.title == "Steamed Rice"

    def test_when_every_candidate_is_unsafe_the_reason_says_so(self, sample_recipes):
        dairy_only = [r for r in sample_recipes if r.title == "Simple Omelette"]
        orchestrator = RecipeOrchestrator(dairy_only)
        result = orchestrator.run("eggs and butter", avoid=["dairy"])
        assert result.recipe is None
        assert "dairy" in result.reason

    def test_the_accepted_recipe_never_contains_an_avoided_allergen(self, orchestrator):
        from agents.safety import detect_allergens

        result = orchestrator.run("rice, water, salt", avoid=["dairy", "gluten"])
        if result.recipe:
            for ingredient in result.recipe.ingredients:
                assert not (detect_allergens(ingredient.name) & {"milk", "wheat"})


class TestDegenerateInput:
    def test_unreadable_input_returns_a_reason_not_a_crash(self, orchestrator):
        result = orchestrator.run("!!! ???")
        assert result.recipe is None
        assert "No ingredients" in result.reason

    @pytest.mark.parametrize("text", ["", "   "])
    def test_empty_input(self, orchestrator, text):
        assert orchestrator.run(text).recipe is None

    def test_ingredients_that_match_nothing(self, orchestrator):
        result = orchestrator.run("dragonfruit and gochujang")
        assert result.recipe is None
        assert "No recipe" in result.reason


class _RejectingCritic(CriticAgent):
    """Rejects everything, to exercise the retry and fallback loops."""

    def __init__(self, message="always wrong"):
        super().__init__()
        self.calls = 0
        self._message = message

    def review(self, composed, **kwargs):
        self.calls += 1
        return Verdict((Finding("stub", self._message, "error"),))


class _RecordingComposer(ComposerAgent):
    """Records the feedback each compose call received."""

    def __init__(self):
        super().__init__()
        self.feedback_seen = []

    def compose(self, screened, nutrition, **kwargs):
        self.feedback_seen.append(list(kwargs.get("feedback", ())))
        return super().compose(screened, nutrition, **kwargs)


class TestRetryAndFallback:
    def test_a_rejected_composition_is_not_accepted(self, sample_recipes):
        orchestrator = RecipeOrchestrator(sample_recipes, critic=_RejectingCritic())
        result = orchestrator.run("3 eggs and butter")
        assert not result.accepted
        assert result.recipe is not None  # best effort is still returned
        assert "best attempt" in result.reason

    def test_deterministic_compositions_are_not_retried(self, sample_recipes):
        """The fallback composer would return the same steps every time."""
        composer = _RecordingComposer()
        orchestrator = RecipeOrchestrator(
            sample_recipes,
            composer=composer,
            critic=_RejectingCritic(),
            max_candidates=1,
        )
        orchestrator.run("3 eggs and butter")
        assert len(composer.feedback_seen) == 1

    def test_other_candidates_are_tried_when_one_fails(self, sample_recipes):
        orchestrator = RecipeOrchestrator(
            sample_recipes, critic=_RejectingCritic(), max_candidates=3
        )
        result = orchestrator.run("3 eggs, butter, rice and garlic")
        assert len({a.candidate_title for a in result.attempts}) > 1

    def test_candidate_count_is_bounded(self, sample_recipes):
        orchestrator = RecipeOrchestrator(
            sample_recipes, critic=_RejectingCritic(), max_candidates=2
        )
        result = orchestrator.run("3 eggs, butter, rice and garlic")
        assert len({a.candidate_title for a in result.attempts}) <= 2

    def test_every_attempt_is_recorded_with_its_verdict(self, sample_recipes):
        orchestrator = RecipeOrchestrator(sample_recipes, critic=_RejectingCritic())
        result = orchestrator.run("3 eggs and butter")
        assert all(not a.verdict.ok for a in result.attempts)
        assert all(a.candidate_title for a in result.attempts)

    @pytest.mark.parametrize(("attempts", "candidates"), [(0, 1), (1, 0), (-1, -1)])
    def test_bounds_must_be_at_least_one(self, sample_recipes, attempts, candidates):
        with pytest.raises(ValueError, match="at least 1"):
            RecipeOrchestrator(
                sample_recipes, max_attempts=attempts, max_candidates=candidates
            )


class TestInjection:
    def test_every_stage_can_be_replaced(self, sample_recipes):
        orchestrator = RecipeOrchestrator(
            sample_recipes,
            retrieval=RetrievalAgent(sample_recipes),
            safety=SafetyAgent(),
            nutrition=NutritionAgent(),
            composer=ComposerAgent(),
            critic=CriticAgent(),
        )
        assert orchestrator.run("3 eggs and butter").recipe is not None

    def test_runs_with_no_api_keys_at_all(self, sample_recipes):
        """The whole pipeline must work offline, which is what these tests do."""
        result = RecipeOrchestrator(sample_recipes).run("3 eggs, butter and salt")
        assert result.accepted
        assert result.recipe.composed_by == "fallback"

    def test_without_a_nutrition_source_the_recipe_still_composes(
        self, sample_recipes
    ):
        result = RecipeOrchestrator(sample_recipes).run("3 eggs, butter and salt")
        assert result.recipe.nutrition_per_serving.complete is False
        assert result.accepted


@pytest.fixture(scope="module")
def corpus():
    """The real 790-recipe corpus, skipped if it has not been built."""
    from agents.recipe import load_recipes
    from config import PROCESSED_DATA_DIR

    path = PROCESSED_DATA_DIR / "recipes.json"
    if not path.exists():
        pytest.skip("recipe corpus not built")
    return load_recipes(path)


class TestAgainstTheRealCorpus:
    def test_a_realistic_request_produces_an_accepted_recipe(self, corpus):
        result = RecipeOrchestrator(corpus).run(
            "chicken breast, 2 onions, 4 cloves garlic, tomatoes, cumin, olive oil"
        )
        assert result.accepted
        assert result.recipe.steps

    def test_ranking_prefers_a_recipe_that_uses_the_pantry(self, corpus):
        """A recipe of mostly staples must not win on a single match."""
        result = RecipeOrchestrator(corpus).run(
            "chicken breast, 2 onions, 4 cloves garlic, tomatoes, cumin, olive oil"
        )
        used = sum(1 for i in result.recipe.ingredients if i.have)
        assert used >= 2

    @pytest.mark.parametrize("avoid", [["dairy"], ["gluten"], ["dairy", "nuts"]])
    def test_allergen_runs_stay_safe(self, corpus, avoid):
        from agents.safety import detect_allergens, normalize_allergens

        result = RecipeOrchestrator(corpus).run(
            "chicken breast, onion, garlic, rice, tomatoes", avoid=avoid
        )
        if result.recipe:
            allergens = normalize_allergens(avoid)
            for ingredient in result.recipe.ingredients:
                assert not (detect_allergens(ingredient.name) & allergens)
