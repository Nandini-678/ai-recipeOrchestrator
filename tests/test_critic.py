"""Unit tests for the critic agent.

The critic's checks are mechanical, so they are tested by construction: build a
composed recipe with a specific defect and assert exactly that finding fires.
"""

import pytest

from agents.composer import ComposedIngredient, ComposedRecipe, NutritionSummary
from agents.critic import (
    CriticAgent,
    Finding,
    Verdict,
    mentioned_ingredients,
)
from agents.recipe import Recipe
from agents.retrieval import RecipeMatch
from agents.safety import ScreenedMatch, Substitution


def _composed(ingredients, steps=("Cook it.",), *, composed_by="llm", **kwargs):
    return ComposedRecipe(
        title=kwargs.pop("title", "Test Dish"),
        servings=kwargs.pop("servings", 4),
        ingredients=[
            i if isinstance(i, ComposedIngredient) else ComposedIngredient(name=i)
            for i in ingredients
        ],
        steps=list(steps),
        nutrition_per_serving=NutritionSummary(
            calories=100, protein_g=1, fat_g=1, carbs_g=1,
            fiber_g=0, sugar_g=0, sodium_mg=0,
        ),
        composed_by=composed_by,
        **kwargs,
    )


def _screened(names, substitutions=()):
    recipe = Recipe(
        id="1", title="T", ingredients=[{"name": n} for n in names], steps=["x"]
    )
    return ScreenedMatch(
        match=RecipeMatch(recipe=recipe, score=1.0),
        violations={},
        substitutions=tuple(substitutions),
    )


class TestMentionedIngredients:
    def test_finds_single_and_multi_word_names(self):
        found = mentioned_ingredients("Fry the onions, then add the chicken.")
        assert found == {"onion", "chicken"}

    def test_longest_match_wins(self):
        """"olive oil" is one ingredient, not also a mention of "olive"."""
        assert mentioned_ingredients("a splash of olive oil") == {"olive oil"}

    def test_plurals_are_matched(self):
        assert "tomato" in mentioned_ingredients("Add the tomatoes.")

    def test_regional_names_are_resolved(self):
        assert "eggplant" in mentioned_ingredients("Slice the aubergines.")

    def test_unknown_words_are_not_reported(self):
        assert mentioned_ingredients("Stir vigorously with a spurtle.") == set()

    def test_empty_text(self):
        assert mentioned_ingredients("") == set()


class TestStructureChecks:
    def test_a_sound_recipe_passes(self):
        verdict = CriticAgent().review(_composed(["rice"], ("Cook the rice.",)))
        assert verdict.ok
        assert verdict.findings == ()

    def test_missing_ingredients_are_flagged(self):
        verdict = CriticAgent().review(_composed([], ("Cook it.",)))
        assert not verdict.ok
        assert any(f.code == "no_ingredients" for f in verdict.errors)

    def test_blank_steps_are_flagged(self):
        composed = _composed(["rice"], ("Cook the rice.",))
        composed.steps.append("   ")
        errors = CriticAgent().review(composed).errors
        assert any(f.code == "blank_step" for f in errors)


class TestAllergenChecks:
    def test_an_avoided_allergen_in_the_list_is_an_error(self):
        verdict = CriticAgent().review(
            _composed(["cheddar cheese"], ("Melt it.",)), avoid=["dairy"]
        )
        assert not verdict.ok
        assert any(f.code == "allergen_in_ingredients" for f in verdict.errors)

    def test_an_avoided_allergen_hidden_in_a_step_is_an_error(self):
        """A clean list does not make a recipe safe if the steps add butter."""
        verdict = CriticAgent().review(
            _composed(["rice"], ("Finish with a knob of butter.",)), avoid=["dairy"]
        )
        assert any(f.code == "allergen_in_steps" for f in verdict.errors)

    def test_the_finding_names_the_allergen_and_the_step(self):
        verdict = CriticAgent().review(
            _composed(["rice"], ("Stir.", "Add cheddar cheese.")), avoid=["dairy"]
        )
        message = verdict.errors[0].message
        assert "Step 2" in message and "milk" in message

    def test_no_allergens_means_no_allergen_findings(self):
        verdict = CriticAgent().review(_composed(["cheddar cheese"], ("Melt it.",)))
        assert verdict.ok

    def test_an_unrelated_allergy_does_not_fire(self):
        verdict = CriticAgent().review(
            _composed(["cheddar cheese"], ("Melt it.",)), avoid=["shellfish"]
        )
        assert verdict.ok


class TestInventedIngredientCheck:
    def test_model_written_steps_using_an_unlisted_ingredient_are_an_error(self):
        verdict = CriticAgent().review(
            _composed(["rice"], ("Cook the rice with the shrimp.",), composed_by="llm")
        )
        assert not verdict.ok
        finding = verdict.errors[0]
        assert finding.code == "invented_ingredient"
        assert "shrimp" in finding.message

    def test_source_written_steps_only_warn(self):
        """Retrying cannot change steps copied verbatim from the dataset."""
        verdict = CriticAgent().review(
            _composed(
                ["rice"], ("Cook the rice with the shrimp.",), composed_by="fallback"
            )
        )
        assert verdict.ok
        assert any(f.code == "invented_ingredient" for f in verdict.warnings)

    def test_listed_ingredients_are_fine(self):
        verdict = CriticAgent().review(
            _composed(["rice", "shrimp"], ("Cook the rice with the shrimp.",))
        )
        assert verdict.ok

    def test_staples_may_be_mentioned_without_being_listed(self):
        verdict = CriticAgent().review(
            _composed(["rice"], ("Boil the rice in salted water.",))
        )
        assert verdict.ok

    def test_a_token_of_a_listed_ingredient_is_allowed(self):
        """"Add the chicken" is fine when the list says "chicken breast"."""
        verdict = CriticAgent().review(
            _composed(["chicken breast"], ("Sear the chicken.",))
        )
        assert verdict.ok


class TestSubstitutionChecks:
    def test_a_replaced_ingredient_left_in_the_list_is_an_error(self):
        screened = _screened(
            ["butter"],
            [Substitution(original="butter", replacement="olive oil", ratio=0.75)],
        )
        verdict = CriticAgent().review(
            _composed(["butter"], ("Melt the butter.",)), screened=screened
        )
        assert any(f.code == "substitution_not_applied" for f in verdict.errors)

    def test_a_replaced_ingredient_still_named_in_a_step_is_an_error(self):
        screened = _screened(
            ["butter"],
            [Substitution(original="butter", replacement="olive oil", ratio=0.75)],
        )
        composed = _composed(
            [ComposedIngredient(name="olive oil", substituted_for="butter")],
            ("Melt the butter in a pan.",),
        )
        verdict = CriticAgent().review(composed, screened=screened)
        codes = {f.code for f in verdict.errors}
        assert "substitution_ignored_in_steps" in codes

    def test_the_feedback_names_the_replacement(self):
        screened = _screened(
            ["butter"], [Substitution(original="butter", replacement="olive oil")]
        )
        composed = _composed(
            [ComposedIngredient(name="olive oil", substituted_for="butter")],
            ("Melt the butter.",),
        )
        feedback = " ".join(CriticAgent().review(composed, screened=screened).feedback)
        assert "olive oil" in feedback and "butter" in feedback

    def test_a_correctly_applied_substitution_passes(self):
        screened = _screened(
            ["butter"], [Substitution(original="butter", replacement="olive oil")]
        )
        composed = _composed(
            [ComposedIngredient(name="olive oil", substituted_for="butter")],
            ("Heat the olive oil.",),
        )
        assert CriticAgent().review(composed, screened=screened).ok


class TestHaveFlagChecks:
    def test_a_wrong_have_flag_is_a_warning_not_an_error(self):
        composed = _composed(
            [ComposedIngredient(name="rice", have=True)], ("Cook the rice.",)
        )
        verdict = CriticAgent().review(composed, pantry=set())
        assert verdict.ok
        assert any(f.code == "wrong_have_flag" for f in verdict.warnings)

    def test_correct_flags_produce_nothing(self):
        composed = _composed(
            [ComposedIngredient(name="rice", have=True)], ("Cook the rice.",)
        )
        assert CriticAgent().review(composed, pantry={"rice"}).findings == ()

    def test_staples_are_exempt(self):
        composed = _composed(
            [ComposedIngredient(name="salt", have=True, assumed_staple=True)],
            ("Season.",),
        )
        assert CriticAgent().review(composed, pantry=set()).findings == ()


class TestVerdict:
    def test_errors_block_and_warnings_do_not(self):
        verdict = Verdict((
            Finding("a", "bad", "error"),
            Finding("b", "meh", "warning"),
        ))
        assert not verdict.ok
        assert len(verdict.errors) == 1 and len(verdict.warnings) == 1

    def test_feedback_carries_only_blocking_messages(self):
        verdict = Verdict((
            Finding("a", "fix this", "error"),
            Finding("b", "just so you know", "warning"),
        ))
        assert verdict.feedback == ["fix this"]

    def test_an_empty_verdict_passes(self):
        assert Verdict().ok


class TestAgainstTheRealCorpus:
    """The critic must not reject the dataset wholesale."""

    def test_source_recipes_are_never_blocked(self, sample_recipes):
        from agents.composer import ComposerAgent
        from agents.nutrition import NutritionReport

        composer, critic = ComposerAgent(), CriticAgent()
        for recipe in sample_recipes:
            screened = ScreenedMatch(
                match=RecipeMatch(recipe=recipe, score=1.0), violations={}
            )
            composed = composer.compose(
                screened, NutritionReport(servings=4), pantry=set()
            )
            assert critic.review(composed, screened=screened).ok, recipe.title


@pytest.mark.parametrize("avoid", [(), ("dairy",), ("gluten", "nuts")])
def test_review_never_raises_on_valid_input(avoid):
    composed = _composed(["rice", "tofu"], ("Cook.", "Serve."))
    assert isinstance(CriticAgent().review(composed, avoid=avoid), Verdict)
