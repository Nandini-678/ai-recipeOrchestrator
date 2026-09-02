"""Unit tests for the safety agent.

Allergen detection is the one place in this pipeline where being wrong hurts
someone, so it is tested by enumeration rather than by sampling: every marker
category, every documented exception, and the cases that motivated the rules.
No network, no API key -- the substitution advisor is faked.
"""

import pytest

import agents.safety as safety_module
from agents.normalization import canonicalize
from agents.recipe import Recipe
from agents.retrieval import RecipeMatch
from agents.safety import (
    ALLERGEN_MARKERS,
    ALLERGEN_SAFE,
    KNOWN_ALLERGENS,
    SUBSTITUTIONS,
    GroqSubstitutionAdvisor,
    SafetyAgent,
    ScreenedMatch,
    Substitution,
    apply_substitutions,
    detect_allergens,
    normalize_allergens,
    suggest_substitution,
)


class TestDetectAllergens:
    @pytest.mark.parametrize(
        ("ingredient", "expected"),
        [
            ("cheddar cheese", {"milk"}),
            ("heavy cream", {"milk"}),
            ("ghee", {"milk"}),
            ("egg white", {"egg"}),
            ("mayonnaise", {"egg"}),
            ("peanut butter", {"peanut"}),
            ("walnut", {"tree nut"}),
            ("pine nut", {"tree nut"}),
            ("chestnut", {"tree nut"}),
            ("tofu", {"soy"}),
            ("all-purpose flour", {"wheat"}),
            ("couscous", {"wheat"}),
            ("salmon", {"fish"}),
            ("shrimp", {"shellfish"}),
            ("tahini", {"sesame"}),
        ],
    )
    def test_each_allergen_is_detected(self, ingredient, expected):
        assert set(detect_allergens(ingredient)) == expected

    @pytest.mark.parametrize(
        "ingredient", ["rice", "carrot", "olive oil", "basil", "chicken breast"]
    )
    def test_safe_ingredients_trigger_nothing(self, ingredient):
        assert detect_allergens(ingredient) == frozenset()

    @pytest.mark.parametrize(
        "ingredient",
        ["eggplant", "butternut squash", "nutmeg", "pineapple", "buckwheat"],
    )
    def test_token_matching_prevents_lookalike_false_positives(self, ingredient):
        """These contain an allergen word as a *substring*, not as a token."""
        assert detect_allergens(ingredient) == frozenset()

    @pytest.mark.parametrize(
        ("ingredient", "expected"),
        [
            ("almond milk", {"tree nut"}),
            ("soy milk", {"soy"}),
            ("oat milk", set()),
            ("coconut cream", {"tree nut"}),
            ("cocoa butter", set()),
            ("butter bean", set()),
            ("cream of tartar", set()),
            ("rice flour", set()),
            ("brown rice noodle", set()),
            ("rice stick noodle", set()),
            ("cassava flour", set()),
            ("corn tortilla", set()),
        ],
    )
    def test_generic_markers_are_negated_by_their_source(self, ingredient, expected):
        """"Butter" and "flour" name a form, not an ingredient."""
        assert set(detect_allergens(ingredient)) == expected

    @pytest.mark.parametrize(
        ("ingredient", "expected"),
        [
            ("egg noodle", {"egg", "wheat"}),
            ("bread flour", {"wheat"}),
            ("flour tortilla", {"wheat"}),
            ("ice cream", {"milk"}),
            ("buttermilk", {"milk"}),
            ("sour cream", {"milk"}),
        ],
    )
    def test_qualified_names_that_are_still_the_allergen(self, ingredient, expected):
        assert set(detect_allergens(ingredient)) == expected

    @pytest.mark.parametrize(
        ("ingredient", "expected"),
        [
            ("soy sauce", {"soy", "wheat"}),
            ("worcestershire sauce", {"fish"}),
            ("pesto", {"milk", "tree nut"}),
            ("oyster sauce", {"shellfish", "soy", "wheat"}),
            ("gravy", {"wheat"}),
        ],
    )
    def test_hidden_allergens_the_name_cannot_reveal(self, ingredient, expected):
        """Standard soy sauce is brewed with wheat; the name never says so."""
        assert set(detect_allergens(ingredient)) == expected

    @pytest.mark.parametrize(
        "ingredient", ["flax egg", "water chestnut", "oyster mushroom"]
    )
    def test_documented_exceptions(self, ingredient):
        assert detect_allergens(ingredient) == frozenset()

    def test_input_is_canonicalized_defensively(self):
        """Callers may pass raw text, not just parsed names."""
        assert detect_allergens("Chopped Cheddar Cheeses") == frozenset({"milk"})

    def test_multiple_allergens_are_all_reported(self):
        assert set(detect_allergens("egg noodle")) == {"egg", "wheat"}

    @pytest.mark.parametrize("blank", ["", "   ", "!!!"])
    def test_unparseable_input_is_not_an_allergen(self, blank):
        assert detect_allergens(blank) == frozenset()


class TestTableIntegrity:
    """The tables *are* the mechanism, so their shape is worth asserting."""

    def test_every_documented_exception_is_load_bearing(self):
        """A redundant exception is dead weight that can drift out of sync."""
        for allergen, names in ALLERGEN_SAFE.items():
            original = ALLERGEN_SAFE[allergen]
            for name in names:
                safety_module.ALLERGEN_SAFE[allergen] = original - {name}
                try:
                    triggered = allergen in detect_allergens(name)
                finally:
                    safety_module.ALLERGEN_SAFE[allergen] = original
                assert triggered, f"{name!r} is redundant in ALLERGEN_SAFE"

    def test_exception_keys_are_known_allergens(self):
        assert set(ALLERGEN_SAFE) <= KNOWN_ALLERGENS

    def test_substitution_names_are_canonical(self):
        """A non-canonical name would never match a pantry or an allergen rule."""
        for original, candidates in SUBSTITUTIONS.items():
            assert canonicalize(original) == original, original
            for replacement, _, _ in candidates:
                assert canonicalize(replacement) == replacement, replacement

    def test_no_substitution_replaces_an_ingredient_with_itself(self):
        for original, candidates in SUBSTITUTIONS.items():
            assert all(r != original for r, _, _ in candidates), original

    def test_substitution_ratios_are_positive(self):
        for candidates in SUBSTITUTIONS.values():
            assert all(ratio > 0 for _, ratio, _ in candidates)

    def test_marker_sets_do_not_silently_overlap_across_allergens(self):
        """Shared markers are legitimate but must be intentional, not typos."""
        shared = {}
        for allergen, markers in ALLERGEN_MARKERS.items():
            for other, other_markers in ALLERGEN_MARKERS.items():
                if allergen < other:
                    common = markers & other_markers
                    if common:
                        shared[(allergen, other)] = common
        assert shared == {("egg", "milk"): {"custard"}}


class TestNormalizeAllergens:
    @pytest.mark.parametrize(
        ("typed", "expected"),
        [
            ("dairy", "milk"),
            ("Dairy", "milk"),
            ("lactose", "milk"),
            ("nuts", "tree nut"),
            ("tree nuts", "tree nut"),
            ("gluten", "wheat"),
            ("peanuts", "peanut"),
            ("shell fish", "shellfish"),
        ],
    )
    def test_common_spellings_map_onto_categories(self, typed, expected):
        assert normalize_allergens([typed]) == frozenset({expected})

    def test_unknown_entries_are_dropped(self):
        assert normalize_allergens(["nightshades", "gluten"]) == frozenset({"wheat"})

    def test_empty_input(self):
        assert normalize_allergens([]) == frozenset()


class TestSuggestSubstitution:
    def test_returns_the_best_table_candidate(self):
        result = suggest_substitution("butter")
        assert result.replacement == "olive oil"
        assert result.ratio == 0.75
        assert result.source == "table"

    def test_prefers_a_candidate_the_user_already_has(self):
        result = suggest_substitution("butter", pantry={"coconut oil"})
        assert result.replacement == "coconut oil"
        assert result.in_pantry is True

    def test_unsafe_candidates_are_skipped(self):
        """Milk is the first choice for buttermilk; a dairy allergy rules it out."""
        assert suggest_substitution("buttermilk", avoid={"dairy"}) is None

    def test_falls_through_to_the_next_safe_candidate(self):
        result = suggest_substitution("peanut butter", avoid={"tree nut"})
        assert result.replacement == "sunflower seed butter"

    def test_unknown_ingredient_without_an_advisor_returns_none(self):
        assert suggest_substitution("dragonfruit") is None

    def test_advisor_is_not_consulted_when_the_table_has_an_answer(self):
        calls = []

        def advisor(missing, avoid):
            calls.append(missing)
            return Substitution(original=missing, replacement="tofu")

        suggest_substitution("butter", advisor=advisor)
        assert calls == []

    def test_advisor_is_consulted_when_the_table_has_nothing(self):
        def advisor(missing, avoid):
            return Substitution(original=missing, replacement="jackfruit", ratio=2.0)

        result = suggest_substitution("dragonfruit", advisor=advisor)
        assert result.replacement == "jackfruit"
        assert result.ratio == 2.0
        assert result.source == "llm"

    def test_an_unsafe_advisor_suggestion_is_rejected(self):
        """The advisor is never trusted on safety, whatever the prompt said."""

        def advisor(missing, avoid):
            return Substitution(original=missing, replacement="almond milk")

        result = suggest_substitution(
            "dragonfruit", avoid={"nuts"}, advisor=advisor
        )
        assert result is None

    def test_advisor_returning_nothing_is_handled(self):
        assert suggest_substitution("dragonfruit", advisor=lambda m, a: None) is None

    def test_advisor_result_records_pantry_membership(self):
        def advisor(missing, avoid):
            return Substitution(original=missing, replacement="jackfruit")

        result = suggest_substitution(
            "dragonfruit", pantry={"jackfruit"}, advisor=advisor
        )
        assert result.in_pantry is True


def _match(title, names, missing=()):
    """Build a RecipeMatch directly, bypassing retrieval."""
    recipe = Recipe(
        id=title, title=title, ingredients=[{"name": n} for n in names], steps=["x"]
    )
    return RecipeMatch(recipe=recipe, score=1.0, matched=(), missing=tuple(missing))


class TestSafetyAgent:
    def test_unsafe_recipes_are_dropped_by_default(self):
        matches = [
            _match("Omelette", ["egg", "butter"]),
            _match("Rice Bowl", ["rice", "carrot"]),
        ]
        screened = SafetyAgent().screen(matches, avoid=["dairy"])
        assert [s.recipe.title for s in screened] == ["Rice Bowl"]

    def test_violations_are_reported_when_unsafe_recipes_are_kept(self):
        matches = [_match("Omelette", ["egg", "cheddar cheese"])]
        screened = SafetyAgent().screen(matches, avoid=["dairy"], drop_unsafe=False)
        assert screened[0].is_safe is False
        assert screened[0].violations == {"cheddar cheese": frozenset({"milk"})}

    def test_every_violating_ingredient_is_named(self):
        matches = [_match("Bake", ["butter", "all-purpose flour", "rice"])]
        screened = SafetyAgent().screen(
            matches, avoid=["dairy", "gluten"], drop_unsafe=False
        )
        assert set(screened[0].violations) == {"butter", "all-purpose flour"}

    def test_no_allergens_means_nothing_is_dropped(self):
        matches = [_match("Omelette", ["egg", "butter"])]
        screened = SafetyAgent().screen(matches)
        assert len(screened) == 1 and screened[0].is_safe

    def test_substitutions_are_planned_for_missing_ingredients(self):
        matches = [_match("Cake", ["butter", "sugar"], missing=["butter"])]
        screened = SafetyAgent().screen(matches, pantry={"olive oil"})
        assert screened[0].substitutions[0].replacement == "olive oil"
        assert screened[0].substitutions[0].in_pantry is True

    def test_missing_ingredients_with_no_substitute_are_reported_unresolved(self):
        matches = [_match("Exotic", ["dragonfruit"], missing=["dragonfruit"])]
        screened = SafetyAgent().screen(matches)
        assert screened[0].unresolved == ("dragonfruit",)
        assert screened[0].substitutions == ()

    def test_substitutions_respect_the_users_allergens(self):
        matches = [_match("Bake", ["buttermilk"], missing=["buttermilk"])]
        screened = SafetyAgent().screen(matches, avoid=["dairy"], drop_unsafe=False)
        assert screened[0].unresolved == ("buttermilk",)

    def test_input_order_is_preserved(self):
        matches = [_match(t, ["rice"]) for t in ("A", "B", "C")]
        screened = SafetyAgent().screen(matches)
        assert [s.recipe.title for s in screened] == ["A", "B", "C"]

    def test_empty_input(self):
        assert SafetyAgent().screen([]) == []

    def test_composes_with_the_retrieval_agent(self, sample_recipes):
        """Retrieval output must feed straight in with no glue."""
        from agents.ingredient import parse_ingredients
        from agents.retrieval import RetrievalAgent

        pantry = parse_ingredients("3 eggs, butter and salt")
        matches = RetrievalAgent(sample_recipes).retrieve(pantry, top_n=3)
        screened = SafetyAgent().screen(
            matches, pantry={i.name for i in pantry}, avoid=["dairy"]
        )
        assert all("butter" not in s.recipe.ingredient_names for s in screened)


class _FakeCompletion:
    def __init__(self, content):
        self.choices = [
            type("Choice", (), {"message": type("Msg", (), {"content": content})()})()
        ]


class _FakeGroqClient:
    def __init__(self, content, raises=None):
        self._content, self._raises = content, raises
        self.chat = type(
            "Chat", (), {"completions": type("C", (), {"create": self._create})()}
        )()

    def _create(self, **kwargs):
        if self._raises:
            raise self._raises
        return _FakeCompletion(self._content)


class TestGroqSubstitutionAdvisor:
    def test_valid_response_becomes_a_substitution(self):
        client = _FakeGroqClient(
            '{"replacement": "Jackfruit", "ratio": 1.5, "note": "shreds well"}'
        )
        advisor = GroqSubstitutionAdvisor(client=client, model="fake")
        result = advisor("pulled pork", frozenset())
        assert result.replacement == "jackfruit"
        assert result.ratio == 1.5
        assert result.source == "llm"

    def test_replacement_is_canonicalized(self):
        client = _FakeGroqClient('{"replacement": "Spring Onions", "ratio": 1}')
        advisor = GroqSubstitutionAdvisor(client=client, model="fake")
        assert advisor("chive", frozenset()).replacement == "green onion"

    def test_explicit_null_means_no_suggestion(self):
        client = _FakeGroqClient('{"replacement": null, "ratio": 1}')
        advisor = GroqSubstitutionAdvisor(client=client, model="fake")
        assert advisor("x", frozenset()) is None

    @pytest.mark.parametrize(
        "content",
        ['not json', '{"ratio": "lots"}', '{"replacement": "x", "ratio": -1}', "{}"],
    )
    def test_malformed_responses_yield_no_suggestion(self, content):
        advisor = GroqSubstitutionAdvisor(client=_FakeGroqClient(content), model="fake")
        assert advisor("butter", frozenset()) is None

    def test_a_failing_client_yields_no_suggestion(self):
        client = _FakeGroqClient(None, raises=RuntimeError("network down"))
        advisor = GroqSubstitutionAdvisor(client=client, model="fake")
        assert advisor("butter", frozenset()) is None

    def test_suggesting_the_same_ingredient_back_is_rejected(self):
        client = _FakeGroqClient('{"replacement": "butter", "ratio": 1}')
        advisor = GroqSubstitutionAdvisor(client=client, model="fake")
        assert advisor("butter", frozenset()) is None


class TestApplySubstitutions:
    """Nutrition must be computed on the list the user will actually cook."""

    def _screened_with(self, ingredients, substitutions=()):
        recipe = Recipe(
            id="1", title="T",
            ingredients=[dict(i) for i in ingredients], steps=["x"],
        )
        return ScreenedMatch(
            match=RecipeMatch(recipe=recipe, score=1.0),
            violations={},
            substitutions=tuple(substitutions),
        )

    def test_replaces_the_name_and_scales_the_quantity(self):
        screened = self._screened_with(
            [{"name": "butter", "quantity": 100, "unit": "g"}],
            [Substitution(original="butter", replacement="olive oil", ratio=0.75)],
        )
        resolved = apply_substitutions(screened)
        assert (resolved[0].name, resolved[0].quantity) == ("olive oil", 75.0)
        assert resolved[0].unit == "g"

    def test_untouched_ingredients_pass_through(self):
        screened = self._screened_with([{"name": "rice", "quantity": 2, "unit": "cup"}])
        resolved = apply_substitutions(screened)
        assert (resolved[0].name, resolved[0].quantity) == ("rice", 2.0)

    def test_unquantified_ingredients_stay_unquantified(self):
        screened = self._screened_with(
            [{"name": "butter"}],
            [Substitution(original="butter", replacement="olive oil", ratio=0.75)],
        )
        assert apply_substitutions(screened)[0].quantity is None

    def test_the_original_recipe_is_not_mutated(self):
        screened = self._screened_with(
            [{"name": "butter", "quantity": 100, "unit": "g"}],
            [Substitution(original="butter", replacement="olive oil", ratio=0.75)],
        )
        apply_substitutions(screened)
        assert screened.recipe.ingredients[0].name == "butter"

    def test_order_is_preserved(self):
        screened = self._screened_with(
            [{"name": "rice"}, {"name": "butter"}, {"name": "salt"}],
            [Substitution(original="butter", replacement="olive oil")],
        )
        assert [i.name for i in apply_substitutions(screened)] == [
            "rice", "olive oil", "salt",
        ]

    def test_agrees_with_what_the_composer_puts_in_the_ingredient_list(self):
        """The two paths derive from the same substitution plan and must match."""
        from agents.composer import ComposerAgent
        from agents.nutrition import NutritionReport

        screened = self._screened_with(
            [{"name": "butter", "quantity": 100, "unit": "g"}, {"name": "rice"}],
            [Substitution(original="butter", replacement="olive oil", ratio=0.75)],
        )
        resolved = apply_substitutions(screened)
        composed = ComposerAgent().compose(screened, NutritionReport(servings=4))
        assert [i.name for i in resolved] == [i.name for i in composed.ingredients]
        assert [i.quantity for i in resolved] == [
            i.quantity for i in composed.ingredients
        ]
