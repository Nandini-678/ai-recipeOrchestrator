"""Unit tests for the nutrition agent.

The headline property is that serving-scale math is exact and linear, so the
scaling tests use dyadic factors (halves, quarters) where binary floating point
is exact and equality can be asserted outright rather than approximated.

Fully offline: the USDA client is driven by a fake session.
"""

import json

import pytest

from agents.ingredient import Ingredient, parse_ingredients
from agents.nutrition import (
    MASS_TO_GRAMS,
    SEARCH_ALIASES,
    VOLUME_TO_ML,
    FoodDataCentralClient,
    NutritionAgent,
    NutritionFacts,
    NutritionLookupError,
    NutritionReport,
    _rank_score,
    scale_ingredients,
    to_grams,
)


class TestMassConversion:
    """Mass conversions are definitions, so they are asserted exactly."""

    @pytest.mark.parametrize(
        ("quantity", "unit", "grams"),
        [
            (1, "g", 1.0),
            (2, "kg", 2000.0),
            (500, "mg", 0.5),
            (1, "lb", 453.59237),
            (1, "oz", 28.349523125),
            (16, "oz", 453.59237),
        ],
    )
    def test_exact(self, quantity, unit, grams):
        assert to_grams(quantity, unit, "anything").grams == grams

    def test_mass_is_flagged_exact(self):
        assert to_grams(1, "kg", "flour").exact is True

    def test_mass_does_not_depend_on_the_ingredient(self):
        assert to_grams(1, "lb", "flour").grams == to_grams(1, "lb", "lead").grams

    def test_pound_is_sixteen_ounces_exactly(self):
        assert MASS_TO_GRAMS["lb"] == 16 * MASS_TO_GRAMS["oz"]


class TestVolumeConversion:
    def test_uses_the_ingredients_density(self):
        """A cup of honey weighs far more than a cup of water."""
        water = to_grams(1, "cup", "water").grams
        honey = to_grams(1, "cup", "honey").grams
        assert water == pytest.approx(236.588, abs=0.01)
        assert honey == pytest.approx(336.0, abs=1.0)
        assert honey > water * 1.4

    def test_flour_is_much_lighter_than_water(self):
        assert to_grams(1, "cup", "all-purpose flour").grams == pytest.approx(
            125.4, abs=0.5
        )

    def test_unknown_ingredients_fall_back_to_water(self):
        estimate = to_grams(1, "cup", "dragonfruit puree")
        assert estimate.grams == pytest.approx(236.588, abs=0.01)
        assert "assumed water" in estimate.basis

    def test_volume_is_never_flagged_exact(self):
        assert to_grams(1, "cup", "water").exact is False

    def test_spoon_measures_are_consistent_with_cups(self):
        assert VOLUME_TO_ML["cup"] == pytest.approx(16 * VOLUME_TO_ML["tbsp"])
        assert VOLUME_TO_ML["tbsp"] == pytest.approx(3 * VOLUME_TO_ML["tsp"])


class TestCountConversion:
    @pytest.mark.parametrize(
        ("quantity", "unit", "name", "grams"),
        [
            (2, None, "egg", 100.0),
            (1, None, "onion", 110.0),
            (3, "clove", "garlic", 9.0),
            (2, "slice", "bread", 56.0),
            (1, "stick", "butter", 113.0),
            (1, "stalk", "celery", 40.0),
        ],
    )
    def test_typical_item_weights(self, quantity, unit, name, grams):
        assert to_grams(quantity, unit, name).grams == grams

    def test_pair_weights_beat_generic_unit_weights(self):
        """A clove of garlic is 3g; the generic clove fallback must not win."""
        assert to_grams(1, "clove", "garlic").grams == 3.0

    def test_unknown_countable_unit_falls_back_to_a_generic_weight(self):
        assert to_grams(1, "can", "dragonfruit").grams == 400.0


class TestUnconvertible:
    @pytest.mark.parametrize(
        ("quantity", "unit", "name"),
        [
            (None, "cup", "rice"),      # no quantity at all
            (0, "cup", "rice"),         # nonsense quantity
            (2, None, "saffron"),       # countable rule, uncountable thing
            (2, "furlong", "rice"),     # unknown unit
        ],
    )
    def test_returns_none_rather_than_zero(self, quantity, unit, name):
        """None means "cannot say"; zero would silently understate the total."""
        assert to_grams(quantity, unit, name) is None


class TestNutritionFactsAlgebra:
    @pytest.fixture
    def facts(self):
        return NutritionFacts(
            calories=200.0, protein_g=10.0, fat_g=8.0, carbs_g=20.0,
            fiber_g=2.0, sugar_g=5.0, sodium_mg=300.0,
        )

    def test_scaling_by_one_is_the_identity(self, facts):
        assert facts.scaled(1.0) == facts

    def test_scaling_by_zero_zeroes_everything(self, facts):
        assert facts.scaled(0.0) == NutritionFacts()

    def test_scaling_is_exact_for_dyadic_factors(self, facts):
        assert facts.scaled(2.0).calories == 400.0
        assert facts.scaled(0.5).calories == 100.0
        assert facts.scaled(0.25).protein_g == 2.5

    def test_scaling_round_trips(self, facts):
        assert facts.scaled(8.0).scaled(0.125) == facts

    def test_scaling_touches_every_field(self, facts):
        doubled = facts.scaled(2.0)
        for field, value in facts.as_dict().items():
            assert doubled.as_dict()[field] == value * 2

    def test_addition_is_field_wise(self, facts):
        assert (facts + facts) == facts.scaled(2.0)

    def test_adding_zero_is_the_identity(self, facts):
        assert facts + NutritionFacts() == facts

    def test_addition_is_commutative(self, facts):
        other = NutritionFacts(calories=50.0, protein_g=1.0)
        assert facts + other == other + facts

    def test_rounding_is_display_only(self, facts):
        assert facts.scaled(1 / 3).rounded(1).calories == 66.7

    def test_as_dict_covers_every_nutrient(self, facts):
        assert set(facts.as_dict()) == {
            "calories", "protein_g", "fat_g", "carbs_g",
            "fiber_g", "sugar_g", "sodium_mg",
        }


def _facts(calories, **kwargs):
    return NutritionFacts(calories=calories, **kwargs)


def _contribution(name, grams, facts):
    from agents.nutrition import IngredientNutrition

    return IngredientNutrition(
        name=name, grams=grams, facts=facts, source_description=f"{name}, raw",
        exact_mass=True, basis="mass",
    )


class TestNutritionReport:
    def test_total_is_the_sum_of_contributions(self):
        report = NutritionReport(
            servings=4,
            contributions=(
                _contribution("a", 100, _facts(200.0, protein_g=10.0)),
                _contribution("b", 50, _facts(150.0, protein_g=5.0)),
            ),
        )
        assert report.total.calories == 350.0
        assert report.total.protein_g == 15.0

    def test_per_serving_divides_the_total(self):
        report = NutritionReport(
            servings=4, contributions=(_contribution("a", 100, _facts(800.0)),)
        )
        assert report.per_serving.calories == 200.0

    def test_per_serving_times_servings_returns_the_total(self):
        report = NutritionReport(
            servings=8, contributions=(_contribution("a", 100, _facts(1000.0)),)
        )
        assert report.per_serving.scaled(8).calories == pytest.approx(1000.0)

    def test_one_serving_means_per_serving_equals_total(self):
        report = NutritionReport(
            servings=1, contributions=(_contribution("a", 100, _facts(640.0)),)
        )
        assert report.per_serving == report.total

    def test_empty_report_totals_zero(self):
        assert NutritionReport(servings=4).total == NutritionFacts()

    def test_coverage_reflects_what_could_be_measured(self):
        report = NutritionReport(
            servings=2,
            contributions=(_contribution("a", 100, _facts(100.0)),),
            unestimated=(("b", "no USDA match"), ("c", "unknown unit")),
        )
        assert report.coverage == pytest.approx(1 / 3)

    def test_coverage_of_an_empty_report_is_one(self):
        assert NutritionReport(servings=4).coverage == 1.0


class _FakeResponse:
    def __init__(self, payload, status=200):
        self._payload, self.status = payload, status

    def raise_for_status(self):
        if self.status >= 400:
            raise RuntimeError(f"HTTP {self.status}")

    def json(self):
        return self._payload


class _FakeSession:
    """Stands in for requests.Session, recording what was asked for."""

    def __init__(self, payload=None, status=200, error=None):
        self.payload, self.status, self.error = payload or {}, status, error
        self.calls = []

    def get(self, url, params=None, timeout=None):
        self.calls.append(params)
        if self.error:
            raise self.error
        return _FakeResponse(self.payload, self.status)


def _food(description, nutrients):
    """Build a USDA-search-shaped record."""
    return {
        "description": description,
        "foodNutrients": [
            {"nutrientId": nid, "value": value} for nid, value in nutrients.items()
        ],
    }


ONION = _food("Onions, raw", {1008: 40.0, 1003: 1.1, 1004: 0.1, 1005: 9.34, 1093: 4.0})


class TestFoodDataCentralClient:
    def test_returns_facts_and_the_matched_description(self):
        session = _FakeSession({"foods": [ONION]})
        client = FoodDataCentralClient(api_key="k", session=session)
        facts, description = client.lookup("onion")
        assert facts.calories == 40.0
        assert facts.protein_g == 1.1
        assert description == "Onions, raw"

    def test_energy_falls_back_to_the_atwater_ids(self):
        """Foundation foods report energy as 2047/2048, not 1008."""
        session = _FakeSession(
            {"foods": [_food("Chicken, raw", {2047: 106.0, 2048: 112.0})]}
        )
        client = FoodDataCentralClient(api_key="k", session=session)
        assert client.lookup("chicken")[0].calories == 112.0

    def test_entries_without_any_energy_are_unusable(self):
        session = _FakeSession({"foods": [_food("Oil, olive", {1003: 0.0})]})
        client = FoodDataCentralClient(api_key="k", session=session)
        assert client.lookup("olive oil") is None

    def test_picks_the_best_ranked_candidate_not_the_first(self):
        session = _FakeSession({
            "foods": [
                _food("DENNY'S, onion rings", {1008: 385.0}),
                _food("Onions, raw", {1008: 40.0}),
            ]
        })
        client = FoodDataCentralClient(api_key="k", session=session)
        assert client.lookup("onion")[1] == "Onions, raw"

    def test_no_results_is_a_definitive_miss(self):
        client = FoodDataCentralClient(api_key="k", session=_FakeSession({"foods": []}))
        assert client.lookup("dragonfruit") is None

    def test_transient_failures_raise_rather_than_report_a_miss(self):
        """A rate limit is not an absence of data."""
        session = _FakeSession(status=429)
        client = FoodDataCentralClient(api_key="k", session=session)
        with pytest.raises(NutritionLookupError):
            client.lookup("onion")

    def test_network_errors_raise(self):
        session = _FakeSession(error=OSError("connection reset"))
        client = FoodDataCentralClient(api_key="k", session=session)
        with pytest.raises(NutritionLookupError):
            client.lookup("onion")

    def test_results_are_cached_in_memory(self):
        session = _FakeSession({"foods": [ONION]})
        client = FoodDataCentralClient(api_key="k", session=session)
        client.lookup("onion")
        client.lookup("onion")
        assert len(session.calls) == 1

    def test_definitive_misses_are_cached_too(self):
        session = _FakeSession({"foods": []})
        client = FoodDataCentralClient(api_key="k", session=session)
        client.lookup("dragonfruit")
        client.lookup("dragonfruit")
        assert len(session.calls) == 1

    def test_transient_failures_are_never_cached(self):
        """Caching a 429 would make the outage permanent."""
        session = _FakeSession(status=500)
        client = FoodDataCentralClient(api_key="k", session=session)
        for _ in range(2):
            with pytest.raises(NutritionLookupError):
                client.lookup("onion")
        assert len(session.calls) == 2

    def test_cache_persists_to_disk_and_is_reused(self, tmp_path):
        cache = tmp_path / "nutrition.json"
        session = _FakeSession({"foods": [ONION]})
        FoodDataCentralClient(api_key="k", session=session, cache_path=cache).lookup(
            "onion"
        )
        assert cache.exists()

        second = _FakeSession({"foods": [ONION]})
        client = FoodDataCentralClient(api_key="k", session=second, cache_path=cache)
        assert client.lookup("onion")[1] == "Onions, raw"
        assert second.calls == []

    def test_a_corrupt_cache_is_ignored_not_fatal(self, tmp_path):
        cache = tmp_path / "nutrition.json"
        cache.write_text("{ not json")
        session = _FakeSession({"foods": [ONION]})
        client = FoodDataCentralClient(api_key="k", session=session, cache_path=cache)
        assert client.lookup("onion")[0].calories == 40.0

    def test_lookup_key_is_canonicalized(self):
        """"Chopped Onions" and "onion" must not be two cache entries."""
        session = _FakeSession({"foods": [ONION]})
        client = FoodDataCentralClient(api_key="k", session=session)
        client.lookup("onion")
        client.lookup("Chopped Onions")
        assert len(session.calls) == 1

    def test_the_api_key_is_sent_but_never_stored_in_the_cache(self, tmp_path):
        cache = tmp_path / "nutrition.json"
        session = _FakeSession({"foods": [ONION]})
        client = FoodDataCentralClient(
            api_key="secret-key", session=session, cache_path=cache
        )
        client.lookup("onion")
        assert session.calls[0]["api_key"] == "secret-key"
        assert "secret-key" not in json.dumps(json.loads(cache.read_text()))


class TestRankScore:
    @pytest.mark.parametrize(
        ("query", "better", "worse"),
        [
            (
                "olive oil",
                "Oil, olive, salad or cooking",
                "Oil, corn, peanut, and olive",
            ),
            ("onion", "Onions, raw", "DENNY'S, onion rings"),
            (
                "chicken breast",
                "Chicken, breast, boneless, skinless, raw",
                "Chicken breast tenders, breaded, uncooked",
            ),
            ("tomato", "Tomatoes, red, ripe, raw", "Tomato products, canned, paste"),
        ],
    )
    def test_prefers_the_raw_ingredient_over_the_prepared_product(
        self, query, better, worse
    ):
        assert _rank_score(query, better) < _rank_score(query, worse)

    def test_missing_query_terms_are_heavily_penalised(self):
        assert _rank_score("onion", "Garlic, raw") > _rank_score("onion", "Onions, raw")


class _StubClient:
    """Returns fixed per-100g facts for known names."""

    def __init__(self, table=None, error=None):
        self.table = table or {"onion": _facts(40.0, protein_g=1.1)}
        self.error = error

    def lookup(self, name):
        if self.error:
            raise self.error
        facts = self.table.get(name)
        return None if facts is None else (facts, f"{name}, raw")


class TestNutritionAgent:
    def test_computes_a_contribution_per_ingredient(self):
        agent = NutritionAgent(_StubClient())
        report = agent.analyze([Ingredient(name="onion", quantity=200, unit="g")])
        assert len(report.contributions) == 1
        assert report.contributions[0].grams == 200.0
        assert report.contributions[0].facts.calories == 80.0  # 40 per 100g

    def test_scales_facts_by_mass_exactly(self):
        agent = NutritionAgent(_StubClient())
        report = agent.analyze([Ingredient(name="onion", quantity=50, unit="g")])
        assert report.total.calories == 20.0

    def test_totals_across_ingredients(self):
        agent = NutritionAgent(
            _StubClient({"onion": _facts(40.0), "rice": _facts(360.0)})
        )
        report = agent.analyze([
            Ingredient(name="onion", quantity=100, unit="g"),
            Ingredient(name="rice", quantity=100, unit="g"),
        ])
        assert report.total.calories == 400.0

    def test_ingredients_without_a_usda_match_are_reported_not_dropped(self):
        agent = NutritionAgent(_StubClient())
        report = agent.analyze([
            Ingredient(name="onion", quantity=100, unit="g"),
            Ingredient(name="dragonfruit", quantity=100, unit="g"),
        ])
        assert report.unestimated == (("dragonfruit", "no USDA match"),)
        assert report.coverage == 0.5

    def test_unconvertible_quantities_are_reported(self):
        agent = NutritionAgent(_StubClient())
        report = agent.analyze([Ingredient(name="onion", quantity=None)])
        assert report.unestimated[0][1] == "no quantity or unknown unit"

    def test_a_rate_limited_lookup_is_reported_distinctly(self):
        """The user must be able to tell an outage from a genuine gap."""
        agent = NutritionAgent(_StubClient(error=NutritionLookupError("429")))
        report = agent.analyze([Ingredient(name="onion", quantity=100, unit="g")])
        assert report.unestimated == (("onion", "lookup unavailable"),)

    def test_without_a_client_nothing_is_estimated(self):
        report = NutritionAgent().analyze(
            [Ingredient(name="onion", quantity=1, unit="g")]
        )
        assert report.total == NutritionFacts()
        assert report.unestimated[0][1] == "no nutrition source configured"

    @pytest.mark.parametrize("servings", [0, -1])
    def test_servings_must_be_positive(self, servings):
        with pytest.raises(ValueError, match="servings"):
            NutritionAgent().analyze([], servings=servings)

    def test_doubling_every_quantity_doubles_the_total_exactly(self):
        agent = NutritionAgent(_StubClient())
        base = [Ingredient(name="onion", quantity=100, unit="g")]
        single = agent.analyze(base).total
        double = agent.analyze(scale_ingredients(base, 2.0)).total
        assert double.calories == single.calories * 2

    def test_rescaling_servings_does_not_recompute_the_total(self):
        agent = NutritionAgent(_StubClient())
        report = agent.analyze(
            [Ingredient(name="onion", quantity=400, unit="g")], servings=4
        )
        rescaled = agent.scale_to_servings(report, 8)
        assert rescaled.total == report.total
        assert rescaled.per_serving.calories == report.per_serving.calories / 2

    def test_composes_with_the_ingredient_agent(self):
        agent = NutritionAgent(
            _StubClient({"onion": _facts(40.0), "egg": _facts(143.0)})
        )
        report = agent.analyze(parse_ingredients("2 onions and 3 eggs"), servings=2)
        assert {c.name for c in report.contributions} == {"onion", "egg"}
        assert report.total.calories == pytest.approx(88.0 + 214.5)


class TestScaleIngredients:
    def test_scales_quantities(self):
        scaled = scale_ingredients([Ingredient(name="rice", quantity=2, unit="cup")], 3)
        assert scaled[0].quantity == 6.0

    def test_vague_quantities_are_left_alone(self):
        """Nothing to scale, and inventing a number would be worse."""
        scaled = scale_ingredients([Ingredient(name="salt", quantity=None)], 2)
        assert scaled[0].quantity is None

    def test_original_is_not_mutated(self):
        original = [Ingredient(name="rice", quantity=2, unit="cup")]
        scale_ingredients(original, 3)
        assert original[0].quantity == 2.0

    @pytest.mark.parametrize("factor", [0, -1])
    def test_factor_must_be_positive(self, factor):
        with pytest.raises(ValueError, match="factor"):
            scale_ingredients([], factor)


class TestSearchAliases:
    """Guards for the lookup failures found by testing against the live API.

    Searching USDA for "rice" returns ten rice crackers before any grain,
    "milk" returns milk crackers and milk chocolate, and "egg" returns egg
    bagels. Re-ranking cannot fix a result set that never contained the food,
    so ambiguous names get a precise query instead.
    """

    def test_alias_keys_are_canonical(self):
        """A non-canonical key could never be looked up."""
        from agents.normalization import canonicalize

        offenders = {k: canonicalize(k) for k in SEARCH_ALIASES if canonicalize(k) != k}
        assert offenders == {}

    def test_no_alias_is_a_no_op(self):
        assert all(key != value for key, value in SEARCH_ALIASES.items())

    def test_the_alias_is_what_gets_sent_to_usda(self):
        session = _FakeSession({"foods": [ONION]})
        client = FoodDataCentralClient(api_key="k", session=session)
        client.lookup("egg")
        assert session.calls[0]["query"] == SEARCH_ALIASES["egg"]

    def test_names_without_an_alias_are_searched_as_is(self):
        session = _FakeSession({"foods": [ONION]})
        client = FoodDataCentralClient(api_key="k", session=session)
        client.lookup("onion")
        assert session.calls[0]["query"] == "onion"

    def test_the_cache_key_is_the_ingredient_not_the_alias(self):
        """Two ingredients sharing an alias must not collide in the cache."""
        session = _FakeSession({"foods": [ONION]})
        client = FoodDataCentralClient(api_key="k", session=session)
        client.lookup("egg")
        client.lookup("egg")
        assert len(session.calls) == 1


class TestRankingAgainstRealDescriptions:
    """Every pair here is verbatim from a live USDA response.

    The left-hand entry is the food a recipe means; the right-hand one is what
    the original ranking actually picked, which would have reported egg bagels
    (278 kcal) as the nutrition for an egg (143 kcal).
    """

    @pytest.mark.parametrize(
        ("query", "wanted", "wrong"),
        [
            ("egg", "Egg, whole, raw, fresh", "Bagels, egg"),
            ("egg", "Egg, whole, raw, fresh", "Bread, egg"),
            ("milk", "Milk, whole, 3.25% milkfat", "Crackers, milk"),
            ("milk", "Milk, whole, 3.25% milkfat", "Candies, milk chocolate"),
            ("beef", "Beef, tenderloin steak, raw", "Bologna, beef"),
            (
                "rice",
                "Rice, white, long-grain, regular, raw, unenriched",
                "Rice crackers",
            ),
            ("sugar", "Sugars, granulated", "Cookies, sugar, refrigerated dough"),
            ("onion", "Onions, raw", "DENNY'S, onion rings"),
            (
                "tomato",
                "Tomatoes, red, ripe, raw, year round average",
                "Tomato powder",
            ),
        ],
    )
    def test_the_food_beats_the_product_made_from_it(self, query, wanted, wrong):
        assert _rank_score(query, wanted) < _rank_score(query, wrong)

    def test_a_description_led_by_the_query_wins(self):
        """"Onions, raw" answers "onion"; "Bologna, beef" does not answer "beef"."""
        assert _rank_score("beef", "Beef, ground, raw") < _rank_score(
            "beef", "Bologna, beef"
        )
