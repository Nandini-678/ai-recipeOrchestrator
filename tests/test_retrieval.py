"""Unit tests for the retrieval agent.

Runs entirely offline: the Chroma index is built in memory with the hashing
embedder from conftest, so no model is downloaded and no API key is needed.
"""

import pytest

from agents.ingredient import Ingredient, parse_ingredients
from agents.recipe import Recipe
from agents.retrieval import (
    PANTRY_STAPLES,
    RecipeIndex,
    RetrievalAgent,
    pantry_names,
    score_overlap,
)


class TestPantryNames:
    def test_accepts_parsed_ingredients(self):
        parsed = [Ingredient(name="rice"), Ingredient(name="egg")]
        assert pantry_names(parsed) == {"rice", "egg"}

    def test_accepts_bare_strings(self):
        assert pantry_names(["rice", "egg"]) == {"rice", "egg"}


class TestScoreOverlap:
    def test_full_coverage_scores_one(self, sample_recipes):
        omelette = sample_recipes[2]  # egg, butter, salt
        match = score_overlap({"egg", "butter", "salt"}, omelette)
        assert match.score == 1.0
        assert match.is_complete
        assert match.missing == ()

    def test_partial_coverage_is_a_fraction_of_countable_ingredients(self):
        recipe = Recipe(
            id="x",
            title="Test",
            ingredients=[{"name": n} for n in ["rice", "egg", "tofu", "leek"]],
        )
        match = score_overlap({"rice", "egg"}, recipe)
        assert match.score == 0.5
        assert match.matched == ("egg", "rice")
        assert match.missing == ("leek", "tofu")

    def test_zero_overlap_scores_zero(self, sample_recipes):
        assert score_overlap({"tofu"}, sample_recipes[4]).score == 0.0

    def test_staples_are_assumed_present_and_excluded_from_the_score(self):
        """Staples are excluded from the denominator.

        Salt appears in 242 of the 790 corpus recipes; counting it as missing
        would drag every recipe's score down uniformly.
        """
        recipe = Recipe(
            id="x", title="T", ingredients=[{"name": "egg"}, {"name": "salt"}]
        )
        match = score_overlap({"egg"}, recipe)
        assert match.score == 1.0
        assert match.staples_assumed == ("salt",)
        assert match.missing == ()

    def test_staples_the_user_did_list_are_scored_normally(self):
        recipe = Recipe(
            id="x", title="T", ingredients=[{"name": "egg"}, {"name": "salt"}]
        )
        match = score_overlap({"egg", "salt"}, recipe)
        assert match.staples_assumed == ()
        assert match.matched == ("egg", "salt")

    def test_staple_assumption_can_be_turned_off(self):
        recipe = Recipe(
            id="x", title="T", ingredients=[{"name": "egg"}, {"name": "salt"}]
        )
        match = score_overlap({"egg"}, recipe, assume_staples=False)
        assert match.score == 0.5
        assert match.missing == ("salt",)

    def test_a_recipe_of_only_staples_is_fully_covered(self):
        recipe = Recipe(id="x", title="T", ingredients=[{"name": "salt"}])
        assert score_overlap({"egg"}, recipe).score == 1.0

    def test_every_staple_is_a_canonical_name(self):
        """A staple spelled non-canonically would silently never match."""
        from agents.normalization import canonicalize

        assert all(canonicalize(name) == name for name in PANTRY_STAPLES)


class TestRankingWithoutIndex:
    def test_best_covered_recipe_comes_first(self, sample_recipes):
        agent = RetrievalAgent(sample_recipes)
        results = agent.retrieve(["egg", "butter"], top_n=3)
        assert results[0].recipe.title == "Simple Omelette"
        assert results[0].score == 1.0

    def test_absolute_matches_break_a_coverage_tie(self, sample_recipes):
        """Two recipes at equal coverage: the one using more pantry wins."""
        agent = RetrievalAgent(sample_recipes)
        results = agent.retrieve(
            ["chicken breast", "rice", "egg", "green onion", "soy sauce", "garlic"],
            top_n=2,
        )
        assert results[0].recipe.title == "Chicken Fried Rice"
        assert results[0].score == 1.0

    def test_results_are_sorted_by_score_descending(self, sample_recipes):
        agent = RetrievalAgent(sample_recipes)
        scores = [m.score for m in agent.retrieve(["egg", "tomato", "garlic"], top_n=6)]
        assert scores == sorted(scores, reverse=True)

    def test_top_n_is_respected(self, sample_recipes):
        agent = RetrievalAgent(sample_recipes)
        assert len(agent.retrieve(["egg", "garlic", "rice"], top_n=2)) == 2

    def test_recipes_with_no_overlap_are_excluded(self, sample_recipes):
        agent = RetrievalAgent(sample_recipes)
        titles = [m.recipe.title for m in agent.retrieve(["chocolate"], top_n=6)]
        assert titles == ["Chocolate Cake"]

    def test_min_coverage_filters_weak_matches(self, sample_recipes):
        agent = RetrievalAgent(sample_recipes)
        results = agent.retrieve(["egg", "garlic"], top_n=6, min_coverage=0.9)
        assert all(m.score >= 0.9 for m in results)

    def test_empty_pantry_returns_nothing(self, sample_recipes):
        agent = RetrievalAgent(sample_recipes)
        assert agent.retrieve([]) == []

    def test_ordering_is_deterministic_across_runs(self, sample_recipes):
        agent = RetrievalAgent(sample_recipes)
        first = [m.recipe.id for m in agent.retrieve(["egg", "salt"], top_n=6)]
        second = [m.recipe.id for m in agent.retrieve(["egg", "salt"], top_n=6)]
        assert first == second

    def test_accepts_output_of_the_ingredient_agent_directly(self, sample_recipes):
        """The two agents must compose without any glue."""
        agent = RetrievalAgent(sample_recipes)
        pantry = parse_ingredients("3 eggs, a knob of butter and some salt")
        assert agent.retrieve(pantry, top_n=1)[0].recipe.title == "Simple Omelette"

    def test_reports_the_evidence_behind_a_score(self, sample_recipes):
        agent = RetrievalAgent(sample_recipes)
        match = agent.retrieve(["chicken breast", "tomato", "garlic"], top_n=1)[0]
        assert set(match.matched) <= {"chicken breast", "tomato", "garlic"}
        assert "chicken breast" not in match.missing


class TestRecipeIndex:
    def test_builds_and_recalls_candidates(self, sample_recipes, embedding_function):
        index = RecipeIndex.build(
            sample_recipes, embedding_function=embedding_function
        )
        ids = index.candidates({"chicken breast", "rice"}, n_results=3)
        assert len(ids) == 3
        assert all(i in {r.id for r in sample_recipes} for i in ids)

    def test_empty_pantry_recalls_nothing(self, sample_recipes, embedding_function):
        index = RecipeIndex.build(
            sample_recipes, embedding_function=embedding_function
        )
        assert index.candidates(set(), n_results=3) == []

    def test_n_results_is_clamped_to_corpus_size(
        self, sample_recipes, embedding_function
    ):
        index = RecipeIndex.build(
            sample_recipes, embedding_function=embedding_function
        )
        assert len(index.candidates({"rice"}, n_results=999)) == len(sample_recipes)

    def test_rebuilding_replaces_rather_than_duplicates(
        self, sample_recipes, embedding_function
    ):
        import chromadb

        client = chromadb.EphemeralClient()
        for _ in range(2):
            index = RecipeIndex.build(
                sample_recipes, client=client, embedding_function=embedding_function
            )
        assert len(index.candidates({"rice"}, n_results=999)) == len(sample_recipes)


class TestRetrievalWithIndex:
    @pytest.fixture
    def indexed_agent(self, sample_recipes, embedding_function):
        index = RecipeIndex.build(
            sample_recipes, embedding_function=embedding_function
        )
        return RetrievalAgent(sample_recipes, index=index)

    def test_ranking_still_comes_from_overlap_not_embedding_distance(
        self, indexed_agent
    ):
        """The index only narrows; the visible order must stay explainable."""
        results = indexed_agent.retrieve(["egg", "butter"], top_n=3, candidate_pool=6)
        assert results[0].recipe.title == "Simple Omelette"
        assert results[0].score == 1.0

    def test_falls_back_to_the_full_corpus_when_recall_finds_nothing(
        self, sample_recipes
    ):
        class _EmptyIndex:
            def candidates(self, pantry, n_results):
                return []

        agent = RetrievalAgent(sample_recipes, index=_EmptyIndex())
        assert agent.retrieve(["chocolate"], top_n=1)[0].recipe.title == (
            "Chocolate Cake"
        )

    def test_unknown_ids_from_the_index_are_ignored(self, sample_recipes):
        class _StaleIndex:
            def candidates(self, pantry, n_results):
                return ["999", "3"]

        agent = RetrievalAgent(sample_recipes, index=_StaleIndex())
        assert agent.retrieve(["egg"], top_n=2)[0].recipe.id == "3"
