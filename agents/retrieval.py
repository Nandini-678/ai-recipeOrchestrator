"""Retrieval agent: pantry in, ranked recipe matches out.

Two stages, deliberately split:

1. **Recall** -- a Chroma vector search over the recipe corpus narrows 790
   recipes to a candidate pool. This is the RAG layer, and it is fuzzy by
   design: it catches recipes whose wording differs from the pantry's.
2. **Ranking** -- a pure, deterministic overlap score orders that pool. No
   embedding distance reaches the user-visible ranking, because "you have 4 of
   these 6 ingredients" is a claim the critic agent can verify and the UI can
   explain, and a cosine distance is neither.

The index is optional. Without one the agent scores the whole corpus exhaustively
-- exact, fast enough at this size, and the path the offline tests exercise.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from agents.ingredient import Ingredient
from agents.recipe import Recipe

#: Ingredients almost every kitchen has. Counting these as "missing" would drag
#: every recipe's coverage down uniformly and bury genuinely good matches --
#: salt alone appears in 242 of the 790 corpus recipes. They are excluded from
#: the score but still reported, so the composer can list them.
PANTRY_STAPLES: frozenset[str] = frozenset({
    "salt", "black pepper", "pepper", "water", "olive oil", "vegetable oil",
    "sugar", "flour", "all-purpose flour", "butter", "oil",
})


@dataclass(frozen=True)
class RecipeMatch:
    """One scored recipe, with the evidence behind its score.

    Attributes:
        recipe: The matched recipe.
        score: Fraction of the recipe's non-staple ingredients the pantry
            covers, in ``[0, 1]``.
        matched: Pantry ingredients this recipe uses, sorted.
        missing: Non-staple ingredients the pantry lacks, sorted.
        staples_assumed: Staples the recipe needs that the pantry did not list,
            excluded from the score but surfaced for the composer.
    """

    recipe: Recipe
    score: float
    matched: tuple[str, ...] = field(default=())
    missing: tuple[str, ...] = field(default=())
    staples_assumed: tuple[str, ...] = field(default=())

    @property
    def is_complete(self) -> bool:
        """True when the pantry covers every non-staple ingredient."""
        return not self.missing


def pantry_names(ingredients: Iterable[Ingredient | str]) -> set[str]:
    """Reduce parsed ingredients (or bare names) to a set of canonical names."""
    return {
        item.name if isinstance(item, Ingredient) else str(item)
        for item in ingredients
    }


def score_overlap(
    pantry: set[str], recipe: Recipe, *, assume_staples: bool = True
) -> RecipeMatch:
    """Score how well ``pantry`` covers ``recipe``.

    Args:
        pantry: Canonical ingredient names the user has.
        recipe: The recipe to score.
        assume_staples: When true, :data:`PANTRY_STAPLES` the user did not list
            are assumed present and excluded from the denominator.

    Returns:
        A :class:`RecipeMatch` whose ``score`` is the fraction of countable
        ingredients covered. A recipe made entirely of staples scores ``1.0``.
    """
    required = recipe.ingredient_names
    staples = (required & PANTRY_STAPLES) - pantry if assume_staples else set()
    countable = required - staples

    matched = countable & pantry
    missing = countable - pantry
    score = len(matched) / len(countable) if countable else 1.0

    return RecipeMatch(
        recipe=recipe,
        score=score,
        matched=tuple(sorted(matched)),
        missing=tuple(sorted(missing)),
        staples_assumed=tuple(sorted(staples)),
    )


#: Pseudo-count added to the denominator when ranking. It leaves a recipe with
#: many ingredients essentially unchanged while heavily discounting one with
#: only a couple, which is what stops a trivially-covered recipe from winning.
_RANK_SMOOTHING = 1.0


def adjusted_coverage(match: RecipeMatch) -> float:
    """Coverage discounted for how little the recipe actually asks for.

    Raw coverage is the honest number to *show* a user -- "you have 4 of these
    6 ingredients" -- but it is a poor thing to *rank* by, because a recipe
    made almost entirely of pantry staples has one countable ingredient and
    scores a perfect 1.0 off a single match. Ranked that way, a flatbread of
    flour, water, salt and oil beats a chicken dish using five of the six
    things you actually have.

    Adding one to the denominator fixes the ordering without touching the
    reported score: 1 of 1 becomes 0.50, while 5 of 6 becomes 0.71.
    """
    countable = len(match.matched) + len(match.missing)
    if not countable:
        return 0.0
    return len(match.matched) / (countable + _RANK_SMOOTHING)


def _rank_key(match: RecipeMatch) -> tuple:
    """Sort key: adjusted coverage, then most used, then simplest, then title.

    Title last keeps the ordering stable and reproducible in tests.
    """
    return (
        -adjusted_coverage(match),
        -len(match.matched),
        len(match.recipe.ingredients),
        match.recipe.title,
    )


class RecipeIndex:
    """Chroma-backed semantic recall over the recipe corpus."""

    def __init__(self, collection) -> None:
        """Wrap an existing Chroma collection."""
        self._collection = collection

    @classmethod
    def build(
        cls,
        recipes: Sequence[Recipe],
        *,
        client=None,
        embedding_function=None,
        collection_name: str = "recipes",
        persist: bool = False,
    ) -> RecipeIndex:
        """Embed ``recipes`` into a fresh Chroma collection.

        Args:
            recipes: Corpus to index.
            client: Chroma client. Defaults to an in-memory client, or a
                persistent one under ``data/chroma`` when ``persist`` is set.
            embedding_function: Override the embedder. Chroma's default
                downloads a model on first use, so tests inject their own.
            collection_name: Collection to create, replacing any existing one.
            persist: Store the index on disk instead of in memory.

        Returns:
            The populated index.
        """
        import chromadb

        if client is None:
            if persist:
                from config import CHROMA_DIR

                CHROMA_DIR.mkdir(parents=True, exist_ok=True)
                client = chromadb.PersistentClient(path=str(CHROMA_DIR))
            else:
                client = chromadb.EphemeralClient()

        try:
            client.delete_collection(collection_name)
        except Exception:  # noqa: BLE001 - absent collection is the normal case
            pass

        kwargs = {"name": collection_name}
        if embedding_function is not None:
            kwargs["embedding_function"] = embedding_function
        collection = client.create_collection(**kwargs)

        collection.add(
            ids=[recipe.id for recipe in recipes],
            documents=[recipe.searchable_text() for recipe in recipes],
            metadatas=[{"title": recipe.title} for recipe in recipes],
        )
        return cls(collection)

    def candidates(self, pantry: set[str], n_results: int) -> list[str]:
        """Return recipe ids most semantically similar to the pantry."""
        if not pantry:
            return []
        result = self._collection.query(
            query_texts=[" ".join(sorted(pantry))],
            n_results=min(n_results, max(self._collection.count(), 1)),
        )
        ids = result.get("ids") or [[]]
        return list(ids[0])


class RetrievalAgent:
    """Rank corpus recipes by how well a pantry covers them."""

    def __init__(
        self, recipes: Sequence[Recipe], index: RecipeIndex | None = None
    ) -> None:
        """Build the agent over a corpus.

        Args:
            recipes: The recipe corpus.
            index: Optional Chroma recall stage. Without it the agent scores
                every recipe exhaustively: exact, but linear in corpus size.
        """
        self._recipes = {recipe.id: recipe for recipe in recipes}
        self._index = index

    def __len__(self) -> int:
        """Number of recipes in the corpus."""
        return len(self._recipes)

    def _candidate_recipes(self, pantry: set[str], pool: int) -> list[Recipe]:
        """Narrow the corpus to a candidate pool, via the index when present."""
        if self._index is None:
            return list(self._recipes.values())
        ids = self._index.candidates(pantry, pool)
        found = [self._recipes[i] for i in ids if i in self._recipes]
        return found or list(self._recipes.values())

    def retrieve(
        self,
        ingredients: Iterable[Ingredient | str],
        *,
        top_n: int = 5,
        candidate_pool: int = 60,
        assume_staples: bool = True,
        min_coverage: float = 0.0,
    ) -> list[RecipeMatch]:
        """Return the ``top_n`` best-covered recipes for a pantry.

        Args:
            ingredients: Parsed ingredients (or canonical names) the user has.
            top_n: Maximum matches to return.
            candidate_pool: How many recipes the index stage recalls before
                ranking. Ignored when no index is attached.
            assume_staples: Passed through to :func:`score_overlap`.
            min_coverage: Drop matches scoring below this threshold.

        Returns:
            Matches sorted best-first. Empty when the pantry is empty, since
            every recipe would otherwise tie at zero coverage.
        """
        pantry = pantry_names(ingredients)
        if not pantry:
            return []

        candidates = self._candidate_recipes(pantry, candidate_pool)
        scored = [
            score_overlap(pantry, recipe, assume_staples=assume_staples)
            for recipe in candidates
        ]
        keep = [m for m in scored if m.score >= min_coverage and m.matched]
        keep.sort(key=_rank_key)
        return keep[:top_n]
