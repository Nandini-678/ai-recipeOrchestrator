"""Shared fixtures: a small recipe corpus and an offline embedding function."""

import hashlib

import pytest

from agents.recipe import Recipe


# Chroma powers an optional recall stage, so it is an optional dependency.
# Importing it here unconditionally would make the entire suite uncollectable
# without it, which would misrepresent what the app actually requires.
def _embedding_base():
    """Return chroma's EmbeddingFunction base, or object if it is absent."""
    try:
        from chromadb.api.types import EmbeddingFunction
    except ImportError:
        return object
    return EmbeddingFunction


class HashingEmbeddingFunction(_embedding_base()):
    """Deterministic bag-of-words embedder for tests.

    Chroma's default embedder downloads a model on first use. This one hashes
    tokens into a fixed-width vector instead: no network, no model, and
    documents sharing tokens still land near each other, which is all the
    retrieval tests need from the recall stage.
    """

    def __init__(self, dimensions: int = 64) -> None:
        """Args: dimensions: Width of the produced vectors."""
        self._dimensions = dimensions

    def __call__(self, input) -> list[list[float]]:  # noqa: A002 - chroma API
        """Embed each document as a hashed token-count vector."""
        vectors = []
        for document in input:
            vector = [0.0] * self._dimensions
            for token in str(document).lower().split():
                digest = hashlib.md5(token.encode()).hexdigest()  # noqa: S324
                vector[int(digest, 16) % self._dimensions] += 1.0
            vectors.append(vector)
        return vectors

    @staticmethod
    def name() -> str:
        """Identifier Chroma stores alongside the collection."""
        return "hashing-test"

    def get_config(self) -> dict:
        """Config Chroma persists so the embedder can be rebuilt."""
        return {"dimensions": self._dimensions}

    @staticmethod
    def build_from_config(config: dict) -> "HashingEmbeddingFunction":
        """Reconstruct the embedder from a stored config."""
        return HashingEmbeddingFunction(dimensions=config["dimensions"])


def _recipe(id_: str, title: str, names: list[str], **kwargs) -> Recipe:
    """Build a Recipe from canonical ingredient names, for brevity in fixtures.

    Every ingredient gets a real quantity so the nutrition agent has something
    to measure; without one it correctly reports the whole recipe as
    unestimated, which makes pipeline tests assert nothing useful.
    """
    return Recipe(
        id=id_,
        title=title,
        ingredients=[{"name": name, "quantity": 100.0, "unit": "g"} for name in names],
        steps=["Do the thing.", "Serve."],
        **kwargs,
    )


@pytest.fixture
def sample_recipes() -> list[Recipe]:
    """A small, hand-built corpus with known overlap properties."""
    return [
        _recipe(
            "1", "Chicken Fried Rice",
            ["chicken breast", "rice", "egg", "green onion", "soy sauce",
             "garlic", "salt"],
            category="Chicken", area="Chinese",
        ),
        _recipe(
            "2", "Garlic Tomato Pasta",
            ["pasta", "tomato", "garlic", "olive oil", "basil", "salt"],
            category="Pasta", area="Italian",
        ),
        _recipe(
            "3", "Simple Omelette",
            ["egg", "butter", "salt"],
            category="Breakfast", area="French",
        ),
        _recipe(
            "4", "Chicken Tomato Curry",
            ["chicken breast", "tomato", "onion", "garlic", "ginger", "cumin",
             "turmeric", "coconut milk"],
            category="Chicken", area="Indian",
        ),
        _recipe(
            "5", "Chocolate Cake",
            ["chocolate", "sugar", "all-purpose flour", "egg", "butter"],
            category="Dessert", area="British",
        ),
        _recipe(
            "6", "Steamed Rice",
            ["rice", "water", "salt"],
            category="Side", area="Japanese",
        ),
    ]


@pytest.fixture
def embedding_function() -> HashingEmbeddingFunction:
    """Offline embedder for building test indexes."""
    pytest.importorskip("chromadb", reason="optional vector-index dependency")
    return HashingEmbeddingFunction()
