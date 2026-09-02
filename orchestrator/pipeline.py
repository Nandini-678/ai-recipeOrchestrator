"""The orchestrator: plain sequential Python, no agent framework.

The pipeline is a straight line -- ingredients, retrieval, safety, nutrition,
composition, criticism -- and expressing it as a function that calls six other
functions in order is both the simplest thing that works and the easiest thing
to debug. A graph library would add a scheduler, a state object and a new
vocabulary to describe control flow that a reader can already follow here.

Two loops sit inside that line:

* **Retry.** When the critic rejects a composition, the composer is called
  again with the critic's specific complaints. Bounded by ``max_attempts``.
* **Fallback.** When a candidate recipe cannot be composed acceptably at all,
  the next-best candidate is tried. Bounded by ``max_candidates``.

Every stage is injectable, so a caller can swap in a stub and the whole
pipeline runs offline.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from agents.composer import ComposedRecipe, ComposerAgent
from agents.critic import CriticAgent, Verdict
from agents.ingredient import Extractor, Ingredient, parse_ingredients
from agents.nutrition import NutritionAgent, NutritionReport
from agents.recipe import Recipe
from agents.retrieval import RetrievalAgent
from agents.safety import SafetyAgent, apply_substitutions


@dataclass(frozen=True)
class Attempt:
    """One composition attempt and what the critic made of it."""

    recipe: ComposedRecipe
    verdict: Verdict
    candidate_title: str


@dataclass(frozen=True)
class PipelineResult:
    """Everything the pipeline produced, accepted or not.

    Attributes:
        recipe: The accepted recipe, or the best effort when nothing passed.
        verdict: The critic's verdict on ``recipe``.
        accepted: Whether ``recipe`` passed the critic.
        pantry: The parsed ingredients the run was based on.
        nutrition: The nutrition report behind ``recipe``.
        candidates_retrieved: How many recipes retrieval returned.
        candidates_safe: How many survived allergen screening.
        attempts: Every composition attempt made, in order.
        reason: Why there is no recipe, when there is none.
    """

    recipe: ComposedRecipe | None = None
    verdict: Verdict = field(default_factory=Verdict)
    accepted: bool = False
    pantry: tuple[Ingredient, ...] = ()
    nutrition: NutritionReport | None = None
    candidates_retrieved: int = 0
    candidates_safe: int = 0
    attempts: tuple[Attempt, ...] = ()
    reason: str = ""

    @property
    def attempt_count(self) -> int:
        """How many times the composer was called."""
        return len(self.attempts)


class RecipeOrchestrator:
    """Runs the full pipeline for one request."""

    def __init__(
        self,
        recipes: Sequence[Recipe],
        *,
        retrieval: RetrievalAgent | None = None,
        safety: SafetyAgent | None = None,
        nutrition: NutritionAgent | None = None,
        composer: ComposerAgent | None = None,
        critic: CriticAgent | None = None,
        extractor: Extractor | None = None,
        max_attempts: int = 3,
        max_candidates: int = 3,
    ) -> None:
        """Wire the pipeline.

        Args:
            recipes: The recipe corpus.
            retrieval: Retrieval agent. Built over ``recipes`` if omitted.
            safety: Safety agent. A default one is built if omitted.
            nutrition: Nutrition agent. Without a configured client it will
                report everything as unestimated rather than fail.
            composer: Composer agent. Without a client it uses source steps.
            critic: Critic agent. A default one is built if omitted.
            extractor: Optional LLM segmentation for the ingredient agent.
            max_attempts: Composition attempts per candidate, at least 1.
            max_candidates: How many retrieved recipes to try before giving up.

        Raises:
            ValueError: If either bound is less than 1.
        """
        if max_attempts < 1 or max_candidates < 1:
            raise ValueError("max_attempts and max_candidates must be at least 1")

        self._retrieval = retrieval or RetrievalAgent(recipes)
        self._safety = safety or SafetyAgent()
        self._nutrition = nutrition or NutritionAgent()
        self._composer = composer or ComposerAgent()
        self._critic = critic or CriticAgent()
        self._extractor = extractor
        self._max_attempts = max_attempts
        self._max_candidates = max_candidates

    def run(
        self,
        text: str,
        *,
        avoid: Iterable[str] = (),
        servings: int = 4,
        top_n: int = 10,
        max_missing: int | None = None,
    ) -> PipelineResult:
        """Turn raw pantry text into a validated recipe.

        Args:
            text: Whatever the user typed.
            avoid: Allergens to exclude, in any accepted spelling.
            servings: Servings to scale nutrition to.
            top_n: How many recipes retrieval should rank before screening.
            max_missing: Most ingredients the user is willing to go and buy.
                ``0`` restricts results to what is cookable right now.

        Returns:
            A :class:`PipelineResult`. ``accepted`` says whether the critic
            passed the recipe; when nothing could be produced, ``reason``
            explains why and ``recipe`` is ``None``.
        """
        avoid = tuple(avoid)
        pantry = parse_ingredients(text, extractor=self._extractor)
        if not pantry:
            return PipelineResult(
                reason="No ingredients could be read from that input."
            )

        pantry_names = {item.name for item in pantry}
        matches = self._retrieval.retrieve(
            pantry, top_n=top_n, max_missing=max_missing
        )
        if not matches:
            if max_missing is not None:
                reason = (
                    f"Nothing can be made with at most {max_missing} extra "
                    "ingredient(s). Allow a couple more and try again."
                )
            else:
                reason = "No recipe in the corpus uses any of those ingredients."
            return PipelineResult(pantry=tuple(pantry), reason=reason)

        screened = self._safety.screen(matches, pantry=pantry_names, avoid=avoid)
        if not screened:
            return PipelineResult(
                pantry=tuple(pantry),
                candidates_retrieved=len(matches),
                reason=(
                    f"All {len(matches)} matching recipes contain "
                    f"{', '.join(avoid)}."
                ),
            )

        attempts: list[Attempt] = []
        best: Attempt | None = None
        best_nutrition: NutritionReport | None = None

        for candidate in screened[: self._max_candidates]:
            # Nutrition must describe the list the user will actually see,
            # so it follows the composer's decision about substitutions.
            ingredients = (
                apply_substitutions(candidate)
                if self._composer.applies_substitutions
                else candidate.recipe.ingredients
            )
            nutrition = self._nutrition.analyze(ingredients, servings=servings)
            feedback: list[str] = []

            for _ in range(self._max_attempts):
                composed = self._composer.compose(
                    candidate, nutrition, pantry=pantry_names, feedback=feedback
                )
                verdict = self._critic.review(
                    composed,
                    screened=candidate,
                    avoid=avoid,
                    pantry=pantry_names,
                )
                attempt = Attempt(composed, verdict, candidate.recipe.title)
                attempts.append(attempt)
                best = best or attempt

                if verdict.ok:
                    return PipelineResult(
                        recipe=composed,
                        verdict=verdict,
                        accepted=True,
                        pantry=tuple(pantry),
                        nutrition=nutrition,
                        candidates_retrieved=len(matches),
                        candidates_safe=len(screened),
                        attempts=tuple(attempts),
                    )

                feedback = verdict.feedback
                # Retrying is only useful when the model wrote the steps; the
                # fallback path is deterministic and would return the same
                # thing every time.
                if composed.composed_by != "llm":
                    break

            best_nutrition = best_nutrition or nutrition

        return PipelineResult(
            recipe=best.recipe if best else None,
            verdict=best.verdict if best else Verdict(),
            accepted=False,
            pantry=tuple(pantry),
            nutrition=best_nutrition,
            candidates_retrieved=len(matches),
            candidates_safe=len(screened),
            attempts=tuple(attempts),
            reason="No candidate passed validation; returning the best attempt.",
        )
