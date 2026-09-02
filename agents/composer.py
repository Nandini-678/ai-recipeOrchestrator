"""Composer agent: assembles the final recipe the user actually reads.

The split of labour here follows the rest of the pipeline. Steps 2-5 already
produced exact quantities, exact overlap flags and exact nutrition. Asking a
language model to *restate* those would throw away that exactness and invite
hallucination in the one place the user is most likely to trust the output.

So the model writes only prose -- a title, a one-line summary, cooking times,
and the numbered steps rewritten to account for substitutions. Everything
factual is assembled by :class:`ComposerAgent` in code:

* the ingredient list, with have/missing flags taken from the retrieval match
* substitutions and their scaled quantities, from the safety agent
* the nutrition summary, from the nutrition agent
* warnings about anything the pipeline could not resolve

Every model response is validated against :class:`_ComposerResponse`, which
forbids unexpected fields and rejects empty or malformed steps. If validation
fails, or no model is configured, the agent falls back to the source recipe's
own instructions rather than failing the request -- a slightly duller recipe
beats no recipe.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from agents.nutrition import NutritionReport
from agents.safety import ScreenedMatch, Substitution

# --- Output contract ---------------------------------------------------------


class ComposedIngredient(BaseModel):
    """One line of the final recipe's ingredient list.

    Attributes:
        name: Canonical ingredient name, after any substitution.
        quantity: Amount, scaled by the substitution ratio where one applied.
        unit: Canonical unit, or ``None`` for countable items.
        have: Whether the user's pantry covers this ingredient.
        assumed_staple: True for pantry staples assumed present but not listed.
        substituted_for: The ingredient this replaced, when it is a substitute.
        note: Short caveat carried over from the substitution table.
    """

    name: str = Field(min_length=1)
    quantity: float | None = None
    unit: str | None = None
    have: bool = False
    assumed_staple: bool = False
    substituted_for: str | None = None
    note: str = ""


class NutritionSummary(BaseModel):
    """Per-serving nutrition, rounded for display only.

    Attributes:
        complete: False when some ingredients could not be measured, in which
            case these numbers understate the real total.
        measured_ingredients: How many of the recipe's ingredients are counted.
        total_ingredients: How many there are in all.
    """

    calories: float
    protein_g: float
    fat_g: float
    carbs_g: float
    fiber_g: float
    sugar_g: float
    sodium_mg: float
    complete: bool = True
    measured_ingredients: int = 0
    total_ingredients: int = 0


class ComposedRecipe(BaseModel):
    """The finished recipe, ready to render or hand to the critic.

    Attributes:
        title: Recipe name.
        summary: One-line description.
        servings: Servings the quantities and nutrition refer to.
        prep_time_minutes: Estimated prep time, ``None`` if not estimated.
        cook_time_minutes: Estimated cook time, ``None`` if not estimated.
        ingredients: Full ingredient list with have/missing flags.
        steps: Numbered instructions.
        nutrition_per_serving: Computed nutrition, never model-generated.
        substitutions: Replacements applied, for the UI to explain.
        warnings: Anything the pipeline could not resolve.
        source_recipe_id: Corpus recipe this was built from, for traceability.
        source_title: Its original title.
        composed_by: ``"llm"`` or ``"fallback"``.
    """

    title: str = Field(min_length=1)
    summary: str = ""
    servings: int = Field(gt=0)
    prep_time_minutes: int | None = None
    cook_time_minutes: int | None = None
    ingredients: list[ComposedIngredient]
    steps: list[str] = Field(min_length=1)
    nutrition_per_serving: NutritionSummary
    substitutions: list[ComposedIngredient] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    source_recipe_id: str = ""
    source_title: str = ""
    composed_by: str = "fallback"

    @property
    def missing_ingredients(self) -> list[str]:
        """Names the user does not have and that were not substituted."""
        return [i.name for i in self.ingredients if not i.have and not i.assumed_staple]


# --- The narrow slice the model is trusted with ------------------------------


class _ComposerResponse(BaseModel):
    """Schema every model response must satisfy before any of it is used.

    ``extra="forbid"`` is deliberate: a response carrying fields we did not ask
    for is a response that did not follow instructions, and the cheapest time
    to find that out is before it reaches the user.
    """

    model_config = ConfigDict(extra="forbid")

    title: str = Field(min_length=1, max_length=200)
    summary: str = Field(default="", max_length=500)
    prep_time_minutes: int | None = Field(default=None, ge=0, le=1440)
    cook_time_minutes: int | None = Field(default=None, ge=0, le=1440)
    steps: list[str] = Field(min_length=1, max_length=40)

    @field_validator("steps")
    @classmethod
    def _clean_steps(cls, values: list[str]) -> list[str]:
        cleaned = [v.strip() for v in values if isinstance(v, str) and v.strip()]
        if not cleaned:
            raise ValueError("steps must contain at least one non-empty instruction")
        return cleaned


COMPOSER_PROMPT = """\
You are writing the final version of a recipe for a home cook.

You will be given the recipe's title, its full ingredient list (already
resolved, including any substitutions), and the original instructions.

Rewrite the instructions so they read clearly and reflect the ingredient list
you were given. If an ingredient was substituted, adjust the wording to use the
replacement.

Rules:
- Use ONLY the ingredients listed. Never introduce a new one.
- Do not restate quantities that contradict the ingredient list.
- Keep every step a single concrete action.
- Estimate prep and cook time in whole minutes.

Respond with JSON only, exactly these fields:
{"title": "...", "summary": "...", "prep_time_minutes": 0,
 "cook_time_minutes": 0, "steps": ["...", "..."]}\
"""


class ComposerError(RuntimeError):
    """Raised when a model response cannot be trusted."""


@dataclass(frozen=True)
class _Assembly:
    """Factual parts of the recipe, built in code before the model is called."""

    ingredients: list[ComposedIngredient]
    substitutions: list[ComposedIngredient]
    warnings: list[str]


class ComposerAgent:
    """Turns pipeline output into a finished, schema-valid recipe."""

    def __init__(self, client=None, model: str | None = None) -> None:
        """Build the agent.

        Args:
            client: A ``groq.Groq`` instance, or any object exposing the same
                ``chat.completions.create``. When omitted the agent composes
                deterministically from the source recipe.
            model: Model id. Falls back to ``GROQ_MODEL`` in the environment.
        """
        self._client = client
        self._model = model
        if client is not None and model is None:
            from config import settings

            self._model = settings.groq_model

    # -- factual assembly, no model involved ---------------------------------

    @property
    def applies_substitutions(self) -> bool:
        """Whether this composer can actually act on a substitution.

        Replacing an ingredient means nothing unless the instructions change to
        match: a list that says flaxseed above steps that say egg contradicts
        itself. Only a composer that rewrites the prose can apply one, so
        without a model the substitutions become suggestions instead.
        """
        return self._client is not None

    def _assemble(
        self, screened: ScreenedMatch, nutrition: NutritionReport, pantry: set[str]
    ) -> _Assembly:
        """Build the ingredient list, substitutions and warnings from the facts."""
        by_original: dict[str, Substitution] = (
            {s.original: s for s in screened.substitutions}
            if self.applies_substitutions
            else {}
        )
        staples = set(screened.match.staples_assumed)

        ingredients: list[ComposedIngredient] = []
        substitutions: list[ComposedIngredient] = []

        for item in screened.recipe.ingredients:
            replacement = by_original.get(item.name)
            if replacement is not None:
                quantity = (
                    item.quantity * replacement.ratio
                    if item.quantity is not None
                    else None
                )
                composed = ComposedIngredient(
                    name=replacement.replacement,
                    quantity=quantity,
                    unit=replacement.unit or item.unit,
                    have=replacement.in_pantry,
                    substituted_for=item.name,
                    note=replacement.note,
                )
                substitutions.append(composed)
            else:
                composed = ComposedIngredient(
                    name=item.name,
                    quantity=item.quantity,
                    unit=item.unit,
                    have=item.name in pantry,
                    assumed_staple=item.name in staples,
                )
            ingredients.append(composed)

        warnings: list[str] = []
        if screened.substitutions and not self.applies_substitutions:
            suggestions = ", ".join(
                f"{s.original} -> {s.replacement}" for s in screened.substitutions
            )
            substitutions = [
                ComposedIngredient(
                    name=s.replacement,
                    have=s.in_pantry,
                    substituted_for=s.original,
                    note=s.note,
                )
                for s in screened.substitutions
            ]
            warnings.append(
                "Suggested substitutions, not applied to the steps below: "
                + suggestions
            )
        if screened.unresolved:
            warnings.append(
                "No substitute found for: " + ", ".join(screened.unresolved)
            )
        if screened.violations:
            offenders = ", ".join(sorted(screened.violations))
            warnings.append(f"Contains avoided allergens: {offenders}")
        if nutrition.coverage < 1.0:
            skipped = ", ".join(name for name, _ in nutrition.unestimated)
            warnings.append(
                f"Nutrition covers {len(nutrition.contributions)} of "
                f"{len(nutrition.contributions) + len(nutrition.unestimated)} "
                f"ingredients; not counted: {skipped}"
            )
        return _Assembly(ingredients, substitutions, warnings)

    @staticmethod
    def _summarize_nutrition(nutrition: NutritionReport) -> NutritionSummary:
        """Round the computed per-serving facts for display."""
        facts = nutrition.per_serving.rounded(1)
        measured = len(nutrition.contributions)
        return NutritionSummary(
            **facts.as_dict(),
            complete=nutrition.coverage >= 1.0,
            measured_ingredients=measured,
            total_ingredients=measured + len(nutrition.unestimated),
        )

    # -- the model call ------------------------------------------------------

    def _write_prose(
        self,
        screened: ScreenedMatch,
        ingredients: list[ComposedIngredient],
        feedback: Sequence[str] = (),
    ) -> _ComposerResponse:
        """Ask the model for title, summary, times and steps.

        Raises:
            ComposerError: If the response is malformed or fails validation.
        """
        listing = "\n".join(
            f"- {i.name}"
            + (f" ({i.quantity:g} {i.unit})" if i.quantity and i.unit else "")
            + (f" [replaces {i.substituted_for}]" if i.substituted_for else "")
            for i in ingredients
        )
        original = "\n".join(
            f"{n}. {step}" for n, step in enumerate(screened.recipe.steps, start=1)
        )
        payload = (
            f"Title: {screened.recipe.title}\n\n"
            f"Ingredients:\n{listing}\n\n"
            f"Original instructions:\n{original}"
        )
        if feedback:
            # Retry feedback is specific ("step 3 still says butter"), so it
            # goes in verbatim rather than as a generic "try again".
            problems = "\n".join(f"- {item}" for item in feedback)
            payload += (
                "\n\nYour previous attempt was rejected for these reasons. "
                f"Fix each one:\n{problems}"
            )

        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": COMPOSER_PROMPT},
                    {"role": "user", "content": payload},
                ],
                response_format={"type": "json_object"},
                temperature=0.2,
            )
            content = response.choices[0].message.content
        except Exception as exc:  # noqa: BLE001 - any client failure degrades
            # to the deterministic fallback rather than failing the request.
            raise ComposerError(f"composer request failed: {exc}") from exc

        try:
            return _ComposerResponse.model_validate_json(content)
        except (ValidationError, json.JSONDecodeError, TypeError) as exc:
            raise ComposerError(f"composer response failed validation: {exc}") from exc

    # -- entry point ---------------------------------------------------------

    def compose(
        self,
        screened: ScreenedMatch,
        nutrition: NutritionReport,
        *,
        pantry: set[str] | None = None,
        servings: int | None = None,
        feedback: Sequence[str] = (),
    ) -> ComposedRecipe:
        """Compose the final recipe.

        Args:
            screened: Safety-screened retrieval match for one recipe.
            nutrition: Nutrition computed for that recipe.
            pantry: Canonical names the user has. Defaults to the match's own
                ``matched`` set.
            servings: Override the serving count the nutrition was computed
                for. Defaults to the nutrition report's.
            feedback: Specific problems from a previous attempt, passed to the
                model verbatim so the retry addresses them rather than
                re-rolling the dice.

        Returns:
            A validated :class:`ComposedRecipe`. ``composed_by`` records
            whether the prose came from the model or the fallback.
        """
        pantry = pantry if pantry is not None else set(screened.match.matched)
        assembly = self._assemble(screened, nutrition, pantry)

        composed_by = "fallback"
        title = screened.recipe.title
        summary = ""
        prep_time = cook_time = None
        steps = list(screened.recipe.steps)

        if self._client is not None:
            try:
                written = self._write_prose(
                    screened, assembly.ingredients, feedback
                )
            except ComposerError as exc:
                assembly.warnings.append(f"Used the original instructions ({exc})")
            else:
                composed_by = "llm"
                title = written.title
                summary = written.summary
                prep_time = written.prep_time_minutes
                cook_time = written.cook_time_minutes
                steps = written.steps

        if not steps:
            steps = ["Combine the ingredients and cook to taste."]
            assembly.warnings.append("The source recipe had no instructions.")

        return ComposedRecipe(
            title=title,
            summary=summary,
            # `or` would treat an explicit servings=0 as unset and
            # silently substitute the report's value; it must be rejected.
            servings=servings if servings is not None else nutrition.servings,
            prep_time_minutes=prep_time,
            cook_time_minutes=cook_time,
            ingredients=assembly.ingredients,
            steps=steps,
            nutrition_per_serving=self._summarize_nutrition(nutrition),
            substitutions=assembly.substitutions,
            warnings=assembly.warnings,
            source_recipe_id=screened.recipe.id,
            source_title=screened.recipe.title,
            composed_by=composed_by,
        )
