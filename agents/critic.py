"""Critic agent: validates a composed recipe before the user ever sees it.

Every check here is mechanical. That is deliberate -- the questions the critic
asks ("does this step mention an ingredient that is not in the list?", "does
this recipe contain something the user is allergic to?") have exact answers,
and an exact answer beats a second opinion from the same class of system that
produced the output.

Findings carry the feedback text the composer receives on retry, so a failure
is not just "invalid" but "you used butter in step 3; it was replaced with
olive oil". Vague criticism produces vague corrections.
"""

from __future__ import annotations

from dataclasses import dataclass

from agents.composer import ComposedRecipe
from agents.normalization import canonicalize, mentioned_ingredients
from agents.safety import ScreenedMatch, detect_allergens, normalize_allergens

#: Ingredients a step may mention without them being in the list: things every
#: kitchen has and that recipes reference as technique rather than as content.
ALWAYS_ALLOWED: frozenset[str] = frozenset({
    "water", "salt", "black pepper", "olive oil", "vegetable oil", "sugar",
    "butter", "flour", "all-purpose flour", "ice",
    # "clove" in a step is almost always the unit ("add the garlic cloves"),
    # not the spice; "stock" and "broth" are used interchangeably in prose.
    "clove", "stock", "broth",
})

SEVERITY_ERROR = "error"
SEVERITY_WARNING = "warning"


@dataclass(frozen=True)
class Finding:
    """One thing the critic objects to.

    Attributes:
        code: Stable machine-readable identifier for the check that fired.
        message: Specific, actionable text, given verbatim to the composer on
            retry.
        severity: ``"error"`` blocks acceptance; ``"warning"`` does not.
    """

    code: str
    message: str
    severity: str = SEVERITY_ERROR


@dataclass(frozen=True)
class Verdict:
    """The critic's decision about one composed recipe."""

    findings: tuple[Finding, ...] = ()

    @property
    def ok(self) -> bool:
        """True when nothing blocking was found."""
        return not self.errors

    @property
    def errors(self) -> tuple[Finding, ...]:
        """Findings that block acceptance."""
        return tuple(f for f in self.findings if f.severity == SEVERITY_ERROR)

    @property
    def warnings(self) -> tuple[Finding, ...]:
        """Findings worth surfacing but not blocking."""
        return tuple(f for f in self.findings if f.severity == SEVERITY_WARNING)

    @property
    def feedback(self) -> list[str]:
        """Blocking messages, for the composer's retry prompt."""
        return [f.message for f in self.errors]


class CriticAgent:
    """Checks a composed recipe against its inputs and the user's constraints."""

    def __init__(self, allow_extra: frozenset[str] = ALWAYS_ALLOWED) -> None:
        """Args: allow_extra: Names a step may mention without being listed."""
        self._allow_extra = allow_extra

    def review(
        self,
        composed: ComposedRecipe,
        *,
        screened: ScreenedMatch | None = None,
        avoid: tuple[str, ...] | list[str] = (),
        pantry: set[str] | None = None,
    ) -> Verdict:
        """Validate ``composed`` and return a verdict.

        Args:
            composed: The recipe to check.
            screened: The safety-screened match it was built from, used to
                verify substitutions were actually applied.
            avoid: The user's allergens, in any accepted spelling.
            pantry: Canonical names the user has, used to verify have-flags.

        Returns:
            A :class:`Verdict`. ``ok`` is false when any blocking finding was
            raised, and ``feedback`` carries the text for a retry.
        """
        allergens = normalize_allergens(avoid)
        findings: list[Finding] = []
        findings += self._check_structure(composed)
        findings += self._check_allergens(composed, allergens)
        findings += self._check_no_invented_ingredients(composed)
        if screened is not None:
            findings += self._check_substitutions_applied(composed, screened)
        if pantry is not None:
            findings += self._check_have_flags(composed, pantry)
        return Verdict(tuple(findings))

    # -- individual checks ---------------------------------------------------

    @staticmethod
    def _check_structure(composed: ComposedRecipe) -> list[Finding]:
        """The recipe must be usable at all."""
        findings = []
        if not composed.ingredients:
            findings.append(
                Finding("no_ingredients", "The recipe has no ingredients listed.")
            )
        if not composed.steps:
            findings.append(
                Finding("no_steps", "The recipe has no instructions.")
            )
        if any(not step.strip() for step in composed.steps):
            findings.append(
                Finding("blank_step", "One or more steps are blank. Remove them.")
            )
        if not composed.title.strip():
            findings.append(Finding("no_title", "The recipe has no title."))
        return findings

    @staticmethod
    def _check_allergens(
        composed: ComposedRecipe, allergens: frozenset[str]
    ) -> list[Finding]:
        """No ingredient, listed or mentioned, may trigger an avoided allergen.

        Steps are checked as well as the ingredient list: a recipe whose list
        is clean but whose steps say "finish with a knob of butter" is not safe
        for a dairy allergy.
        """
        if not allergens:
            return []

        findings = []
        for ingredient in composed.ingredients:
            triggered = detect_allergens(ingredient.name) & allergens
            if triggered:
                findings.append(
                    Finding(
                        "allergen_in_ingredients",
                        f"{ingredient.name!r} contains "
                        f"{', '.join(sorted(triggered))}, which must be avoided. "
                        f"Remove it or replace it.",
                    )
                )

        listed = {i.name for i in composed.ingredients}
        for step_number, step in enumerate(composed.steps, start=1):
            for name in sorted(mentioned_ingredients(step)):
                if name in listed:
                    continue
                triggered = detect_allergens(name) & allergens
                if triggered:
                    findings.append(
                        Finding(
                            "allergen_in_steps",
                            f"Step {step_number} mentions {name!r}, which "
                            f"contains {', '.join(sorted(triggered))}. "
                            f"Rewrite the step without it.",
                        )
                    )
        return findings

    def _check_no_invented_ingredients(
        self, composed: ComposedRecipe
    ) -> list[Finding]:
        """Steps may not introduce ingredients absent from the list.

        Severity depends on who wrote the steps. When the model wrote them, an
        unlisted ingredient is a composition failure and retrying with specific
        feedback can fix it. When the steps came from the source recipe
        verbatim, the discrepancy is in the *dataset* -- TheMealDB's Ajo blanco
        genuinely calls for bread its ingredient list omits -- and retrying
        would produce the identical steps forever. Those are reported as
        warnings so the user still sees them.
        """
        severity = (
            SEVERITY_ERROR if composed.composed_by == "llm" else SEVERITY_WARNING
        )
        allowed = set(self._allow_extra)
        for ingredient in composed.ingredients:
            allowed.add(ingredient.name)
            allowed.update(ingredient.name.split())
            if ingredient.substituted_for:
                allowed.update(ingredient.substituted_for.split())

        findings = []
        for step_number, step in enumerate(composed.steps, start=1):
            for name in sorted(mentioned_ingredients(step)):
                if name in allowed or any(part in allowed for part in name.split()):
                    continue
                findings.append(
                    Finding(
                        "invented_ingredient",
                        f"Step {step_number} uses {name!r}, which is not in the "
                        f"ingredient list. Rewrite the step using only the "
                        f"listed ingredients.",
                        severity=severity,
                    )
                )
        return findings

    @staticmethod
    def _check_substitutions_applied(
        composed: ComposedRecipe, screened: ScreenedMatch
    ) -> list[Finding]:
        """A replaced ingredient must not survive in the list or the steps.

        Only checked when the composer actually applied the substitutions. A
        composer with no model leaves them as suggestions and keeps the
        original ingredient deliberately, which is consistent rather than
        wrong -- faulting it would reject every recipe and fix none.
        """
        # A substitution was applied only if the replacement is actually in
        # the ingredient list; a suggestion leaves the original there.
        applied = {
            i.substituted_for for i in composed.ingredients if i.substituted_for
        }
        findings = []
        for substitution in screened.substitutions:
            original = substitution.original
            if original not in applied:
                continue
            if any(i.name == original for i in composed.ingredients):
                findings.append(
                    Finding(
                        "substitution_not_applied",
                        f"{original!r} was replaced with "
                        f"{substitution.replacement!r} but still appears in the "
                        f"ingredient list.",
                    )
                )
            for step_number, step in enumerate(composed.steps, start=1):
                if original in mentioned_ingredients(step):
                    findings.append(
                        Finding(
                            "substitution_ignored_in_steps",
                            f"Step {step_number} still says {original!r}. It was "
                            f"replaced with {substitution.replacement!r} — "
                            f"rewrite the step to use the replacement.",
                        )
                    )
        return findings

    @staticmethod
    def _check_have_flags(
        composed: ComposedRecipe, pantry: set[str]
    ) -> list[Finding]:
        """Have-flags must match the pantry they were derived from."""
        findings = []
        for ingredient in composed.ingredients:
            if ingredient.assumed_staple or ingredient.substituted_for:
                continue
            expected = canonicalize(ingredient.name) in pantry
            if ingredient.have != expected:
                findings.append(
                    Finding(
                        "wrong_have_flag",
                        f"{ingredient.name!r} is marked "
                        f"{'available' if ingredient.have else 'missing'} but the "
                        f"pantry says otherwise.",
                        severity=SEVERITY_WARNING,
                    )
                )
        return findings
