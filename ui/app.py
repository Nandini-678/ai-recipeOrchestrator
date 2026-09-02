"""Streamlit front end for the recipe orchestrator.

Run from the project root::

    streamlit run ui/app.py

The app degrades rather than fails. With no API keys it still retrieves,
screens, composes from the source instructions and validates -- everything
except LLM prose and USDA nutrition. The sidebar says which parts are live.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _load_hosted_secrets() -> None:
    """Copy Streamlit Cloud secrets into the environment.

    ``config`` reads plain environment variables so it stays usable from the
    CLI and the tests. On Streamlit Community Cloud the keys arrive in
    ``st.secrets`` instead, so they are bridged here -- before ``config`` is
    imported, since it snapshots the environment at import time.
    """
    try:
        secrets = dict(st.secrets)
    except Exception:  # noqa: BLE001 - no secrets file is the normal local case
        return
    for key in ("GROQ_API_KEY", "GROQ_MODEL", "USDA_API_KEY", "THEMEALDB_API_KEY"):
        value = secrets.get(key)
        if value and not os.environ.get(key):
            os.environ[key] = str(value)


_load_hosted_secrets()

from agents.composer import ComposerAgent  # noqa: E402
from agents.ingredient import GroqExtractor  # noqa: E402
from agents.nutrition import FoodDataCentralClient, NutritionAgent  # noqa: E402
from agents.recipe import load_recipes  # noqa: E402
from agents.safety import (  # noqa: E402
    ALLERGEN_LABELS,
    KNOWN_ALLERGENS,
    GroqSubstitutionAdvisor,
    SafetyAgent,
)
from config import PROCESSED_DATA_DIR, default_db_path, settings  # noqa: E402
from orchestrator import RecipeOrchestrator  # noqa: E402
from orchestrator.memory import MemoryStore  # noqa: E402

CORPUS_PATH = PROCESSED_DATA_DIR / "recipes.json"
#: Categories in a stable, human order, labelled for display.
ALLERGEN_CHOICES = sorted(KNOWN_ALLERGENS, key=lambda a: ALLERGEN_LABELS[a])


# --- cached resources --------------------------------------------------------


@st.cache_resource(show_spinner="Loading the recipe corpus…")
def get_corpus():
    """Load the recipe corpus once per server process."""
    return load_recipes(CORPUS_PATH)


@st.cache_resource
def get_memory(db_path: str) -> MemoryStore:
    """One SQLite store per database path.

    Keyed on the path rather than taking it from config internally: a cache
    that ignores what its result depends on will happily hand back a store
    pointing at a different file.
    """
    return MemoryStore(db_path)


@st.cache_resource
def get_groq_client():
    """A Groq client, or ``None`` when no key is configured."""
    if not settings.groq_api_key:
        return None
    from groq import Groq

    return Groq(api_key=settings.groq_api_key)


@st.cache_resource
def get_orchestrator(use_llm: bool, use_usda: bool) -> RecipeOrchestrator:
    """Build the pipeline for the current capability toggles."""
    client = get_groq_client() if use_llm else None
    nutrition_client = (
        FoodDataCentralClient.default() if use_usda and settings.usda_api_key else None
    )
    return RecipeOrchestrator(
        get_corpus(),
        safety=SafetyAgent(
            advisor=GroqSubstitutionAdvisor(client=client) if client else None
        ),
        nutrition=NutritionAgent(nutrition_client),
        composer=ComposerAgent(client=client),
        extractor=GroqExtractor(client=client) if client else None,
    )


# --- rendering ---------------------------------------------------------------


def render_ingredients(recipe) -> None:
    """Show the ingredient list with have/missing badges."""
    for item in recipe.ingredients:
        amount = ""
        if item.quantity:
            amount = f"{item.quantity:g} {item.unit or ''}".strip()
        if item.have:
            badge = ":green[have]"
        elif item.assumed_staple:
            badge = ":gray[staple]"
        else:
            badge = ":red[need]"
        # An empty amount must not render as an empty bold span.
        measure = f"**{amount}** " if amount else ""
        line = f"{badge} &nbsp; {measure}{item.name}"
        if item.substituted_for:
            line += f" &nbsp; :violet[← replaces {item.substituted_for}]"
        st.markdown(line, unsafe_allow_html=True)
        if item.note:
            st.caption(f"　{item.note}")


def render_nutrition(recipe) -> None:
    """Show the per-serving nutrition panel."""
    facts = recipe.nutrition_per_serving
    st.subheader("Nutrition per serving")
    columns = st.columns(4)
    for column, (label, value) in zip(
        columns,
        [
            ("Calories", f"{facts.calories:g}"),
            ("Protein", f"{facts.protein_g:g} g"),
            ("Fat", f"{facts.fat_g:g} g"),
            ("Carbs", f"{facts.carbs_g:g} g"),
        ],
        strict=False,
    ):
        column.metric(label, value)

    if facts.complete:
        st.caption(
            f"Computed from all {facts.total_ingredients} ingredients "
            "(USDA FoodData Central, scaled in code)."
        )
    else:
        st.caption(
            f":orange[Partial: {facts.measured_ingredients} of "
            f"{facts.total_ingredients} ingredients could be measured. "
            "The real totals are higher.]"
        )


def render_recipe(recipe, *, memory: MemoryStore, key_prefix: str = "") -> None:
    """Render one composed recipe in full."""
    st.header(recipe.title)
    if recipe.summary:
        st.write(recipe.summary)

    meta = [f"{recipe.servings} servings"]
    if recipe.prep_time_minutes is not None:
        meta.append(f"{recipe.prep_time_minutes} min prep")
    if recipe.cook_time_minutes is not None:
        meta.append(f"{recipe.cook_time_minutes} min cook")
    st.caption(" · ".join(meta))

    for warning in recipe.warnings:
        st.warning(warning, icon="⚠️")

    left, right = st.columns([2, 3])
    with left:
        st.subheader("Ingredients")
        render_ingredients(recipe)
    with right:
        st.subheader("Method")
        for number, step in enumerate(recipe.steps, start=1):
            st.markdown(f"**{number}.** {step}")

    render_nutrition(recipe)

    if memory.is_favourite(recipe):
        st.success("Saved to your recipes.", icon="⭐")
    elif st.button("Save this recipe", key=f"{key_prefix}save"):
        memory.save_favourite(recipe)
        st.rerun()

    st.caption(
        f"Adapted from “{recipe.source_title}” · prose by "
        f"{'the model' if recipe.composed_by == 'llm' else 'the source recipe'}"
    )


# --- pages -------------------------------------------------------------------


def run_search(memory: MemoryStore, pantry_text: str) -> None:
    """Run the pipeline and stash the result in session state.

    The result has to survive reruns. Streamlit re-executes the whole script on
    every interaction, so a result held only in a local variable disappears the
    moment the user clicks anything -- including the Save button, whose handler
    would then never run at all.
    """
    avoid = st.session_state.get("avoid", [])
    servings = st.session_state.get("servings", 4)
    max_missing = st.session_state.get("max_missing", 2)
    memory.set_preference("last_pantry", pantry_text)

    orchestrator = get_orchestrator(
        st.session_state.get("use_llm", True), st.session_state.get("use_usda", True)
    )
    with st.spinner("Retrieving, screening, costing and composing…"):
        result = orchestrator.run(
            pantry_text, avoid=avoid, servings=servings, max_missing=max_missing
        )

    memory.record_run(
        pantry_text,
        avoid=avoid,
        servings=servings,
        accepted=result.accepted,
        recipe=result.recipe,
    )
    st.session_state["result"] = result


def find_page(memory: MemoryStore) -> None:
    """The main search page."""
    st.title("What can I cook?")
    st.write(
        "List what you have. Rough is fine — quantities, typos and slang all work."
    )

    pantry_text = st.text_area(
        "Your ingredients",
        value=memory.get_preference("last_pantry", ""),
        height=110,
        placeholder="2 chicken breasts, an onion, some garlic, tomatoes, cumin…",
    )

    if st.button("Find a recipe", type="primary"):
        if not pantry_text.strip():
            st.info("Add a few ingredients first.")
            return
        run_search(memory, pantry_text)

    result = st.session_state.get("result")
    if result is None:
        return

    if result.recipe is None:
        st.error(result.reason)
        return

    if not result.accepted and result.verdict.errors:
        st.warning(
            "The critic could not fully validate this one — "
            f"{result.verdict.errors[0].message}",
            icon="⚠️",
        )

    shopping = result.recipe.missing_ingredients
    used = sum(1 for i in result.recipe.ingredients if i.have)
    if shopping:
        st.info(
            f"Uses **{used}** of your ingredients. You would need to buy: "
            f"**{', '.join(shopping)}**.",
            icon="🛒",
        )
    else:
        st.success(
            f"Uses **{used}** of your ingredients and nothing else. "
            "You can make this right now.",
            icon="✅",
        )

    st.caption(
        f"Parsed {len(result.pantry)} ingredients · "
        f"{result.candidates_retrieved} recipes matched · "
        f"{result.candidates_safe} passed allergen screening · "
        f"{result.attempt_count} composition attempt(s)"
    )
    render_recipe(result.recipe, memory=memory)


def history_page(memory: MemoryStore) -> None:
    """Past runs."""
    st.title("History")
    entries = memory.recent_runs(limit=25)
    if not entries:
        st.info("Nothing yet. Find a recipe and it will show up here.")
        return

    for entry in entries:
        status = "✅" if entry.accepted else "⚠️"
        title = entry.recipe_title or "no recipe found"
        with st.expander(f"{status} {title} — {entry.created_at}"):
            st.caption(f"You had: {entry.pantry_text}")
            if entry.avoid:
                st.caption(f"Avoiding: {', '.join(entry.avoid)}")
            if entry.recipe:
                render_recipe(entry.recipe, memory=memory, key_prefix=f"h{entry.id}")

    if st.button("Clear history"):
        memory.clear_history()
        st.rerun()


def saved_page(memory: MemoryStore) -> None:
    """Saved recipes."""
    st.title("Saved recipes")
    favourites = memory.list_favourites()
    if not favourites:
        st.info("No saved recipes yet.")
        return
    for recipe in favourites:
        with st.expander(recipe.title):
            render_recipe(
                recipe, memory=memory, key_prefix=f"f{recipe.source_recipe_id}"
            )
            if st.button("Remove", key=f"rm{recipe.source_recipe_id}"):
                memory.remove_favourite(recipe.source_recipe_id or recipe.title)
                st.rerun()


def sidebar(memory: MemoryStore) -> str:
    """Render the sidebar and return the selected page.

    Preference widgets carry an explicit ``key`` and are seeded from session
    state rather than a ``default=``. Without a key, Streamlit derives a
    widget's identity from its arguments -- so a default read back from the
    store changes the identity the moment the preference is written, and the
    selection silently resets.
    """
    if "avoid" not in st.session_state:
        st.session_state["avoid"] = list(memory.get_preference("avoid", []))
    if "servings" not in st.session_state:
        st.session_state["servings"] = int(memory.get_preference("servings", 4))
    if "max_missing" not in st.session_state:
        st.session_state["max_missing"] = int(memory.get_preference("max_missing", 2))

    with st.sidebar:
        st.header("Preferences")

        avoid = st.multiselect(
            "Avoid",
            ALLERGEN_CHOICES,
            key="avoid",
            format_func=lambda category: ALLERGEN_LABELS[category],
            help="Filtered by a hard-coded table, not by asking a model.",
        )
        servings = st.number_input(
            "Servings", min_value=1, max_value=24, key="servings"
        )
        max_missing = st.slider(
            "Willing to buy",
            min_value=0, max_value=5, key="max_missing",
            help="How many ingredients you are happy to go out for. "
                 "0 shows only what you can cook right now.",
        )
        if int(max_missing) != memory.get_preference("max_missing", 2):
            memory.set_preference("max_missing", int(max_missing))
        if avoid != memory.get_preference("avoid", []):
            memory.set_preference("avoid", list(avoid))
        if int(servings) != memory.get_preference("servings", 4):
            memory.set_preference("servings", int(servings))

        st.divider()
        st.header("Capabilities")
        has_groq, has_usda = bool(settings.groq_api_key), bool(settings.usda_api_key)
        st.session_state["use_llm"] = st.toggle(
            "Model-written recipes", value=has_groq, disabled=not has_groq,
            help="Rewrites instructions and handles messy input. Needs GROQ_API_KEY.",
        )
        st.session_state["use_usda"] = st.toggle(
            "USDA nutrition", value=has_usda, disabled=not has_usda,
            help="Real nutrition data. Needs USDA_API_KEY.",
        )
        if not has_groq or not has_usda:
            missing = [
                name
                for name, present in (
                    ("GROQ_API_KEY", has_groq),
                    ("USDA_API_KEY", has_usda),
                )
                if not present
            ]
            st.caption(f"Not set: {', '.join(missing)}. Everything else still works.")

        st.divider()
        return st.radio(
            "View",
            ["Find a recipe", "History", "Saved"],
            label_visibility="collapsed",
        )


def main() -> None:
    """Entry point."""
    st.set_page_config(
        page_title="AI Recipe Orchestrator", page_icon="🥘", layout="wide"
    )

    if not CORPUS_PATH.exists():
        st.error(
            f"No recipe corpus at `{CORPUS_PATH}`.\n\n"
            "Build it with `python -m scripts.fetch_recipes`."
        )
        return

    memory = get_memory(str(default_db_path()))
    page = sidebar(memory)
    if page == "History":
        history_page(memory)
    elif page == "Saved":
        saved_page(memory)
    else:
        find_page(memory)


if __name__ == "__main__":
    main()
