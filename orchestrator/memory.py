"""Local SQLite store for preferences, run history and saved recipes.

Deliberately a thin wrapper over ``sqlite3`` rather than an ORM. The schema is
three small tables, the queries are short, and a reader can see exactly what
hits the disk -- which is worth more here than the abstraction would be.

Every write is committed immediately and the connection is opened per store
instance with ``check_same_thread=False``, because Streamlit reruns the script
on a different thread than the one that cached the object.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from agents.composer import ComposedRecipe
from config import default_db_path

SCHEMA = """
CREATE TABLE IF NOT EXISTS preferences (
    key        TEXT PRIMARY KEY,
    value      TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS history (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at   TEXT    NOT NULL,
    pantry_text  TEXT    NOT NULL,
    avoid        TEXT    NOT NULL,
    servings     INTEGER NOT NULL,
    accepted     INTEGER NOT NULL,
    recipe_title TEXT,
    recipe_json  TEXT
);

CREATE TABLE IF NOT EXISTS favourites (
    recipe_id   TEXT PRIMARY KEY,
    title       TEXT NOT NULL,
    saved_at    TEXT NOT NULL,
    recipe_json TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS history_created_at ON history (created_at DESC);
"""


@dataclass(frozen=True)
class HistoryEntry:
    """One past run, as stored.

    Attributes:
        id: Row id.
        created_at: UTC timestamp, ISO 8601.
        pantry_text: Exactly what the user typed.
        avoid: Allergens that run excluded.
        servings: Servings requested.
        accepted: Whether the critic passed the recipe.
        recipe_title: Title of the produced recipe, if any.
        recipe: The full recipe, rehydrated, if one was produced.
    """

    id: int
    created_at: str
    pantry_text: str
    avoid: tuple[str, ...]
    servings: int
    accepted: bool
    recipe_title: str | None
    recipe: ComposedRecipe | None


def _now() -> str:
    """Current UTC time as an ISO 8601 string."""
    return datetime.now(UTC).isoformat(timespec="seconds")


class MemoryStore:
    """Preferences, history and favourites, persisted to one SQLite file."""

    def __init__(self, path: Path | str | None = None) -> None:
        """Open (and create if needed) the store.

        Args:
            path: Database file. Defaults to :func:`config.default_db_path`,
                which honours ``RECIPE_DB_PATH``. Pass ``":memory:"`` for an
                ephemeral store, which is what the tests use.
        """
        self.path = path if path is not None else default_db_path()
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(str(self.path), check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._connection.executescript(SCHEMA)
        self._connection.commit()

    def close(self) -> None:
        """Close the underlying connection."""
        self._connection.close()

    def __enter__(self) -> MemoryStore:
        """Support ``with MemoryStore() as store:``."""
        return self

    def __exit__(self, *exc_info: object) -> None:
        """Close on exit."""
        self.close()

    # -- preferences ---------------------------------------------------------

    def set_preference(self, key: str, value: Any) -> None:
        """Store ``value`` under ``key``, replacing any previous value.

        Values are JSON-encoded, so lists and numbers round-trip as themselves
        rather than as their string forms.
        """
        self._connection.execute(
            "INSERT INTO preferences (key, value, updated_at) VALUES (?, ?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value, "
            "updated_at = excluded.updated_at",
            (key, json.dumps(value), _now()),
        )
        self._connection.commit()

    def get_preference(self, key: str, default: Any = None) -> Any:
        """Return the value stored under ``key``, or ``default`` if unset."""
        row = self._connection.execute(
            "SELECT value FROM preferences WHERE key = ?", (key,)
        ).fetchone()
        if row is None:
            return default
        try:
            return json.loads(row["value"])
        except json.JSONDecodeError:
            return default

    def all_preferences(self) -> dict[str, Any]:
        """Every stored preference, as a plain dict."""
        rows = self._connection.execute("SELECT key, value FROM preferences").fetchall()
        result = {}
        for row in rows:
            try:
                result[row["key"]] = json.loads(row["value"])
            except json.JSONDecodeError:
                continue
        return result

    # -- history -------------------------------------------------------------

    def record_run(
        self,
        pantry_text: str,
        *,
        avoid: Iterable[str] = (),
        servings: int = 4,
        accepted: bool = False,
        recipe: ComposedRecipe | None = None,
    ) -> int:
        """Append one pipeline run to the history.

        Returns:
            The new row's id.
        """
        cursor = self._connection.execute(
            "INSERT INTO history (created_at, pantry_text, avoid, servings, "
            "accepted, recipe_title, recipe_json) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                _now(),
                pantry_text,
                json.dumps(list(avoid)),
                servings,
                int(accepted),
                recipe.title if recipe else None,
                recipe.model_dump_json() if recipe else None,
            ),
        )
        self._connection.commit()
        return int(cursor.lastrowid)

    def recent_runs(self, limit: int = 10) -> list[HistoryEntry]:
        """Return the most recent runs, newest first."""
        rows = self._connection.execute(
            "SELECT * FROM history ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [self._to_entry(row) for row in rows]

    def clear_history(self) -> None:
        """Delete every stored run."""
        self._connection.execute("DELETE FROM history")
        self._connection.commit()

    @staticmethod
    def _to_entry(row: sqlite3.Row) -> HistoryEntry:
        """Rehydrate a history row, tolerating a recipe that no longer parses."""
        recipe = None
        if row["recipe_json"]:
            try:
                recipe = ComposedRecipe.model_validate_json(row["recipe_json"])
            except ValueError:
                # A stored recipe from an older schema is history, not an
                # error: keep the row and its title, drop the unusable body.
                recipe = None
        return HistoryEntry(
            id=row["id"],
            created_at=row["created_at"],
            pantry_text=row["pantry_text"],
            avoid=tuple(json.loads(row["avoid"])),
            servings=row["servings"],
            accepted=bool(row["accepted"]),
            recipe_title=row["recipe_title"],
            recipe=recipe,
        )

    # -- favourites ----------------------------------------------------------

    def save_favourite(self, recipe: ComposedRecipe) -> None:
        """Save ``recipe``, replacing any earlier version of the same recipe."""
        self._connection.execute(
            "INSERT INTO favourites (recipe_id, title, saved_at, recipe_json) "
            "VALUES (?, ?, ?, ?) ON CONFLICT(recipe_id) DO UPDATE SET "
            "title = excluded.title, saved_at = excluded.saved_at, "
            "recipe_json = excluded.recipe_json",
            (
                recipe.source_recipe_id or recipe.title,
                recipe.title,
                _now(),
                recipe.model_dump_json(),
            ),
        )
        self._connection.commit()

    def list_favourites(self) -> list[ComposedRecipe]:
        """Every saved recipe, most recently saved first."""
        rows = self._connection.execute(
            "SELECT recipe_json FROM favourites ORDER BY saved_at DESC"
        ).fetchall()
        recipes = []
        for row in rows:
            try:
                recipes.append(ComposedRecipe.model_validate_json(row["recipe_json"]))
            except ValueError:
                continue
        return recipes

    def is_favourite(self, recipe: ComposedRecipe) -> bool:
        """Whether ``recipe`` has already been saved."""
        key = recipe.source_recipe_id or recipe.title
        row = self._connection.execute(
            "SELECT 1 FROM favourites WHERE recipe_id = ?", (key,)
        ).fetchone()
        return row is not None

    def remove_favourite(self, recipe_id: str) -> None:
        """Delete a saved recipe by its id."""
        self._connection.execute(
            "DELETE FROM favourites WHERE recipe_id = ?", (recipe_id,)
        )
        self._connection.commit()
