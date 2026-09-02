# AI Recipe Orchestrator

A multi-agent AI assistant that turns *"here's what's in my fridge"* into a
reliable, grounded recipe — with allergen filtering, real substitutions, and
nutrition facts computed in code rather than guessed by a language model.

> **Status:** all eight build steps complete. 543 tests, all offline.

## Why this design

Most LLM recipe demos hand the whole problem to one prompt and hope. This one
splits the job across small, independently testable agents and keeps the model
out of anything it is bad at:

- **Allergen filtering is hard-coded**, not prompt-based — a safety rule you can
  unit test is worth more than one you can jailbreak.
- **Nutrition math happens in Python**, not in the model's head.
- **Every LLM response is validated against a schema** before it is trusted, and
  a critic agent can send it back for a bounded number of retries.
- **Recipes are retrieved from a real dataset** (RAG) rather than invented.

## Architecture

```
ingredients (raw text)
        │
        ▼
  ┌───────────────┐
  │ Ingredient    │  messy text ──▶ [{name, quantity, unit}]
  └───────┬───────┘
          ▼
  ┌───────────────┐
  │ Retrieval     │  RAG over local recipe dataset ──▶ top-N candidates
  └───────┬───────┘
          ▼
  ┌───────────────┐
  │ Safety        │  hard-coded allergen filter + substitution table
  └───────┬───────┘
          ▼
  ┌───────────────┐
  │ Nutrition     │  USDA FoodData Central, scaled in code
  └───────┬───────┘
          ▼
  ┌───────────────┐
  │ Composer      │  LLM ──▶ structured recipe JSON
  └───────┬───────┘
          ▼
  ┌───────────────┐
  │ Critic        │  validates vs. inputs + safety rules ──▶ retry or accept
  └───────┬───────┘
          ▼
    final recipe
```

The orchestrator runs these in plain sequential Python — no agent framework.
Two bounded loops sit inside that straight line: the composer is retried with
the critic's specific complaints (`max_attempts`), and if a candidate recipe
cannot be composed acceptably at all, the next-best one is tried
(`max_candidates`).

Where an agent is marked *optional* for LLM use, the model is an enhancement
rather than a dependency: the ingredient agent uses an LLM only to segment
rambling prose into phrases, and falls back to a rule-based split if the call
fails. All quantity math and name normalization is deterministic Python.

| Agent | Responsibility | LLM? |
|---|---|---|
| Ingredient | Parse raw text into structured ingredients | optional |
| Retrieval | Recall via Chroma, rank by ingredient overlap | no |
| Safety | Allergen filtering + substitution lookup | fallback only |
| Nutrition | Fetch and scale nutrition facts | no |
| Composer | Write the final structured recipe | yes |
| Critic | Validate output, request bounded retries | yes |

## Tech stack

All free tier, no paid services.

| Concern | Choice |
|---|---|
| LLM | [Groq](https://console.groq.com) |
| Recipe data | [TheMealDB](https://www.themealdb.com/api.php) |
| Vector store | [Chroma](https://www.trychroma.com), local |
| Nutrition | [USDA FoodData Central](https://fdc.nal.usda.gov/api-guide.html) |
| Memory | SQLite (stdlib) |
| UI | [Streamlit](https://streamlit.io) |

## Project structure

```
.
├── agents/         # one module per agent, each independently testable
├── orchestrator/
│   ├── pipeline.py # sequential pipeline wiring the agents together
│   └── memory.py   # SQLite preferences, history and saved recipes
├── ui/app.py       # Streamlit front end
├── tests/          # unit tests, one module per agent
├── scripts/        # dataset fetchers and the corpus merger
├── data/
│   ├── raw/        # cached API responses (gitignored)
│   └── processed/  # normalized corpus, committed so the repo runs offline
├── config.py       # single place environment settings are read
└── requirements.txt
```

## The recipe corpus

**12,790 recipes** ship with the repo at `data/processed/recipes.json`, so
nothing needs fetching to run the tests or the app. They come from two sources,
and each recipe records which:

| Source | Recipes | Why it is here |
|---|---|---|
| [TheMealDB](https://www.themealdb.com/api.php) | 790 | Curated dishes with images, cuisines and categories |
| [RecipeNLG](https://recipenlg.cs.put.poznan.pl/) | 12,000 | Simple home cooking — median 7 ingredients — which is what makes *cook with what you have* work |

Licences differ between them and neither is MIT. See
[DATA_LICENSES.md](DATA_LICENSES.md) — the short version is that RecipeNLG is
CC BY-NC-SA 4.0, so this project is fine to fork and learn from but not to run
commercially without dropping that half.

To rebuild:

```bash
python -m scripts.fetch_recipes      # TheMealDB (26 API calls)
python -m scripts.fetch_recipenlg    # RecipeNLG (one 120MB parquet shard)
python -m scripts.build_corpus       # merge into data/processed/recipes.json
```

Both fetchers take `--offline` to re-normalize from their cache without
network access. Corpus ingredients are parsed by the *same* pipeline as user
input, which is what makes overlap matching work: "Plain Flour" in a recipe and
"maida" in your pantry both canonicalize to `all-purpose flour` before they are
ever compared.

## How retrieval ranks

Two stages, deliberately separated:

1. **Recall** — a Chroma vector search narrows the corpus to a candidate pool.
2. **Ranking** — a pure, explainable score orders that pool.

No embedding distance reaches the user-visible ranking. "You have 4 of these 6
ingredients" is a claim the critic can verify and the UI can explain; a cosine
distance is neither.

The score answers *"what can I cook with what I have?"* rather than *"what do I
mostly own?"* — those turn out to be different questions. Ranking by coverage
alone rates a flatbread of flour, water and oil a perfect match, while rating a
supper you could make by buying one onion at 60%. So instead: every ingredient
you already have earns a point, every one you would have to buy costs 1.5, and
long or internally inconsistent recipes are nudged down. A `max_missing` cap
("willing to buy" in the UI) hides anything beyond your patience.

Substitutions follow the same logic: a replacement is only offered if you
already have it. Being told to buy flaxseed instead of an egg is not help.

Common staples (salt, water, oil — salt alone is in 298 of the 790 recipes) are
assumed present and excluded, but still reported so the composer can list them.

### Why two sources

TheMealDB alone is a corpus of composed restaurant dishes, median 10
ingredients, and it made strict *cook-with-what-you-have* nearly impossible:

| | TheMealDB alone | With RecipeNLG |
|---|---|---|
| Median ingredients | 10 | **7** |
| Recipes with ≤6 ingredients | 16% | **45%** |
| Self-consistent ingredient lists | 61% | **85%** |
| Options at "buy ≤1" for a 6-item pantry | 4 | **~30** |

RecipeNLG is curated on the way in — 3–9 ingredients, 2–12 steps, a readable
title, and at most one ingredient its steps use but its list omits — so the
half that gets added is the half that answers the question.

Recipes whose steps name an ingredient their list omits are recorded per recipe
in `unlisted_in_steps` and ranked down. They are deliberately *not* auto-added:
sampling showed roughly half are alternatives ("pork or chicken") or optional
garnishes, so adding them would make the shopping list wrong rather than
right.

## How safety works

Allergen filtering is a lookup table over canonical ingredient names, not a
prompt. Three layers, each auditable in `agents/safety.py`:

1. **Marker tokens** — an ingredient belongs to an allergen if one of its whole
   tokens is a marker. Token equality, not substring, is why *eggplant* is not
   an egg and *butternut squash* is not butter.
2. **Generic-marker negation** — `flour`, `noodle`, `butter` and `cream` name a
   form, not an ingredient, so each is paired with the sources that make it
   safe. That is one rule covering *rice noodle*, *brown rice noodle* and
   *cassava flour* alike, rather than an exception list that never keeps up.
3. **Stated hidden allergens** — standard soy sauce is brewed with wheat,
   Worcestershire contains anchovy, pesto carries parmesan and pine nuts. No
   name-based rule can find these, so they are written down.

Substitution is table-first: a static table with exact ratios, preferring
something the user already has. The LLM is consulted only when the table has
nothing, and **its suggestion is re-screened by the same allergen check before
it is returned** — the advisor is never trusted on safety.

## How nutrition is computed

No model touches a number. Facts come from USDA per 100g, quantities are
converted to grams, scaled linearly, and summed — all ordinary Python.

Converting to grams is the hard part, and the agent is explicit about which of
three cases it is in:

| Case | Example | Basis |
|---|---|---|
| Mass | `1 lb butter` | exact — a definition, ingredient-independent |
| Volume | `1 cup honey` | density table; a cup of honey is 336g, a cup of flour 125g |
| Count | `3 cloves garlic` | typical item weight |

Anything it cannot convert, or that USDA has no match for, lands in
`report.unestimated` **with a reason** rather than being counted as zero — a
panel that quietly omits half the recipe is worse than one that says what it
missed. Rate-limited lookups are reported distinctly from genuine misses, and
are never written to the cache.

USDA's own search ranking is poor for recipe ingredients (searching *olive oil*
returns *Oil, corn, peanut, and olive* first), so candidates are re-ranked to
prefer entries that say little beyond the query and that are raw rather than
prepared.

## What the composer lets the model write

By the time the composer runs, the pipeline already knows the exact quantities,
the exact have/missing flags and the exact nutrition. Asking a model to restate
any of that would trade known-correct numbers for plausible ones.

So the model writes **only prose** — title, summary, cooking times, and the
steps rewritten to reflect substitutions. Everything else is assembled in code:

| Assembled in code | Written by the model |
|---|---|
| ingredient list, quantities, units | title, summary |
| have / missing / assumed-staple flags | numbered steps |
| substitutions and their scaled amounts | prep and cook time estimates |
| nutrition per serving | |
| warnings | |

Responses are validated against a schema that **forbids unexpected fields** and
rejects empty or malformed steps. On any validation failure — or with no API
key at all — the agent falls back to the source recipe's own instructions and
records why. The whole pipeline runs end to end with no credentials.

## What the critic checks

Every check is mechanical, because the questions have exact answers:

| Check | Severity |
|---|---|
| An avoided allergen in the ingredient list **or in a step** | error |
| A substituted-away ingredient still named in the list or steps | error |
| Steps introducing an ingredient that is not listed | error if the model wrote them, warning if copied from the source |
| Structure: no steps, no title, blank steps | error |
| Have-flags disagreeing with the pantry | warning |

That third row is the one worth explaining. When the model writes the steps, an
unlisted ingredient is a composition failure and retrying with specific
feedback fixes it. When the steps come from the dataset verbatim, the
discrepancy is *in the dataset* — TheMealDB's Ajo blanco genuinely calls for
bread its ingredient list omits — and retrying would return identical steps
forever. Across all 790 corpus recipes, this distinction is the difference
between blocking 42% of them and blocking none.

Findings carry the feedback text the composer receives on retry, so a rejection
is not "invalid" but *"step 5 still says egg; it was replaced with flaxseed"*.

## Setup

Requires Python 3.12.

```bash
git clone https://github.com/Nandini-678/ai-recipeOrchestrator.git
cd ai-recipeOrchestrator

python3.12 -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate

pip install -r requirements.txt -r requirements-dev.txt

cp .env.example .env                # then add your keys
```

Both API keys are free: [Groq](https://console.groq.com/keys) and
[USDA FoodData Central](https://fdc.nal.usda.gov/api-key-signup.html).
TheMealDB needs no signup.

## Running the app

```bash
streamlit run ui/app.py
```

Preferences, past runs and saved recipes persist to a local SQLite file
(`data/memory.sqlite3`, or wherever `RECIPE_DB_PATH` points).

The app degrades rather than fails. With no keys at all it still retrieves,
screens for allergens, composes from the source instructions and validates —
everything except model-written prose and USDA nutrition. The sidebar says
which parts are live.

### Deploying to Streamlit Community Cloud

1. Go to <https://share.streamlit.io> and sign in with GitHub.
2. **New app** → repository `Nandini-678/ai-recipeOrchestrator`, branch `main`,
   main file path `ui/app.py`.
3. **Advanced settings** → Python version **3.12**.
4. **Secrets** → paste the contents of
   [`.streamlit/secrets.toml.example`](.streamlit/secrets.toml.example) with
   your real keys. Both are optional; without them the app runs in degraded
   mode and says so in the sidebar.
5. Deploy. The first build takes a few minutes.

`ui/app.py` bridges `st.secrets` into the environment before `config` loads, so
the same code runs locally from `.env` and hosted from secrets.

**Two things to expect on the free tier.** Storage is ephemeral, so saved
recipes and history reset whenever the container restarts — set
`RECIPE_DB_PATH` to a persistent volume on a host that offers one. And
`requirements.txt` deliberately omits Chroma: the vector index is never
imported when serving, and it would add ~175MB to a build that gains nothing
from it, since scoring all 12,790 recipes takes 40ms. Install
`requirements-dev.txt` to use it.

## Running the tests

```bash
pytest
ruff check .
```

The offline agents (ingredient parsing, retrieval, safety, nutrition math) are
tested without network access or API keys.

## Build progress

- [x] **1. Repo scaffolding** — structure, venv, requirements, config, README
- [x] **2. Ingredient agent** — quantity/unit/name parsing, typo + slang
      normalization, optional LLM segmentation (116 tests)
- [x] **3. Retrieval agent** — TheMealDB corpus (790 recipes), Chroma
      recall + deterministic overlap ranking (170 tests)
- [x] **4. Safety agent** — hard-coded allergen filtering across the big 9,
      table-first substitution with screened LLM fallback (276 tests)
- [x] **5. Nutrition agent** — USDA lookup with re-ranking and caching,
      gram conversion, exact serving-scale math (355 tests)
- [x] **6. Composer agent** — strict schema validation, code-assembled
      facts, deterministic fallback (408 tests)
- [x] **7. Critic agent + orchestrator** — mechanical validation, bounded
      retry and candidate fallback, sequential pipeline (471 tests)
- [x] **8. SQLite memory + Streamlit UI** — preferences, history, saved
      recipes, and a front end that degrades without keys (543 tests)

## License

Code: MIT. Recipe data: see [DATA_LICENSES.md](DATA_LICENSES.md) —
RecipeNLG is CC BY-NC-SA 4.0, so the shipped corpus is non-commercial.
