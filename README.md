# AI Recipe Orchestrator

A multi-agent AI assistant that turns *"here's what's in my fridge"* into a
reliable, grounded recipe — with allergen filtering, real substitutions, and
nutrition facts computed in code rather than guessed by a language model.

> **Status:** in development. See [Build progress](#build-progress).

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
├── orchestrator/   # sequential pipeline wiring the agents together
├── ui/             # Streamlit front end
├── tests/          # unit tests, one module per agent
├── scripts/        # one-off dataset builders
├── data/
│   ├── raw/        # cached API responses (gitignored)
│   └── processed/  # normalized corpus, committed so the repo runs offline
├── config.py       # single place environment settings are read
└── requirements.txt
```

## The recipe corpus

790 recipes from TheMealDB ship with the repo at
`data/processed/recipes.json`, so nothing needs fetching to run the tests or
the app. To rebuild it:

```bash
python -m scripts.fetch_recipes            # fetch, cache, normalize
python -m scripts.fetch_recipes --offline  # re-normalize from the cache
```

Corpus ingredients are parsed by the *same* pipeline as user input, which is
what makes overlap matching work: "Plain Flour" in a recipe and "maida" in your
pantry both canonicalize to `all-purpose flour` before they are ever compared.

## How retrieval ranks

Two stages, deliberately separated:

1. **Recall** — a Chroma vector search narrows the corpus to a candidate pool.
2. **Ranking** — a pure overlap score orders that pool: what fraction of a
   recipe's ingredients do you actually have?

No embedding distance reaches the user-visible ranking. "You have 4 of these 6
ingredients" is a claim the critic agent can verify and the UI can explain; a
cosine distance is neither. Common staples (salt, water, oil — salt alone is in
298 of the 790 recipes) are assumed present and excluded from the score, but
still reported so the composer can list them.

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
- [ ] 5. Nutrition agent
- [ ] 6. Composer agent
- [ ] 7. Critic agent + full orchestrator
- [ ] 8. SQLite memory + Streamlit UI

## License

MIT
