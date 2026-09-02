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

| Agent | Responsibility | LLM? |
|---|---|---|
| Ingredient | Parse raw text into structured ingredients | yes |
| Retrieval | Rank local recipes by ingredient overlap | no |
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
├── data/
│   ├── raw/        # recipe dataset as fetched
│   └── processed/  # normalized dataset + Chroma index (gitignored)
├── config.py       # single place environment settings are read
└── requirements.txt
```

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
```

The offline agents (ingredient parsing, retrieval, safety, nutrition math) are
tested without network access or API keys.

## Build progress

- [x] **1. Repo scaffolding** — structure, venv, requirements, config, README
- [ ] 2. Ingredient agent
- [ ] 3. Retrieval agent (RAG)
- [ ] 4. Safety / substitution agent
- [ ] 5. Nutrition agent
- [ ] 6. Composer agent
- [ ] 7. Critic agent + full orchestrator
- [ ] 8. SQLite memory + Streamlit UI

## License

MIT
