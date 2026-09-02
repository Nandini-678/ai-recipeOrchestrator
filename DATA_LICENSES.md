# Data licences

The code in this repository is MIT. The recipe data is **not** — it comes from
two sources with different terms, and each recipe carries a `source` field
recording which.

## TheMealDB — 790 recipes (`source: "themealdb"`)

<https://www.themealdb.com/api.php>

A free, open recipe API. The public test key (`1`) is used, as the project
documents for open-source and educational use. Recipes retain their
`source_url` for attribution.

## RecipeNLG — 12,000 recipes (`source: "recipenlg"`)

<https://recipenlg.cs.put.poznan.pl/> · Poznań University of Technology

Licensed **CC BY-NC-SA 4.0**: attribution, non-commercial use only, and
derivative datasets must carry the same licence. The subset in
`data/processed/recipes.json` is such a derivative and is therefore also
CC BY-NC-SA 4.0.

> Bień et al., *RecipeNLG: A Cooking Recipes Dataset for Semi-Structured Text
> Generation*, INLG 2020.

**What this means in practice**

- Using, forking and learning from this project is fine.
- Running it commercially is **not**, while the RecipeNLG half is included.
- To remove the restriction, delete `data/processed/recipenlg.json`, rerun
  `python -m scripts.build_corpus`, and the corpus falls back to TheMealDB
  alone. Nothing in the code depends on either source.

## Nutrition — USDA FoodData Central

<https://fdc.nal.usda.gov/> — US Government work, public domain. Responses are
cached locally and are not redistributed in this repository.
