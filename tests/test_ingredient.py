"""Unit tests for the ingredient agent.

Everything here runs offline. LLM segmentation is exercised through a fake
client so the agent's contract is tested without a network call or an API key.
"""

import pytest

from agents.ingredient import (
    GroqExtractor,
    Ingredient,
    IngredientExtractionError,
    merge_duplicates,
    parse_ingredients,
    parse_phrase,
    split_phrases,
)


class TestParsePhraseQuantities:
    @pytest.mark.parametrize(
        ("phrase", "quantity", "unit", "name"),
        [
            ("2 eggs", 2.0, None, "egg"),
            ("1.5 cups flour", 1.5, "cup", "flour"),
            ("1/2 cup rice", 0.5, "cup", "rice"),
            ("1 1/2 cups milk", 1.5, "cup", "milk"),
            ("½ tsp salt", 0.5, "tsp", "salt"),
            ("1½ tbsp olive oil", 1.5, "tbsp", "olive oil"),
            ("500g chicken", 500.0, "g", "chicken"),
            ("2 tbsp of honey", 2.0, "tbsp", "honey"),
            ("three carrots", 3.0, None, "carrot"),
            ("a dozen eggs", 12.0, None, "egg"),
            ("a couple of onions", 2.0, None, "onion"),
            ("a few mushrooms", 3.0, None, "mushroom"),
        ],
    )
    def test_quantity_forms(self, phrase, quantity, unit, name):
        result = parse_phrase(phrase)
        assert (result.quantity, result.unit, result.name) == (quantity, unit, name)

    def test_range_resolves_to_lower_bound(self):
        """Under-promising the inventory is the safer error."""
        assert parse_phrase("2-3 cloves garlic").quantity == 2.0
        assert parse_phrase("2 to 3 cloves garlic").quantity == 2.0

    @pytest.mark.parametrize(
        "phrase", ["some rice", "a bit of butter", "leftover chicken", "plenty of salt"]
    )
    def test_vague_quantities_become_none_and_do_not_leak_into_the_name(self, phrase):
        result = parse_phrase(phrase)
        assert result.quantity is None
        assert result.name in {"rice", "butter", "chicken", "salt"}

    @pytest.mark.parametrize(
        ("phrase", "quantity", "unit", "name"),
        [
            ("3 garlic cloves", 3.0, "clove", "garlic"),
            ("2 broccoli heads", 2.0, "head", "broccoli"),
            ("2 bread slices", 2.0, "slice", "bread"),
        ],
    )
    def test_trailing_units_are_recognised(self, phrase, quantity, unit, name):
        result = parse_phrase(phrase)
        assert (result.quantity, result.unit, result.name) == (quantity, unit, name)

    def test_a_trailing_unit_is_not_stripped_when_it_is_the_whole_name(self):
        """"2 cloves" is ambiguous, not a nameless quantity."""
        assert parse_phrase("2 cloves") is None

    @pytest.mark.parametrize(
        ("phrase", "quantity"),
        [
            ("maybe 2 onions", 2.0),
            ("like 3 eggs", 3.0),
            ("some 500g flour", 500.0),
            ("maybe a bit of 2 eggs", 2.0),
        ],
    )
    def test_a_hedge_before_a_number_keeps_the_number(self, phrase, quantity):
        """"maybe 2 onions" is two onions, not an unknown quantity of them."""
        assert parse_phrase(phrase).quantity == quantity

    @pytest.mark.parametrize(
        "phrase", ["maybe a bit of cheddar", "like some rice", "just a little salt"]
    )
    def test_stacked_hedges_are_all_stripped(self, phrase):
        result = parse_phrase(phrase)
        assert result.quantity is None
        assert result.name in {"cheddar", "rice", "salt"}

    def test_no_quantity_at_all(self):
        result = parse_phrase("olive oil")
        assert result.quantity is None and result.unit is None
        assert result.name == "olive oil"

    def test_original_text_is_preserved_for_traceability(self):
        assert parse_phrase("  2 Chicken Breasts ").raw == "2 Chicken Breasts"

    @pytest.mark.parametrize("phrase", ["", "   ", "2", "a couple of"])
    def test_phrases_with_no_ingredient_return_none(self, phrase):
        assert parse_phrase(phrase) is None


class TestParsePhraseMessyNames:
    @pytest.mark.parametrize(
        ("phrase", "expected"),
        [
            ("3 Tomatoes", "tomato"),
            ("chopped ONIONS", "onion"),
            ("2 brocolli heads", "broccoli"),
            ("chikcen breast", "chicken breast"),
            ("500g maida", "all-purpose flour"),
            ("2 aubergines", "eggplant"),
            ("a bunch of scallions", "green onion"),
            ("jalapeños", "jalapeno"),
            ("1 can of garbanzo beans", "chickpea"),
        ],
    )
    def test_typos_plurals_and_slang(self, phrase, expected):
        assert parse_phrase(phrase).name == expected


class TestParsePhraseSourceMeasures:
    """Measure formats that appear in the TheMealDB corpus, not in user input."""

    @pytest.mark.parametrize(
        ("phrase", "quantity", "unit", "name"),
        [
            # dual metric/imperial: same amount twice, keep the metric side
            ("175g/6oz digestive biscuits", 175.0, "g", "digestive biscuit"),
            ("50g/1\u00bdoz flaked almonds", 50.0, "g", "flaked almond"),
            ("50ml/2fl oz double cream", 50.0, "ml", "heavy cream"),
            # hyphenated compound measures
            ("8-ounce sliced mushrooms", 8.0, "oz", "mushroom"),
            # mixed number written with a hyphen
            ("2-1/2 cups Flour", 2.5, "cup", "flour"),
            # stacked container words
            ("400g can Chickpeas", 400.0, "g", "chickpea"),
            ("2cm piece Ginger", 2.0, "cm", "ginger"),
            # leading size adjective hiding the unit
            ("Small bunch Coriander", None, "bunch", "cilantro"),
        ],
    )
    def test_corpus_measure_formats(self, phrase, quantity, unit, name):
        result = parse_phrase(phrase)
        assert (result.quantity, result.unit, result.name) == (quantity, unit, name)

    def test_pack_sizes_multiply(self):
        """"3 400g cans of tomatoes" is 1200g of tomatoes, not 3 of something."""
        result = parse_phrase("3 400g Cans Chopped Tomatoes")
        assert (result.quantity, result.unit, result.name) == (1200.0, "g", "tomato")

    def test_juice_of_becomes_the_ingredient_it_names(self):
        assert parse_phrase("Juice of 1/2 Lemon").name == "lemon juice"
        assert parse_phrase("zest of 2 limes").name == "lime zest"

    def test_prep_instructions_do_not_leak_into_the_name(self):
        phrase = "1 chopped into \u00bd-inch pieces Carrots"
        assert parse_phrase(phrase).name == "carrot"

    def test_a_range_with_a_fractional_upper_bound(self):
        assert parse_phrase("1-1\u00bd cups sugar").quantity == 1.0


class TestSplitPhrases:
    def test_splits_on_commas_and_connectives(self):
        assert split_phrases("2 eggs, milk and butter") == ["2 eggs", "milk", "butter"]

    def test_strips_conversational_lead_in(self):
        assert split_phrases("I've got chicken, rice") == ["chicken", "rice"]
        assert split_phrases("there's some tofu") == ["some tofu"]

    @pytest.mark.parametrize(
        "text",
        [
            "uhh I think I have rice",
            "um, so I've got rice",
            "well, there's rice",
            "okay so I have rice",
        ],
    )
    def test_conversational_filler_is_stripped_repeatedly(self, text):
        assert split_phrases(text) == ["rice"]

    def test_handles_bulleted_lists(self):
        assert split_phrases("• 2 eggs\n• 1 cup rice") == ["2 eggs", "1 cup rice"]

    def test_empty_segments_are_dropped(self):
        assert split_phrases("eggs,, , milk") == ["eggs", "milk"]


class TestMergeDuplicates:
    def test_same_name_and_unit_sums(self):
        merged = merge_duplicates(
            [
                Ingredient(name="rice", quantity=1, unit="cup"),
                Ingredient(name="rice", quantity=2, unit="cup"),
            ]
        )
        assert len(merged) == 1 and merged[0].quantity == 3.0

    def test_mismatched_units_are_kept_separate(self):
        """Cups and grams cannot be added without a density table."""
        merged = merge_duplicates(
            [
                Ingredient(name="rice", quantity=1, unit="cup"),
                Ingredient(name="rice", quantity=200, unit="g"),
            ]
        )
        assert len(merged) == 2

    def test_a_known_quantity_fills_in_for_a_vague_one(self):
        merged = merge_duplicates(
            [
                Ingredient(name="egg", quantity=None),
                Ingredient(name="egg", quantity=2),
            ]
        )
        assert len(merged) == 1 and merged[0].quantity == 2.0

    def test_first_seen_order_is_preserved(self):
        merged = merge_duplicates(
            [
                Ingredient(name="rice", quantity=1),
                Ingredient(name="egg", quantity=1),
                Ingredient(name="rice", quantity=1),
            ]
        )
        assert [i.name for i in merged] == ["rice", "egg"]


class TestParseIngredients:
    def test_end_to_end_on_a_messy_sentence(self):
        result = parse_ingredients(
            "I've got 2 chicken breasts, 1/2 cup rice, some brocolli and 3 eggs"
        )
        assert [(i.name, i.quantity, i.unit) for i in result] == [
            ("chicken breast", 2.0, None),
            ("rice", 0.5, "cup"),
            ("broccoli", None, None),
            ("egg", 3.0, None),
        ]

    @pytest.mark.parametrize("text", ["", "   ", "\n"])
    def test_empty_input_returns_empty_list(self, text):
        assert parse_ingredients(text) == []

    def test_never_raises_on_junk_input(self):
        assert isinstance(parse_ingredients("!!! ??? 123 ***"), list)

    def test_duplicates_are_merged_end_to_end(self):
        result = parse_ingredients("1 cup rice and 1 cup rice")
        assert len(result) == 1 and result[0].quantity == 2.0


class _FakeCompletion:
    """Minimal stand-in for the shape groq's client returns."""

    def __init__(self, content):
        self.choices = [
            type("Choice", (), {"message": type("Msg", (), {"content": content})()})()
        ]


class _FakeGroqClient:
    def __init__(self, content):
        self._content = content
        self.chat = type(
            "Chat", (), {"completions": type("C", (), {"create": self._create})()}
        )()

    def _create(self, **kwargs):
        self.last_kwargs = kwargs
        return _FakeCompletion(self._content)


class TestGroqExtractor:
    def test_valid_response_is_segmented(self):
        client = _FakeGroqClient('{"ingredients": ["2 eggs", "1 cup rice"]}')
        extractor = GroqExtractor(client=client, model="fake-model")
        assert extractor("2 eggs and 1 cup rice") == ["2 eggs", "1 cup rice"]

    def test_hallucinated_ingredients_are_discarded(self):
        """Phrases the model did not copy from the source text are dropped."""
        client = _FakeGroqClient('{"ingredients": ["2 eggs", "saffron"]}')
        extractor = GroqExtractor(client=client, model="fake-model")
        assert extractor("2 eggs") == ["2 eggs"]

    @pytest.mark.parametrize(
        "content", ["not json at all", '{"wrong_key": []}', '{"ingredients": "eggs"}']
    )
    def test_malformed_responses_raise_rather_than_return_garbage(self, content):
        extractor = GroqExtractor(client=_FakeGroqClient(content), model="fake-model")
        with pytest.raises(IngredientExtractionError):
            extractor("2 eggs")

    def test_agent_falls_back_to_rules_when_the_llm_fails(self):
        """A broken LLM degrades the result, it must not fail the request."""
        extractor = GroqExtractor(
            client=_FakeGroqClient("garbage"), model="fake-model"
        )
        result = parse_ingredients("2 eggs, 1 cup rice", extractor=extractor)
        assert [i.name for i in result] == ["egg", "rice"]
