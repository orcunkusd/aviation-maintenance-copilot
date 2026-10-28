from textutil import tokenize


def test_lowercases_and_splits_words():
    assert tokenize("Rivet Spacing") == ["rivet", "spacing"]


def test_en_dash_in_ad_numbers_is_normalised():
    assert "2020-06-15" in tokenize("AD 2020–06–15")


def test_keeps_hyphenated_part_numbers_together():
    assert "13-1b" in tokenize("AC 43.13-1B")


def test_empty_string():
    assert tokenize("") == []
