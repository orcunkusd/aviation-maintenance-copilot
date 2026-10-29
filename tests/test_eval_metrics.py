from eval_metrics import (key_in_passages, citation_correct, cited_numbers, expected_pages, is_refusal,
                          retrieval_hit, score_row, summarize)


def hit(page, source="AC.pdf"):
    return {"id": f"{source}{page}", "text": "t", "meta": {"source": source, "page": page}}


ITEM = {"q": "q?", "answerable": True, "source": "AC.pdf", "pages": [10, 11]}


def test_cited_numbers_and_refusal():
    assert cited_numbers("a [2] b [1][2]") == [1, 2]
    assert is_refusal("I could not find it in the provided documents.")
    assert not is_refusal("Use 3 diameters [1].")


def test_expected_pages_accepts_page_or_pages():
    assert expected_pages({"page": 5}) == [5]
    assert expected_pages({"pages": [1, 2], "page": 9}) == [1, 2]
    assert expected_pages({}) == []


def test_retrieval_hit_requires_source_and_page():
    assert retrieval_hit([hit(3), hit(11)], ITEM)
    assert not retrieval_hit([hit(3), hit(11, source="Other.pdf")], ITEM)


def test_citation_correct_needs_a_valid_cited_expected_page():
    hits = [hit(3), hit(10)]
    assert citation_correct("Yes [2].", hits, ITEM)
    assert not citation_correct("Yes [1].", hits, ITEM)      # cites the wrong passage
    assert not citation_correct("Yes [9].", hits, ITEM)      # cites a passage that does not exist
    assert not citation_correct("Yes.", hits, ITEM)          # no citation


def test_score_row_for_unanswerable_has_no_retrieval_fields():
    row = score_row({"q": "x", "answerable": False}, "I could not find it.", [hit(1)])
    assert row["refused"] is True and row["retrieval_hit"] is None and row["citation_correct"] is None


def test_summarize_rates():
    rows = [
        dict(answerable=True, refused=False, retrieval_hit=True, citation_correct=True, llm_calls=1, seconds=1.0),
        dict(answerable=True, refused=True, retrieval_hit=True, citation_correct=False, llm_calls=1, seconds=3.0),
        dict(answerable=False, refused=True, retrieval_hit=None, citation_correct=None, llm_calls=1, seconds=2.0),
    ]
    s = summarize(rows)
    assert s["citation_correct"] == 0.5 and s["false_refusal"] == 0.5 and s["correct_refusal"] == 1.0
    assert s["n_answerable"] == 2 and s["n_unanswerable"] == 1 and s["avg_seconds"] == 2.0


def test_comma_separated_citations_are_understood():
    assert cited_numbers("No more than three threads [2, 4].") == [2, 4]
    assert cited_numbers("A [1][3] and B [2,3]") == [1, 2, 3]
    assert cited_numbers("no brackets 2, 4") == []
    assert cited_numbers("array[0] is not a citation list") == [0]  # digits in brackets count; harmless


def test_key_in_passages_is_stricter_than_page_hit():
    item = {"q": "q?", "answerable": True, "source": "AC.pdf", "pages": [10], "key": "no more than three"}
    right_page_wrong_text = [{"id": "a", "text": "Other text on the page.", "meta": {"source": "AC.pdf", "page": 10}}]
    right_text = [{"id": "b", "text": "Use  NO MORE\nthan three threads.", "meta": {"source": "AC.pdf", "page": 10}}]
    assert retrieval_hit(right_page_wrong_text, item) is True
    assert key_in_passages(right_page_wrong_text, item) is False
    assert key_in_passages(right_text, item) is True
    assert key_in_passages(right_text, {"q": "x"}) is None
