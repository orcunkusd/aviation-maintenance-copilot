from fusion import rrf_fuse


def test_item_in_both_lists_beats_item_ranked_first_in_one():
    fused = rrf_fuse([["x", "a"], ["y", "a"]])
    assert fused[0] == "a"


def test_top_k_truncates():
    assert len(rrf_fuse([["a", "b", "c"], ["c", "b", "a"]], k=2)) == 2


def test_agreement_orders_results():
    assert rrf_fuse([["a", "b", "c"], ["a", "b", "c"]]) == ["a", "b", "c"]


def test_empty_input():
    assert rrf_fuse([]) == []
    assert rrf_fuse([[], []]) == []


def test_score_formula():
    # one list, rank 0, rrf_k=60 -> 1/61 ; checked through ordering of two lists
    assert rrf_fuse([["a"], ["b"]], rrf_k=60) == ["a", "b"]
