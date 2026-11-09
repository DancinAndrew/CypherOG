"""Reciprocal rank fusion as defined by Cormack, Clarke and Buettcher (SIGIR 2009)."""

import pytest

from rag_query import reciprocal_rank_fusion


def as_dict(fused):
    return dict(fused)


def ids(fused):
    return [cid for cid, _ in fused]


def test_single_list_keeps_its_order_with_the_standard_formula():
    fused = reciprocal_rank_fusion([["a", "b", "c"]], k=60)
    assert ids(fused) == ["a", "b", "c"]
    assert as_dict(fused)["a"] == pytest.approx(1 / 61)
    assert as_dict(fused)["c"] == pytest.approx(1 / 63)


def test_ids_found_by_both_retrievers_rank_first():
    fused = reciprocal_rank_fusion([["a", "b"], ["b", "c"]], k=60)
    assert ids(fused)[0] == "b"
    assert as_dict(fused)["b"] == pytest.approx(1 / 62 + 1 / 61)


def test_a_list_that_misses_an_id_adds_nothing_to_it():
    fused = reciprocal_rank_fusion([["a"], ["z"]], k=60)
    assert as_dict(fused)["a"] == pytest.approx(1 / 61)
    assert as_dict(fused)["z"] == pytest.approx(1 / 61)


def test_bm25_only_hits_interleave_with_dense_hits():
    # The old 0.7/0.3 weighting gave a BM25-only rank-1 hit 0.0056, below every
    # dense candidate, so lexical matches could never enter the top 5.
    dense = [f"d{i}" for i in range(15)]
    fused = reciprocal_rank_fusion([dense, ["x", "y"]], k=60)
    assert ids(fused)[:5] == ["d0", "x", "d1", "y", "d2"]


def test_ties_go_to_the_earlier_list_then_to_the_id():
    assert ids(reciprocal_rank_fusion([["b"], ["a"]], k=60)) == ["b", "a"]
    assert ids(reciprocal_rank_fusion([["b", "a"], ["a", "b"]], k=60)) == ["a", "b"]


def test_duplicates_within_a_list_count_once_at_their_first_rank():
    fused = reciprocal_rank_fusion([["a", "b", "a"]], k=60)
    assert ids(fused) == ["a", "b"]
    assert as_dict(fused)["a"] == pytest.approx(1 / 61)


def test_weights_scale_each_list():
    fused = reciprocal_rank_fusion([["a"], ["b"]], k=60, weights=[2.0, 1.0])
    assert as_dict(fused)["a"] == pytest.approx(2 / 61)
    assert as_dict(fused)["b"] == pytest.approx(1 / 61)


def test_weights_must_match_the_number_of_lists():
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([["a"]], k=60, weights=[1.0, 1.0])


def test_empty_input_fuses_to_nothing():
    assert reciprocal_rank_fusion([], k=60) == []
    assert reciprocal_rank_fusion([[], []], k=60) == []


# --- extensions -------------------------------------------------------------


def test_k_shifts_every_rank_by_the_same_constant():
    fused = reciprocal_rank_fusion([["a", "b"]], k=0)
    assert as_dict(fused)["a"] == pytest.approx(1.0)
    assert as_dict(fused)["b"] == pytest.approx(0.5)


def test_negative_k_is_rejected():
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([["a"]], k=-1)


def test_on_an_exact_tie_a_better_single_rank_wins():
    # With k=0: p = 1/1, x = 1/1, q = 1/2 + 1/2; all three tie at exactly 1.0.
    # p and x were ranked first by one retriever, q only second by both.
    fused = reciprocal_rank_fusion([["p", "q"], ["x", "q"]], k=0)
    assert set(as_dict(fused).values()) == {1.0}
    assert ids(fused) == ["p", "x", "q"]


def test_inputs_are_not_mutated():
    lists = [["a", "b"], ["b"]]
    weights = [1.0, 1.0]
    reciprocal_rank_fusion(lists, k=60, weights=weights)
    assert lists == [["a", "b"], ["b"]]
    assert weights == [1.0, 1.0]
