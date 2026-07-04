"""Eval harness math — pure, model-independent."""
from parla.eval.harness import score_example, prf


def test_perfect_match():
    pred = [{"span": "He go", "suggestion": "He goes"}]
    gold = [{"span": "He go", "correction": "He goes"}]
    assert score_example(pred, gold) == (1, 0, 0)


def test_missed_and_spurious():
    pred = [{"span": "a apple", "suggestion": "an apple"}, {"span": "x", "suggestion": "y"}]
    gold = [{"span": "a apple", "correction": "an apple"}, {"span": "he go", "correction": "he goes"}]
    assert score_example(pred, gold) == (1, 1, 1)


def test_f_beta_precision_weighted():
    m = prf(tp=1, fp=0, fn=1, beta=0.5)
    assert m["precision"] == 1.0 and m["recall"] == 0.5
    assert m["f_beta"] > 0.8


def test_empty_is_zero_not_crash():
    assert prf(0, 0, 0)["f_beta"] == 0.0
