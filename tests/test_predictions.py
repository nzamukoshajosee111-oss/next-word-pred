import pytest
from predict import suggest_next_words


def _words(preds):
    # predict returns [{"word": ..., "probability": ...}, ...]
    return [p["word"] if isinstance(p, dict) else p for p in preds]


def test_prediction_returns_results():
    predictions = suggest_next_words("the cat", top_k=3)
    assert predictions is not None
    assert len(predictions) > 0


def test_prediction_returns_at_most_three_words():
    predictions = suggest_next_words("the weather is", top_k=3)
    assert len(predictions) <= 3


def test_predictions_are_words():
    predictions = suggest_next_words("we want to", top_k=3)
    for item in predictions:
        word = item["word"] if isinstance(item, dict) else item
        assert isinstance(word, str)
        assert word.strip() != ""


def test_empty_input_is_handled():
    predictions = suggest_next_words("", top_k=3)
    assert predictions is not None


def test_top_k_limits_number_of_predictions():
    predictions = suggest_next_words("we like", top_k=2)
    assert len(predictions) <= 2