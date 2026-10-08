import pytest

# Import the prediction function from your model or prediction module.
# Update this import to match where your group implements it.
from model import suggest_next_words


def test_prediction_returns_results():
    """The model should return predictions for an input sentence."""
    predictions = suggest_next_words("I am", top_k=3)

    assert predictions is not None
    assert len(predictions) > 0


def test_prediction_returns_at_most_three_words():
    """The model should return no more than three suggestions."""
    predictions = suggest_next_words("The weather is", top_k=3)

    assert len(predictions) <= 3


def test_predictions_are_words():
    """Each suggestion should be a non-empty word."""
    predictions = suggest_next_words("I want to", top_k=3)

    for word in predictions:
        assert isinstance(word, str)
        assert word.strip() != ""


def test_empty_input_is_handled():
    """Empty input should not crash the prediction function."""
    predictions = suggest_next_words("", top_k=3)

    assert predictions is not None


def test_top_k_limits_number_of_predictions():
    """Requesting two suggestions should return at most two."""
    predictions = suggest_next_words("I like", top_k=2)

    assert len(predictions) <= 2
