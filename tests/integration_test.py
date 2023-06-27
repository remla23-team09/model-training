"""Integration tests."""

import joblib
import numpy as np
import pandas as pd
import pytest

from src.data.make_dataset import _load_data, _text_process, preprocess_test
from src.models.train_model import evaluate_model, train_model


@pytest.fixture()
def test_data():
    """Returns the restaurant reviews data."""
    test_file = "./data/raw/a1_RestaurantReviews_HistoricDump.tsv"
    data = pd.read_csv(test_file, sep="\t", quoting=3)
    yield data


@pytest.fixture()
def trained_model():
    """Returns the trained model."""
    model = joblib.load("./models/sentiment_model.joblib")
    yield model


@pytest.fixture()
def mock_data():
    """Returns mock data."""
    mock_data = pd.DataFrame(
        {"Review": ["I love the fish", "Not good"], "Liked": [1, 0]}
    )
    yield mock_data


def test_get_data():
    """Test that we could load the data, by checking it is a pandas dataframe."""
    input_filepath = "./data/raw/a1_RestaurantReviews_HistoricDump.tsv"
    data = _load_data(input_filepath)
    assert isinstance(data, pd.DataFrame)


def test_text_process(mock_data):
    """Test that the text process function works as expected."""
    processed_review = _text_process(mock_data["Review"][0])
    expected_res = "love fish"
    assert processed_review == expected_res


def test_data_preprocessing(mock_data):
    """Test that the preprocessing function works as expected."""
    processed_data = preprocess_test(mock_data)
    assert isinstance(processed_data, np.ndarray)


def test_model_training():
    """Check if the model return accuracy_score."""
    raw_data_filepath = "./data/raw/a1_RestaurantReviews_HistoricDump.tsv"
    processed_data_filepath = "./data/processed/processed_data.joblib"
    model = train_model(raw_data_filepath, processed_data_filepath, 1)
    assert isinstance(model, float)


def test_model_evaluation(trained_model):
    """Check if the evaluate model returns an accuracy score."""
    processed_data_filepath = "./data/raw/a1_RestaurantReviews_HistoricDump.tsv"
    raw_data_filepath = "./data/processed/processed_data.joblib"
    score = evaluate_model(trained_model, processed_data_filepath, raw_data_filepath)
    assert isinstance(score, float)
    assert score >= 0.0 and score <= 1.0
