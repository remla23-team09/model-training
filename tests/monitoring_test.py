"""Monitoring tests."""
import time

import joblib
import pandas as pd
import psutil
import pytest

from src.data.make_dataset import preprocess_test
from src.models.train_model import evaluate_model, train_model_loaded_data


@pytest.fixture()
def trained_model():
    """Returns the trained model."""
    model = joblib.load("./models/sentiment_model.joblib")
    yield model


@pytest.fixture()
def test_data():
    """Returns the restaurant reviews data."""
    test_file = "./data/raw/a1_RestaurantReviews_HistoricDump.tsv"
    data = pd.read_csv(test_file, sep="\t", quoting=3)
    yield data


def test_data_invariants(test_data):
    """Perform necessary data checks or comparisons"""
    expected_shape = (900, 2)
    assert test_data.shape == expected_shape
    assert test_data.columns.tolist() == ["Review", "Liked"]


def test_model_staleness(trained_model, test_data):
    """Test the model staleness."""
    data_before = preprocess_test(test_data[:450])
    score_before = train_model_loaded_data(trained_model, test_data[:450], data_before)

    # all data come in
    processed_data_filepath = "./data/raw/a1_RestaurantReviews_HistoricDump.tsv"
    raw_data_filepath = "./data/processed/processed_data.joblib"
    score_after = evaluate_model(
        trained_model, processed_data_filepath, raw_data_filepath
    )

    # Compare predictions
    assert score_after >= score_before


def test_training_speed(trained_model):
    """Check the training speed of the trained model."""
    start_time = time.time()

    processed_data_filepath = "./data/raw/a1_RestaurantReviews_HistoricDump.tsv"
    raw_data_filepath = "./data/processed/processed_data.joblib"
    evaluate_model(trained_model, processed_data_filepath, raw_data_filepath)
    training_time = time.time() - start_time

    baseline_training_time = 0.05
    assert training_time <= baseline_training_time


def test_ram_usage(trained_model):
    """Check the ram usage of the model training."""
    processed_data_filepath = "./data/raw/a1_RestaurantReviews_HistoricDump.tsv"
    raw_data_filepath = "./data/processed/processed_data.joblib"
    evaluate_model(trained_model, processed_data_filepath, raw_data_filepath)

    # Measure current RAM
    current_ram_usage = psutil.Process().memory_info().rss

    baseline_ram_usage = 2000000000
    assert current_ram_usage <= baseline_ram_usage
