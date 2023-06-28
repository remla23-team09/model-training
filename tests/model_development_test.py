"""Model development tests."""
import joblib

import os
import pytest
import pandas as pd
from joblib import load
from sklearn.ensemble import RandomForestClassifier

from src.models.train_random_forest import train_and_store_model
from src.data.make_dataset import preprocess_test
from src.models.train_model import evaluate_model, train_model, train_model_loaded_data
from src.models.train_twt_roberta_model import train_and_store_twt_roberta_model


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


def test_nondeternism_robustness(trained_model):
    """Test the nondeternism robustness of our trained model."""
    raw_data_filepath = "./data/raw/a1_RestaurantReviews_HistoricDump.tsv"
    processed_data_filepath = "./data/processed/processed_data.joblib"
    original_score = evaluate_model(
        trained_model, raw_data_filepath, processed_data_filepath
    )
    for seed in [1, 2]:
        model_variant_accuracy = train_model(
            raw_data_filepath, processed_data_filepath, seed
        )
        assert abs(original_score - model_variant_accuracy) <= 0.4


def data_slice(trained_model, sliced_data):
    """Compare the scores of our trained model with a model trained on the sliced data."""
    processed_data_filepath = "./data/raw/a1_RestaurantReviews_HistoricDump.tsv"
    raw_data_filepath = "./data/processed/processed_data.joblib"
    original_score = evaluate_model(
        trained_model, processed_data_filepath, raw_data_filepath
    )
    preprocessed_data = preprocess_test(sliced_data)
    sliced_score = train_model_loaded_data(
        trained_model, sliced_data, preprocessed_data
    )
    return abs(original_score - sliced_score)


def test_data_slice_negative(trained_model, test_data):
    """The sliced data consists only of negative reviews."""
    sliced_data = test_data[test_data["Liked"] == 0].reset_index()
    assert data_slice(trained_model, sliced_data) <= 0.4


def test_data_slice_positive(trained_model, test_data):
    """The sliced data consists only of positive reviews."""
    sliced_data = test_data[test_data["Liked"] == 1].reset_index()
    assert data_slice(trained_model, sliced_data) <= 0.5

def test_train_and_store_twt_roberta_model(tmpdir):
    model_output = os.path.join(tmpdir, "model.pickle")
    train_and_store_twt_roberta_model(model_output)
    # test the model output exists
    assert os.path.exists(model_output)


def test_train_and_store_model(tmpdir):
    model_output = os.path.join(tmpdir, "random_forest_model.joblib")
    train_and_store_model("./data/raw/a1_RestaurantReviews_HistoricDump.tsv", "./data/processed/processed_data.joblib", model_output, random_seed=42)
    # test the model output exists
    assert os.path.exists(model_output)


    classifier = load(model_output)
    # test the classifier is type of RandomForestClassifier
    assert isinstance(classifier, RandomForestClassifier)