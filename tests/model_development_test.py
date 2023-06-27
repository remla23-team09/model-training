import joblib
import pandas
import pytest

from src.data.make_dataset import preprocess_test
from src.models.train_model import evaluate_model, train_model, train_model_loaded_data


@pytest.fixture()
def trained_model():
    model = joblib.load("./models/sentiment_model.joblib")
    yield model


@pytest.fixture()
def test_data():
    test_file = "./data/raw/a1_RestaurantReviews_HistoricDump.tsv"
    data = pandas.read_csv(test_file, sep="\t", quoting=3)
    yield data


def test_nondeternism_robustness(trained_model):
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
    sliced_data = test_data[test_data["Liked"] == 0].reset_index()
    assert data_slice(trained_model, sliced_data) <= 0.4


def test_data_slice_positive(trained_model, test_data):
    sliced_data = test_data[test_data["Liked"] == 1].reset_index()
    assert data_slice(trained_model, sliced_data) <= 0.5
