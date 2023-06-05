import pytest
import pandas as pd
import joblib
from src.models.train_model import train_model, train_model_loaded_data, evaluate_model
from src.data.make_dataset import preprocess_test
import time
import psutil


@pytest.fixture()
def trained_model():
    model = joblib.load("./models/sentiment_model.joblib")
    yield model

@pytest.fixture()
def test_data():
    test_file = "./data/raw/a1_RestaurantReviews_HistoricDump.tsv"
    data = pd.read_csv(test_file, sep='\t', quoting=3)
    yield data

def test_data_invariants(test_data):
    # Perform necessary data checks or comparisons
    expected_shape = (900, 2)
    assert test_data.shape == expected_shape
    assert test_data.columns.tolist() == ["Review", "Liked"]


def test_model_staleness(trained_model,test_data):
    data_before = preprocess_test(test_data[:450])
    score_before = train_model_loaded_data(trained_model, test_data[:450], data_before)

    # all data come in
    processed_data_filepath = "./data/raw/a1_RestaurantReviews_HistoricDump.tsv"
    raw_data_filepath = "./data/processed/processed_data.joblib"
    score_after = evaluate_model(trained_model, processed_data_filepath, raw_data_filepath)

    # Compare predictions
    assert score_after >= score_before

def test_training_speed(trained_model):
    start_time = time.time()

    processed_data_filepath = "./data/raw/a1_RestaurantReviews_HistoricDump.tsv"
    raw_data_filepath = "./data/processed/processed_data.joblib"
    score_for_time = evaluate_model(trained_model, processed_data_filepath, raw_data_filepath)
    training_time = time.time() - start_time

    baseline_training_time = 0.05
    assert training_time <= baseline_training_time

def test_ram_usage(trained_model):
    processed_data_filepath = "./data/raw/a1_RestaurantReviews_HistoricDump.tsv"
    raw_data_filepath = "./data/processed/processed_data.joblib"
    score_for_ram = evaluate_model(trained_model, processed_data_filepath, raw_data_filepath)

    # Measure current RAM
    current_ram_usage = psutil.Process().memory_info().rss

    baseline_ram_usage = 200000000
    assert current_ram_usage <= baseline_ram_usage

