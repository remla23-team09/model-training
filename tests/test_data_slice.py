from src.models.train_model import evaluate_model, train_model_loaded_data
from src.data.make_dataset import _preprocess
import joblib
import pytest
import pandas

@pytest.fixture()
def trained_model():
    trained_model = joblib.load("./models/sentiment_model.joblib")
    yield trained_model

@pytest.fixture()
def test_data():
    test_data = pandas.read_csv("./data/raw/a1_RestaurantReviews_HistoricDump.tsv", sep='\t', quoting=3)
    yield test_data

def test_data_slice_negative(trained_model, test_data):
    processed_data_filepath = "./data/raw/a1_RestaurantReviews_HistoricDump.tsv"
    raw_data_filepath = "./data/processed/processed_data.joblib"
    original_score = evaluate_model(trained_model, processed_data_filepath, raw_data_filepath)
    sliced_data = test_data[test_data['Liked'] == 0]
    _preprocess(sliced_data.reset_index(), "test.joblib")
    sliced_score = train_model_loaded_data(trained_model, sliced_data.reset_index(), "test.joblib")
    assert abs(original_score - sliced_score) <= 0.3

def test_data_slice_positive(trained_model, test_data):
    processed_data_filepath = "./data/raw/a1_RestaurantReviews_HistoricDump.tsv"
    raw_data_filepath = "./data/processed/processed_data.joblib"
    original_score = evaluate_model(trained_model, processed_data_filepath, raw_data_filepath)
    sliced_data = test_data[test_data['Liked'] == 1]
    _preprocess(sliced_data.reset_index(), "test.joblib")
    sliced_score = train_model_loaded_data(trained_model, sliced_data.reset_index(), "test.joblib")
    assert abs(original_score - sliced_score) <= 0.3