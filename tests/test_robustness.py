import joblib
import pytest
from src.models.train_model import train_model, evaluate_model

@pytest.fixture()
def trained_model():
    model = joblib.load("./models/sentiment_model.joblib")
    yield model

def test_nondeternism_robustness(trained_model):
    raw_data_filepath = "./data/raw/a1_RestaurantReviews_HistoricDump.tsv"
    processed_data_filepath = "./data/processed/processed_data.joblib"
    original_score = evaluate_model(trained_model, raw_data_filepath, processed_data_filepath)
    for seed in [1,2]:
        model_variant_accuracy = train_model(raw_data_filepath, processed_data_filepath, seed)
        assert abs(original_score - model_variant_accuracy) <= 0.2
