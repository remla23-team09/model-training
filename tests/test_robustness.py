from src.models.train_model import train_model, evaluate_model
import joblib
import pytest

@pytest.fixture()
def trained_model():
    trained_model = joblib.load("./models/sentiment_model.joblib")
    yield trained_model

def test_nondeternism_robustness(trained_model):
    raw_data_filepath = "./data/raw/a1_RestaurantReviews_HistoricDump.tsv"
    processed_data_filepath = "./data/processed/processed_data.joblib"
    original_score = evaluate_model(trained_model, raw_data_filepath, processed_data_filepath)
    print(original_score)
    for seed in [1,2]:
        model_variant_accuracy = train_model(raw_data_filepath, processed_data_filepath, seed)
        assert abs(original_score - model_variant_accuracy) <= 0.2