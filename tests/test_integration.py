import numpy as np
import pytest
import pandas as pd
import joblib
from src.data.make_dataset import preprocess_test,_text_process
from src.models.train_model import train_model, evaluate_model,load_data
from sklearn.feature_extraction.text import CountVectorizer


@pytest.fixture()
def test_data():
    test_file = "./data/raw/a1_RestaurantReviews_HistoricDump.tsv"
    data = pd.read_csv(test_file, sep='\t', quoting=3)
    yield data

@pytest.fixture()
def trained_model():
    model = joblib.load("./models/sentiment_model.joblib")
    yield model

@pytest.fixture()
def mock_data():
    mock_data = pd.DataFrame({"Review": ["I love the fish", "Not good"], "Liked": [1, 0]})
    yield mock_data

def test_get_data():
    input_filepath = "./data/raw/a1_RestaurantReviews_HistoricDump.tsv"
    data = load_data(input_filepath)
    assert isinstance(data, pd.DataFrame)

def test_text_process(mock_data):
    processed_review = _text_process(mock_data['Review'][0])
    expected_res = "love fish"
    assert processed_review == expected_res

def test_data_preprocessing(mock_data):
    # Mock dataset
    processed_data = preprocess_test(mock_data)
    assert isinstance(processed_data, np.ndarray)

def test_model_training(test_data,trained_model):
    raw_data_filepath = "./data/raw/a1_RestaurantReviews_HistoricDump.tsv"
    processed_data_filepath = "./data/processed/processed_data.joblib"
    model = train_model(raw_data_filepath,processed_data_filepath,1)

    # Check if the model return accuracy_score
    assert isinstance(model, float)

def test_model_evaluation(trained_model):
    processed_data_filepath = "./data/raw/a1_RestaurantReviews_HistoricDump.tsv"
    raw_data_filepath = "./data/processed/processed_data.joblib"
    score = evaluate_model(trained_model, processed_data_filepath, raw_data_filepath)
    assert isinstance(score, float)
    assert score >= 0.0 and score <= 1.0

