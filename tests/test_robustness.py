from src.train_model import main
import random

def test_model_difference():
    seed1 = random.randint(1, 1000)
    seed2 = random.randint(1, 1000)
    processed_data_filepath = "./data/raw/a1_RestaurantReviews_HistoricDump.tsv"
    raw_data_filepath = "./data/processed/processed_data.joblib"
    model_output_filepath = "./models/sentiment_model.joblib"

    output1 = main(processed_data_filepath, raw_data_filepath, model_output_filepath, random_seed=seed1)
    output2 = main(processed_data_filepath, raw_data_filepath, model_output_filepath, random_seed=seed2)
    difference = abs(output1 - output2)
    print(difference)
    assert difference <= 0.1, f"Model difference ({difference}) exceeds the threshold of 0.1"