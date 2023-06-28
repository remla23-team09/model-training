"""Loading the data, preprocessing and outputting the bow sentiment model."""
import logging
import pickle
import re
from pathlib import Path

import click
import nltk
import pandas as pd
from dotenv import find_dotenv, load_dotenv
from joblib import dump
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer

ps = PorterStemmer()
nltk.download("stopwords")
all_stopwords = stopwords.words("english")
all_stopwords.remove("not")


def _load_data(input_filepath: str) -> pd.DataFrame:
    with open(input_filepath, "r", encoding='utf-8') as file:
        reviews: pd.DataFrame = pd.read_csv(file, sep="\t", quoting=3, dtype={"Review": str})
    return reviews


def _text_process(data):
    review = re.sub("[^a-zA-Z]", " ", data)
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
    review = " ".join(review)
    return review


def _preprocess(reviews, output_filepath):
    count_vectorizer = CountVectorizer(max_features=1420)

    corpus = []
    for i in range(0, 900):
        corpus.append(_text_process(reviews["Review"][i]))

    preprocessed_data = count_vectorizer.fit_transform(corpus).toarray()

    bow_path = "data/interim/c1_BoW_Sentiment_Model.pkl"
    with open(bow_path, "wb") as file:
        pickle.dump(count_vectorizer, file)

    dump(preprocessed_data, output_filepath)
    return preprocessed_data


def preprocess_test(reviews):
    """Preprocess the reviews and return it as a count vectorizer."""

    count_vectorizer = CountVectorizer(max_features=1420, ngram_range=(1, 2))

    corpus = []
    for i in range(0, len(reviews)):
        corpus.append(_text_process(reviews["Review"][i]))

    return count_vectorizer.fit_transform(corpus).toarray()


def prepare(review):
    """Preprocess the input reviews in the same way as the training data."""

    bag_of_words = "../../data/interim/c1_BoW_Sentiment_Model.pkl"
    with open(bag_of_words, "rb") as file:
        count_vectorizer = pickle.load(file)
    processed_input = count_vectorizer.transform([review]).toarray()[0]
    return processed_input

@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
def main_cli(input_filepath, output_filepath):
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")

    reviews = _load_data(input_filepath)

    with pd.option_context("expand_frame_repr", False):
        print(reviews)

    _preprocess(reviews, output_filepath)


def main(input_filepath, output_filepath):
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")

    reviews = _load_data(input_filepath)

    with pd.option_context("expand_frame_repr", False):
        print(reviews)

    _preprocess(reviews, output_filepath)


if __name__ == "__main__":
    LOG_FMT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=LOG_FMT)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main_cli()
