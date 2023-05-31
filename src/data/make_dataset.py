# -*- coding: utf-8 -*-
from dotenv import find_dotenv, load_dotenv
import pandas as pd
import nltk
import re
import click
import logging
from pathlib import Path
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from joblib import dump
from sklearn.feature_extraction.text import CountVectorizer
import pickle


ps = PorterStemmer()
nltk.download('stopwords')
all_stopwords = stopwords.words('english')
all_stopwords.remove('not')


# OLD: data/a1_RestaurantReviews_HistoricDump.tsv
# NEW: data/raw/a1_RestaurantReviews_HistoricDump.tsv
def _load_data(input_filepath):
    reviews = pd.read_csv(
        input_filepath,
        sep='\t',
        quoting=3
    )
    return reviews


def _text_process(data):
    review = re.sub('[^a-zA-Z]', ' ', data)
    review = review.lower()
    review = review.split()
    review = [
        ps.stem(word) for word in review if not word in set(all_stopwords)
        ]
    review = ' '.join(review)
    return review


def _preprocess(reviews, output_filepath):

    count_vectorizer = CountVectorizer(max_features=1420)

    corpus = []
    for i in range(0, 900):
        corpus.append(_text_process(reviews['Review'][i]))

    preprocessed_data = count_vectorizer.fit_transform(corpus).toarray()

    bow_path = 'data/interim/c1_BoW_Sentiment_Model.pkl'
    pickle.dump(count_vectorizer, open(bow_path, "wb"))

    dump(preprocessed_data, output_filepath)
    return preprocessed_data


def prepare(review):
    bag_of_words = '../../data/interim/c1_BoW_Sentiment_Model.pkl'
    count_vectorizer = pickle.load(open(bag_of_words, "rb"))
    processed_input = count_vectorizer.transform([review]).toarray()[0]
    return processed_input


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    reviews = _load_data(input_filepath)

    with pd.option_context('expand_frame_repr', False):
        print(reviews)

    _preprocess(reviews, output_filepath)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
