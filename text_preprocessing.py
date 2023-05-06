"""
Preprocess the data to be trained by the learning algorithm.
"""

import pandas as pd
import nltk
import re
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
from sklearn.feature_extraction.text import CountVectorizer
from joblib import dump
all_stopwords = stopwords.words('english')
all_stopwords.remove('not')
import pickle

def _load_data():
    reviews = pd.read_csv(
        'data/a1_RestaurantReviews_HistoricDump.tsv',
        sep='\t',
        quoting = 3
    )
    return reviews

def _text_process(data):
    review = re.sub('[^a-zA-Z]', ' ', data)
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
    review = ' '.join(review)

    return review

def _preprocess(reviews):

    cv = CountVectorizer(max_features = 1420)

    corpus = []
    for i in range(0, 900):
        corpus.append(_text_process(reviews['Review'][i]))
    preprocessed_data = cv.fit_transform(corpus).toarray()

    bow_path = 'output/c1_BoW_Sentiment_Model.pkl'
    pickle.dump(cv, open(bow_path, "wb"))

    dump(preprocessed_data, 'output/preprocessed_data.joblib')
    return preprocessed_data

def prepare(review):
    cvFile='output/c1_BoW_Sentiment_Model.pkl'
    cv = pickle.load(open(cvFile, "rb"))
    processed_input = cv.transform([review]).toarray()[0]
    return processed_input

def main():
    reviews = _load_data()
    print('\n################### Processed Reviews ###################\n')
    with pd.option_context('expand_frame_repr', False):
        print(reviews)
    _preprocess(reviews)

if __name__ == "__main__":
    main()
