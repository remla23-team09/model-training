"""
Preprocess the data to be trained by the learning algorithm.
"""

import pandas as pd
import numpy as np
import nltk
import re
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import make_union, make_pipeline
from joblib import dump, load
all_stopwords = stopwords.words('english')
all_stopwords.remove('not')

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
    '''
    1. Convert word tokens from processed msgs dataframe into a bag of words
    2. Convert bag of words representation into tfidf vectorized representation for each message
    3. Add message length
    '''

    corpus = []
    for i in range(0, 900):
        corpus.append(_text_process(reviews['Review'][i]))
    preprocessed_data = corpus

    #preprocessed_data = preprocessor.fit_transform(reviews['Review'])
    dump(_text_process, 'output/preprocessor.joblib')
    dump(preprocessed_data, 'output/preprocessed_data.joblib')
    return preprocessed_data

def prepare(review):
    preprocessor = load('output/preprocessor.joblib')
    return preprocessor.transform([review])

"""def _preprocess(reviews):
    all_stopwords = stopwords.words('english')
    all_stopwords.remove('not')
    corpus=[]

    for i in range(0, 900):
        review = re.sub('[^a-zA-Z]', ' ', reviews['Review'][i])
        review = review.lower()
        review = review.split()
        review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
        review = ' '.join(review)
        corpus.append(review)

    return corpus"""

def main():
    reviews = _load_data()
    print('\n################### Processed Reviews ###################\n')
    with pd.option_context('expand_frame_repr', False):
        print(reviews)
    _preprocess(reviews)

if __name__ == "__main__":
    main()
