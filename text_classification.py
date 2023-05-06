"""
Train the model using different algorithms.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from joblib import dump, load
from text_preprocessing import _load_data

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score

pd.set_option('display.max_colwidth', None)


def main():

    raw_data = _load_data()
    X = load('output/preprocessed_data.joblib')

    #cv = CountVectorizer(max_features = 1420)
    #X = cv.fit_transform(preprocessed_data).toarray()
    y = raw_data.iloc[:, -1].values  
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

    classifier = GaussianNB()
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    accuracy_score(y_test, y_pred) 

    # Store "best" classifier
    dump(classifier, 'output/c2_Classifier_Sentiment_Model.joblib')

if __name__ == "__main__":
    main()
