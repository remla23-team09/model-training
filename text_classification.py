"""
Train the model.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from joblib import dump, load
from text_preprocessing import _load_data
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score

pd.set_option('display.max_colwidth', None)


def main():
    # Get the preprocessed data, and split it into test and training data.
    raw_data = _load_data()
    X = load('output/preprocessed_data.joblib')
    y = raw_data.iloc[:, -1].values  
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

    # Train the model
    classifier = GaussianNB()
    classifier.fit(X_train, y_train)

    # Classify the test data
    y_pred = classifier.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    accuracy_score(y_test, y_pred) 

    # Store classifier
    dump(classifier, 'output/c2_Classifier_Sentiment_Model.joblib')

if __name__ == "__main__":
    main()
