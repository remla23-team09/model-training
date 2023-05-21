import click
import pandas as pd
from sklearn.model_selection import train_test_split
from joblib import dump, load
from sklearn.naive_bayes import GaussianNB
from src.data.make_dataset import _load_data
from sklearn.metrics import confusion_matrix, accuracy_score

pd.set_option('display.max_colwidth', None)


@click.command()
@click.argument('processed_data_filepath', type=click.Path(exists=True))
@click.argument('raw_data_filepath', type=click.Path(exists=True))
@click.argument('model_output_filepath', type=click.Path())
def main(processed_data_filepath, raw_data_filepath, model_output_filepath):
    # Get the preprocessed data, and split it into test and training data.
    raw_data = _load_data(processed_data_filepath)

    X = load(raw_data_filepath)
    y = raw_data.iloc[:, -1].values

    X_train, X_test, y_train, y_test = train_test_split(
                                        X,
                                        y,
                                        test_size=0.20,
                                        random_state=0
                                        )

    # Train the model
    classifier = GaussianNB()
    classifier.fit(X_train, y_train)

    # Classify the test data
    y_pred = classifier.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    accuracy_score(y_test, y_pred)

    # Store classifier
    dump(classifier, model_output_filepath)


if __name__ == "__main__":
    main()