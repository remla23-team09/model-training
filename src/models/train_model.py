import json
import click
import pandas as pd
from joblib import dump, load
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score

pd.set_option('display.max_colwidth', None)

def load_data(input_filepath):
    """Load data from tsv file."""
    reviews = pd.read_csv(
        input_filepath,
        sep='\t',
        quoting=3
    )
    return reviews

def classify(classifier, X_test, y_test):
    """Classify the test data with the trained model, and return accuracy."""
    y_pred = classifier.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    return accuracy_score(y_test, y_pred)

def metrics(accuracy):
    """Print the accuracy metric in the json file."""
    json_object = json.dumps({"accuracy": accuracy})
    with open("metrics.json", "w") as outfile:
        outfile.write(json_object)

@click.command()
@click.argument('processed_data_filepath', type=click.Path(exists=True))
@click.argument('raw_data_filepath', type=click.Path(exists=True))
@click.argument('model_output_filepath', type=click.Path())
def main(processed_data_filepath, raw_data_filepath, model_output_filepath):
    """Train and store the model."""

    # Get the preprocessed data, and split it into test and training data.
    raw_data = load_data(processed_data_filepath)
    X = load(raw_data_filepath)
    y = raw_data.iloc[:, -1].values

    X_train, X_test, y_train, y_test = train_test_split(
                                        X,
                                        y,
                                        test_size=0.20,
                                        random_state=5
                                        )

    # Train the model
    classifier = GaussianNB()
    classifier.fit(X_train, y_train)

    # Classify the test data and output metrics
    accuracy = classify(classifier, X_test, y_test)
    metrics(accuracy)

    # Store classifier
    dump(classifier, model_output_filepath)

def train_model(raw_data_filepath, processed_data_filepath, seed):
    "Train the model with different seeds. Used in tests."

    raw_data = load_data(raw_data_filepath)

    X = load(processed_data_filepath)
    y = raw_data.iloc[:, -1].values

    X_train, X_test, y_train, y_test = train_test_split(
                                        X,
                                        y,
                                        test_size=0.20,
                                        random_state=seed
                                        )

    classifier = GaussianNB()
    classifier.fit(X_train, y_train)

    return classify(classifier, X_test, y_test)

def train_model_loaded_data(classifier, raw_data, processed_data):
    "Same as train_model() but data is sent in as arguments, instead of file paths. Used in tests."

    X = processed_data
    y = raw_data.iloc[:, -1].values

    _, X_test, _, y_test = train_test_split(
                                        X,
                                        y,
                                        test_size=0.20,
                                        random_state=5
                                        )

    return classify(classifier, X_test, y_test)

def evaluate_model(classifier, processed_data_filepath, raw_data_filepath):
    "Evaluate model accuracy for a previous trained model."

    raw_data = load_data(processed_data_filepath)

    X = load(raw_data_filepath)
    y = raw_data.iloc[:, -1].values

    _, X_test, _, y_test = train_test_split(
                                        X,
                                        y,
                                        test_size=0.20,
                                        random_state=5
                                        )

    return classify(classifier, X_test, y_test)

if __name__ == "__main__":
    main()
