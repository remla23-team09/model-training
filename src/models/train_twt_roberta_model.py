"""Training the TWT Roberta model."""

import json
import pickle
import click
import pandas as pd
from transformers import AutoModelForSequenceClassification

pd.set_option("display.max_colwidth", None)

def metrics():
    """Print the result metric in the json file."""
    with open("roberta.json", "w") as outfile:
        json_object = json.dumps({"result": "Model Packaged"})
        outfile.write(json_object)
    return None


@click.command()
@click.argument("model_output_filepath", type=click.Path())
def main(model_output_filepath):
    # Get the preprocessed data, and split it into test and training data.
    classifier = AutoModelForSequenceClassification.from_pretrained(
        "cardiffnlp/twitter-roberta-base-sentiment"
    )

    # Store classifier
    pickle.dump(classifier, open(model_output_filepath, "wb"))
    metrics()


def train_and_store_twt_roberta_model(model_output_filepath):
    # Get the preprocessed data, and split it into test and training data.
    classifier = AutoModelForSequenceClassification.from_pretrained(
        "cardiffnlp/twitter-roberta-base-sentiment"
    )

    # Store classifier
    pickle.dump(classifier, open(model_output_filepath, "wb"))
    metrics()


if __name__ == "__main__":
    main()
