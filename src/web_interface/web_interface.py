import os
import sys
import time
import joblib
import pickle
from git import Repo
from flasgger import Swagger
from flask import Flask, jsonify, request, render_template
from prometheus_client import Counter, Gauge, Histogram, Summary, make_wsgi_app
from werkzeug.middleware.dispatcher import DispatcherMiddleware

sys.path.append(os.getcwd()[:-14])

from models.train_model import train_and_store_model
from models.train_twt_roberta_model import train_and_store_twt_roberta_model

app = Flask(__name__)
swagger = Swagger(app)

app.wsgi_app = DispatcherMiddleware(app.wsgi_app, {
    '/metrics': make_wsgi_app()
})


PATH_OF_GIT_REPO = '../../.git'

def dvc_push():
    os.system('dvc push')

def git_push(commit_message):
    try:
        repo = Repo(PATH_OF_GIT_REPO)
        repo.git.add(update=True)
        repo.index.commit(commit_message)
        origin = repo.remote(name='origin')
        origin.push()
    except:
        print('Some error occured while pushing the code')


def prepare(text):
    cv = pickle.load(open('c1_BoW_Sentiment_Model.pkl', "rb"))
    processed_input = cv.transform([text]).toarray()[0]
    return [processed_input]


@app.route('/train-nb', methods=['POST'])
def train_nb():
    """
    Naive Bayes Model Training Configuration
    ---
    consumes:
      - application/json
    parameters:
        - name: input_data
          in: body
          description: random_seed.
          required: True
          schema:
            type: object
            required: random_seed
            properties:
                random_seed:
                    type: integer
                    example: 5
    responses:
      200:
        description: "The accuracy of the model on the test set."
    """

    test_accuracy = train_and_store_model(
        '../../data/raw/a1_RestaurantReviews_HistoricDump.tsv',
        '../../data/processed/processed_data.joblib',
        '../../models/sentiment_model.joblib',
        request.get_json().get('random_seed'),
    )

    dvc_push()
    git_push('NB Model Training')
    
    res = {
        "test_accuracy": str(test_accuracy),
    }

    return jsonify(res)


@app.route('/train-twt-roberta', methods=['POST'])
def train_twt_roberta():
    """
    Twitter Roberta
    ---
    consumes:
      - application/json
    parameters:
        - name: input_data
          in: body
          description: parameter_description
          required: False
          schema:
            type: object
            required: parameter_name
            properties:
                parameter_name:
                    type: string
                    example: "parameter_name"
    responses:
      200:
        description: "Model Downloaded & Processed."
    """

    train_and_store_twt_roberta_model('../../models/twt_roberta_model.pkl')

    dvc_push()
    git_push('Twitter Roberta Model Training')
    
    res = {
        "parameter_name": "parameter_value",
    }

    return jsonify(res)


@app.route("/")
def home():
    return render_template('index.html') 


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)