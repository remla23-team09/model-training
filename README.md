model-training
==============================

Project containing the model for restaurant reviews sentiment analysis

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>

# How to run the pipeline

Create virtual environment (Windows):
- `python -m venv venv`

Activate the virtual environment:
- `venv\Scripts\activate`

Install requirements:
- `pip install -r requirements.txt`

Set up DVC:
- `pip install dvc`

Pull the files from the DVC remote:
- `dvc pull` (if you get errors, try to `dvc fetch` the files you could not pull, and pull again)

Run the pipeline:
- `dvc repro`

