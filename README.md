model-training
==============================

Project containing the model for restaurant reviews sentiment analysis

# Model-training

What we have done:
- Rewrote the Jupyter notebooks into text_classification.py and text_preprocessing.py in order to prepare for later automation.
- Created the requirements.txt file.
- The model can be trained by running text_preprocessing.py. 
- The pre-processing is made reusable by saving the BoW sentiment model
- The trained model and the BoW sentiment model is made accessible to the model-service by uploading them as artifacts using a Github workflow. However, for now we have manually copied the models to model-service. 


Project Organization:
<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
