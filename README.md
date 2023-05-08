# Assignment A1: Images and Releases

The tasks are:
- For now, simply store all training related files here, more detailed requirements will follow later in the ML part of the course.
- Identify the set of requirements and create a requirements.txt.
- Identify the required steps to train a model. Put the trained model somewhere, so it can be integrated into the model-service.
- At this point, no automation is required for the model training.
- A major goal is to factor out the pre-processing and to make it reusable in the model-service.

What we have done:
- Rewrote the Jupyter notebooks into text_classification.py and text_preprocessing.py in order to prepare for later automation.
- Created the requirements.txt file.
- The model can be trained by running text_preprocessing.py. 
- The pre-processing is made reusable by saving the BoW sentiment model
- The trained model and the BoW sentiment model is made accessible to the model-service by uploading them as artifacts using a Github workflow. 