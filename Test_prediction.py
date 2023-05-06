from text_preprocessing import prepare
import joblib

classifier = joblib.load('output/c2_Classifier_Sentiment_Model.joblib')
prediction = classifier.predict([prepare("Wow... Loved this place.")])[0]
print(prediction)
# 0 is negative and 1 is positive