#from text_preprocessing import prepare
import joblib
import pickle

def prepare(review):
    cvFile='c1_BoW_Sentiment_Model.pkl'
    cv = pickle.load(open(cvFile, "rb"))
    processed_input = cv.transform([review]).toarray()[0]
    return processed_input

classifier = joblib.load('c2_Classifier_Sentiment_Model.joblib')
prediction = classifier.predict([prepare("")])[0]
print(prediction)
# 0 is negative and 1 is positive