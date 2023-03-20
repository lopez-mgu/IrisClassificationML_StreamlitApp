import joblib

def predict(data):
    model = joblib.load('svm_classifier.sav')
    return model.predict(data)