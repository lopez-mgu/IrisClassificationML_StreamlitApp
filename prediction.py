import joblib

def predict(data, model_name):
    #model_savname = {'Logistic Regression': 'LR_tunned.sav', 'Decision Trees': 'DT_tunned.sav', 'SVM': 'SVC_tunned.sav', 'Voting Classifier': 'svm_classifier.sav'}
    #sav_name = model_savname[model_name]
    #model = joblib.load(sav_name)
    #transformed_data = data_transform.transform(data) if model_name != 'Decision Trees' else data

    model_LR = joblib.load('LR_tunned.sav')
    model_DT = joblib.load('DT_tunned.sav')
    model_SVC = joblib.load('SVC_tunned.sav')
    model_VC = joblib.load('voting_clf.sav')

    data_transform= joblib.load('feature_transformation.sav')
    transformed_data = data_transform.transform(data) 

    return model_LR.predict(transformed_data), model_SVC.predict(transformed_data), model_DT.predict(data), model_VC.predict(data)



