import streamlit as st
import joblib
import pandas as pd
from sklearn.datasets import load_iris
from prediction import predict


header = st.container()
dataset = st.container()
inputs = st.container()
modelReq = st.container()
modelTraining = st.container()



with header:
    st.title('Iris Classification')


with dataset:
    st.header('Iris Flower Classification')
    st.text('Information from Iris DataBase')
    iris = load_iris(as_frame=True)
    st.write(iris.data.head())
    st.write(iris.DESCR)

with inputs:
    st.sidebar.header('Iris Flower Classification')
    sepal_length = st.sidebar.slider('Sepal length (cm):', 1.0, 8.0, 4.9)
    sepal_width = st.sidebar.slider('Sepal width (cm):', 0.1, 5.0, 3.0)
    petal_length = st.sidebar.slider('Petal length (cm):', 1.0, 8.0, 1.4)
    petal_width = st.sidebar.slider('Petal width (cm):', 0.1, 5.0, 0.2)

with modelReq:
    df = pd.DataFrame({
        'ML models': ['Logistic Regression', 'Decision Trees', 'SVM', 'Voting Classifier']
    })
    model_option = st.selectbox(
        'Select the machine learning model to use:',
        df['ML models'])

    
with modelTraining:
    if st.sidebar.button ('Classify'):
        data = {'sepal length (cm)': sepal_length,
                'sepal width (cm)': sepal_width,
                'petal length (cm)': petal_length,
                'petal width (cm)': petal_width}
        features = pd.DataFrame(data, index=[0])
        result_LR, result_SVC, result_DT, result_VC  = predict(features, model_option)
        iris_class = ['Iris-Setosa', 'Iris-Versicolour', 'Iris-Virginica']
        st.sidebar.markdown(f'Iris Type by Logistic Regression:  {iris.target_names[result_LR[0]]}')
        st.sidebar.markdown(f'Iris Type by SVC:  {iris.target_names[result_SVC[0]]}')
        st.sidebar.markdown(f'Iris Type by Desicion Trees:  {iris.target_names[result_DT[0]]}')
        st.sidebar.markdown(f'Iris Type by Voting Classifier:  {iris.target_names[result_VC[0]]}')
