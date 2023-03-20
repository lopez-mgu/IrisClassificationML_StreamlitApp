import streamlit as st
import joblib
import pandas as pd
from sklearn.datasets import load_iris
from prediction import predict


header = st.container()
dataset = st.container()
inputs = st.container()
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
    st.header('Iris Data Inputs')
    st.text('Please enter Petal Length and Petal Width in cm: ')
    #["petal length (cm)", "petal width (cm)"]
    petal_length = st.slider('Petal length', 1.0, 6.9, 1.3)
    petal_width = st.slider('Petal width', 0.1, 2.5, 0.2)

    
with modelTraining:
    if st.button ('Classify'):
        data = {'petal length (cm)': petal_length,
                'petal width (cm)': petal_width}
        features = pd.DataFrame(data, index=[0])
        result = predict(features)
        iris_class = 'Setosa' if result[0] == 0 else 'Versicolor'
        #st.header('Iris Flower Classification Type: ')
        st.text(f'Iris Flower Classification Type:  {iris_class}')

