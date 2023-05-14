import streamlit as st
import pandas as pd
import pickle

#st
st.write("""
# Sales Prediction App

This app predicts the **Total Sales** value!
""")

st.sidebar.header('User Input Parameters')

def user_input_features():
    TV = st.sidebar.slider('Total Number of TV', 0, 300, 100)
    Radio = st.sidebar.slider('Total Number of Radio', 0, 100, 20)
    Newspaper = st.sidebar.slider('Total Number of Newspaper', 0, 150, 50)
    data = {'TV': TV,
            'Radio': Radio,
            'Newspaper': Newspaper}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader('User Input parameters')
st.write(df)

loaded_model = pickle.load(open("sales.h5", "rb"))

prediction = loaded_model.predict(df)

st.subheader('Prediction')
st.write {(prediction[0]).2f}
