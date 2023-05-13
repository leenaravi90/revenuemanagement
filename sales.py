import streamlit as st
import pandas as pd
import pickle
import h5py

#load file
df = pd.read_csv('/content/Advertising.csv')
df = df.drop('Unnamed: 0', axis=1)

#split into train & test
from sklearn.model_selection import train_test_split

X=df.drop('Sales', axis=1)
y=df.Sales

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2)

#linear regression
from sklearn.linear_model import LinearRegression
modellr = LinearRegression()
modellr.fit(X_train, y_train)
y_pred = modellr.predict(X_test)

pickle.dump(modellr, open("sales.h5", "wb"))

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
st.write(prediction)
