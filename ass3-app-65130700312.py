import streamlit as st
import pickle
import numpy as np
from sklearn.linear_model import Perceptron
model = pickle.load(open('per_model-65130700312.sav','rb'))

st.title("Iris species prediction using Perceptron")

x1 = st.slider('Select Input1',0.0,10.0,3.0)
x2 = st.slider('Select Input2',0.0,10.0,5.0)
x3 = st.slider('Select Input3',0.0,10.0,4.0)
x4 = st.slider('Select Input4',0.0,10.0,7.0)

xnew = np.array([[x1,x2,x3,x4]])#.reshape(1,-1)

xnew

pred = model.predict(xnew)
st.write("## Predict Result:")
st.write('Species:',pred[0])
