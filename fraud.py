import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import sklearn as sklearn
import pickle
from tensorflow.keras.models import load_model


st.title('Fraud Detection')

mlmodel = st.selectbox('Please select the maschine learning model',['LogisticRegression','RandomForest','LightGBM','ANN'])


st.markdown("""
<style>
.big-font {
    font-size:100px !important;
}
</style>
""", unsafe_allow_html=True)

st.sidebar.markdown('**Criterias for Fraud Detection Analysis**')
V1 = st.sidebar.number_input('V1 Value of Customer:',-57.0,3.0,0.0,0.01)
V2 = st.sidebar.number_input('V2 Value of Customer:',-73.0,23.0,0.0,0.01)
V3 = st.sidebar.number_input('V3 Value of Customer:',-49.0,10.0,0.0,0.01)
V4 = st.sidebar.number_input('V4 Value of Customer:',-6.0,17.0,0.0,0.01)
V5 = st.sidebar.number_input('V5 Value of Customer:',-114.0,35.0,0.0,0.01) 
V6 = st.sidebar.number_input('V6 Value of Customer:',-27.0,74.0,0.0,0.01)
V7 = st.sidebar.number_input('V7 Value of Customer:',-44.0,121.0,0.0,0.01)
V8 = st.sidebar.number_input('V8 Value of Customer:',-74.0,21.0,0.0,0.01)
V9 = st.sidebar.number_input('V9 Value of Customer:',-14.0,16.0,0.0,0.01)
V10 = st.sidebar.number_input('V10 Value of Customer:',-25.0,24.0,0.0,0.01)
V11 = st.sidebar.number_input('V11 Value of Customer:',-5.0,13.0,0.0,0.01)
V12 = st.sidebar.number_input('V12 Value of Customer:',-19.0,8.0,0.0,0.01)
V13 = st.sidebar.number_input('V13 Value of Customer:',-6.0,8.0,0.0,0.01)
V14 = st.sidebar.number_input('V14 Value of Customer:',-20.0,11.0,0.0,0.01)
V15 = st.sidebar.number_input('V15 Value of Customer:',-5.0,9.0,0.0,0.01)
V16= st.sidebar.number_input('V16 Value of Customer:',-15.0,18.0,0.0,0.01)
V17 = st.sidebar.number_input('V17 Value of Customer:',-26.0,10.0,0.0,0.01)
V18 = st.sidebar.number_input('V18 Value of Customer:',-10.0,6.0,0.0,0.01)
V19 = st.sidebar.number_input('V19 Value of Customer:',-8.0,6.0,0.0,0.01)
V20 = st.sidebar.number_input('V20 Value of Customer:',-55.0,40.0,0.0,0.01)
V21 = st.sidebar.number_input('V21 Value of Customer:',-35.0,28.0,0.0,0.01)
V22 = st.sidebar.number_input('V22 Value of Customer:',-11.0,11.0,0.0,0.01)
V23 = st.sidebar.number_input('V23 Value of Customer:',-45.0,23.0,0.0,0.01)
V24 = st.sidebar.number_input('V24 Value of Customer:',-3.0,5.0,0.0,0.01)
V25 = st.sidebar.number_input('V25 Value of Customer:',-11.0,8.0,0.0,0.01)
V26 = st.sidebar.number_input('V26 Value of Customer:',-3.0,4.0,0.0,0.01)
V27 = st.sidebar.number_input('V27 Value of Customer:',-23.0,32.0,0.0,0.01)
V28 = st.sidebar.number_input('V28 Value of Customer:',-16.0,34.0,0.0,0.01)
Amount = st.sidebar.number_input('Amount of Customer:',0.0,25692.0,1000.0,0.01)



my_dict = {
        'V1':V1, 'V2':V2, 'V3':V3, 'V4':V4, 'V5':V5, 'V6':V6,
        'V7':V7, 'V8':V8, 'V9':V9, 'V10':V10,
        'V11':V11, 'V12':V12, 'V13':V13, 'V14':V14, 'V15':V15, 'V16':V16, 'V17':V17, 'V18':V18, 'V19':V19, 'V20':V20,
        'V21':V21, 'V22':V22, 'V23':V23, 'V24':V24, 'V25':V25, 'V26':V26, 'V27':V27, 'V28':V28, 'Amount':Amount
        }

my_dict = pd.DataFrame([my_dict])


columns=[
        'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
        'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
        'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']


final_scaler = pickle.load(open('scaler.pkl', "rb"))

my_dict[['Amount']] = final_scaler.transform(my_dict[['Amount']])


if mlmodel == 'LogisticRegression':
    filename1 = "logisticreg.pkl"
    model = pickle.load(open(filename1, "rb"))
    y_pred_proba = model.predict_proba(my_dict)
    if y_pred_proba[:,1] >= 0.074:
        pred=1
    else:
        pred=0    
elif mlmodel == 'RandomForest':
    filename2 = 'randomforest.pkl'
    model = pickle.load(open(filename2, "rb"))
    pred = model.predict(my_dict) 
elif mlmodel == 'LightGBM':
    filename3 = 'lgbm.pkl'
    model = pickle.load(open(filename3, "rb"))
    pred = model.predict(my_dict)
else:
    filename4 = "model_ann.h5"
    model = load_model(filename4)
    pred = model.predict(my_dict)

if st.button('Predict'):
    if pred == 0:
        st.success('This is a safe Transaction.')
    else:
        st.error('That seems a fraud Transaction.')






