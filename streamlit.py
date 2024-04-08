# streamlit test, version1

import pandas as pd
import numpy as np
from pickle import load
import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from matplotlib import pyplot as plt
from numpy import sqrt
from numpy import argmax
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
import sklearn.metrics as metrics
from sklearn.metrics import plot_roc_curve
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import confusion_matrix
from sklearn.utils import resample
from pickle import load
import shap

# loading in the model to predict on the data
## xgb models
scaler, model, expl = joblib.load('sme.pkl')

## im non
im_sur = Image.open('alive.jpg')
## im pro
im_die = Image.open('die.jpg')

# custom def : shap
def shap(
    sample_case,
    scaler = scaler,
    explainer = expl
):
    # standardization columns
    std_cols=['age','duration','hptd']    
    # feature extraction from input data UA 
    sample_case_features = sample_case.loc[:,['duration', 'vac_janssen', 'Metformin', 'Pioglitazone', 'hptd', 'hptv', 'Sitagliptin', 'male', 'age', 'recoverd_N']]
    sample_case_features[std_cols] = scaler.transform(sample_case_features[std_cols])
    expl_test = explainer.shap_values(sample_case_features.iloc[0])
    shap_bar = pd.DataFrame(
        {'shap_value(probability)'  : expl_test}, index =  ['duration', 'vac_janssen', 'Metformin', 'Pioglitazone', 'hptd', 'hptv', 'Sitagliptin', 'male', 'age', 'recoverd_N'])
  #  clrs = ['blue' if x < 0 else 'red' for x in shap_var['shap']]
    return shap_bar

# custom def : standardization and prediction
def model_prediction(
    sample_case,
    scaler = scaler, 
    model = model
):
    """
    'recoverd_N'
    'age'
    'male'
    'duration'
    'vac_janssen'
    'hptv'
    'Metformin'
    'hptd'
    'Pioglitazone'
    'Sitagliptin'
    """
    
    # standardization columns
    std_cols=['age','duration','hptd']    
    # feature extraction from input data UA 
    sample_case_features = sample_case.loc[:,['duration', 'vac_janssen', 'Metformin', 'Pioglitazone', 'hptd', 'hptv', 'Sitagliptin', 'male', 'age', 'recoverd_N']]
    sample_case_features[std_cols] = scaler.transform(sample_case_features[std_cols])
    
    # predict probability by model
    prob = model.predict_proba(sample_case_features)[:,1]
            
    return np.float64(prob)

def data_mapping(df):
    """this function preprocess the user input
        return type: pandas dataframe
    """
    df.male = df.male.map({'female':0, 'male':1})
    df.recoverd_N = df.recoverd_N.map({'Yes':0, 'No':1})
    df.vac_janssen = df.vac_janssen.map({'No':0, 'Yes':1})
    df.hptv = df.hptv.map({'No':0, 'Yes':1})
    df.Metformin = df.Metformin.map({'No':0, 'Yes':1})
    df.Pioglitazone = df.Pioglitazone.map({'No':0, 'Yes':1})
    df.Sitagliptin = df.Sitagliptin.map({'No':0, 'Yes':1})

    #df.he_ubld = df.he_ubld.map({"-":0, "+/-":1, "1+":2, "2+":3, "3+":4, "4+":5})
    #df.he_upro = df.he_upro.map({"-":0, "+/-":1, "1+":2, "2+":3, "3+":4, "4+":5})
    #df.he_uglu = df.he_uglu.map({"-":0, "+/-":1, "1+":2, "2+":3, "3+":4, "4+":5})
    return df

def main():
      # giving the webpage a title
    st.title("Will you be alive?")
      
    # here we define some of the front end elements of the web page like 
    # the font and background color, the padding and the text to be displayed
    html_temp = """
    <div style ="background-color:grey;padding:13px">
    <h1 style ="color:black;text-align:center;">Prediction Death in diabetes ML App </h1>
    </div>
    """
      
    # this line allows us to display the front end aspects we have 
    # defined in the above code
    st.markdown(html_temp, unsafe_allow_html = True)
      
    # the following lines create text boxes in which the user can enter 
    # the data required to make the prediction

    age = st.sidebar.slider("age", 0, 100, 1)
    male = st.sidebar.selectbox("sex", ("female", "male"))
    vac_janssen = st.sidebar.selectbox("Have you vaccinated Janssen?", ("Yes", "No"))
    recoverd_N = st.sidebar.selectbox("have you recovered?", ("Yes", "No"))
    duration = st.sidebar.slider("How long have you been vaccinated?", 0, 365, 1)
    hptv = st.sidebar.selectbox("Have you ever been to a hospital?", ("Yes", "No"))
    hptd = st.sidebar.slider("How long have you been in the hospital? (day)", 0, 110, 1)
    Metformin = st.sidebar.selectbox("Do you take metformin?", ("Yes", "No"))
    Pioglitazone = st.sidebar.selectbox("Do you take Pioglitazone?", ("Yes", "No"))
    Sitagliptin = st.sidebar.selectbox("Do you take Sitagliptin?", ("Yes", "No"))
    
    features = {
        "duration"    : duration,
        "vac_janssen" : vac_janssen,
        "Metformin"   : Metformin,
        "Pioglitazone": Pioglitazone,
        "hptd"        : hptd,
        "hptv"        : hptv,
        "Sitagliptin" : Sitagliptin,
        "male"        : male,
        "age"         : age,
        "recoverd_N"  : recoverd_N
        }
    sample_case = pd.DataFrame(features, index=[0])
    
    result = ""
    prob = 0.0
    # the below line ensures that when the button called 'Predict' is clicked, 
    # the prediction function defined above is called to make the prediction 
    # and store it in the variable result
    if st.button("Predict"):
        sample_case_map = data_mapping(sample_case)
        result = model_prediction(sample_case_map)
        prob = result
        shap_bar = shap(sample_case_map)

        st.success('probability : {}'.format(result))
    
        if prob < 0.432 :
            st.success("threshold : 0.432")
            st.image(im_sur)
            st.bar_chart(data=shap_bar)
        else :
            st.success("threshold : 0.432")
            st.image(im_die)
            st.bar_chart(data=shap_bar)
     
if __name__=='__main__':
    main()
