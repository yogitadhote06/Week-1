import streamlit as st
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

#Load required models and encoders
model = joblib.load("C:/Users/HP/Downloads/my code/Python-ML/lr_model.pkl")
scaler = joblib.load("C:/Users/HP/Downloads/my code/Python-ML/scaler.pkl")
#cat_dict = joblib.load("cat_dict.pkl")

st.set_page_config(page_title="CO2 Emission Rate")
st.title("CO2 Emission Rate Prediction App")
st.markdown("Predict CO2 Emission Rate based on factors such as Fuel Type, etc")

# sidebar inputs
st.sidebar.header("Input Parameters")

def get_user_input():
    #'Make', 'Model', 'Vehicle Class', 'Engine Size(L)', 'Cylinders',
      # 'Transmission', 'Fuel Type', 'Fuel Consumption City (L/100 km)',
       #'Fuel Consumption Hwy (L/100 km)', 'Fuel Consumption Comb (L/100 km)',
       #'Fuel Consumption Comb (mpg)'
    #make = st.sidebar.selectbox("Make", cat_dict['Make'])
    #model = st.sidebar.selectbox("Model", cat_dict['Model'])
    #vehicleclass = st.sidebar.selectbox("Vehicle Class", cat_dict['Vehicle Class'])
    #fueltype = st.sidebar.selectbox("Fueltype", cat_dict['Fueltype'])
    fuelconsumptionHwy = st.sidebar.number_input("Fuel Consumption Hwy (L/100 km)", 5, 8, 10)
    enginesize = st.sidebar.number_input("Engine Size", 2,3,4)


    data = {
        #'Make': cat_dict['Make'].tolist().index(make),
        #'Model': cat_dict['Model'].tolist().index(model),
        #'Vehicle Class': cat_dict['Vehicle Class'].tolist().index(vehicleclass),
        #'Fueltype': cat_dict['Fueltype'].tolist().index(fueltype),
        'Fuel Consumption Hwy (L/100 km)':fuelconsumptionHwy,
        'Engine Size':enginesize
    }

    return pd.DataFrame([data])

input_df = get_user_input()

#Prediction Button
if st.button("Predict CO2 Emission Rate"):
    #Optional: scale the input if needed
    # scaled_input = scaler.transform(input_df)
    prediction = model.predict(input_df)

    st.subheader("Prediction Result")
    st.write(f"The estimated value is: .2f")
    
                                                         