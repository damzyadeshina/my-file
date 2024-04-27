# streamlit_app.py
import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Load your trained model
with open("C://Users/toshiba//Downloads//Financial_inclusion_model.pkl", "rb") as f:
    model = pickle.load(f)

# Define input fields for your features
# Replace with the relevant input fields based on your dataset
st.title("Machine Learning Classifier")

feature_1 = st.number_input("Feature 1", value=0.0, step=0.1)
feature_2 = st.number_input("Feature 2", value=0.0, step=0.1)
feature_3 = st.number_input("Feature 3", value=0.0, step=0.1)
# Add more features as needed

# Collect the features into a DataFrame
input_data = pd.DataFrame({
    "Feature 1": [feature_1],
    "Feature 2": [feature_2],
    "Feature 3": [feature_3],
    # Include other features if necessary
})

# Make predictions
if st.button("Predict"):
    prediction = model.predict(input_data)
    st.write("Prediction:", prediction)

