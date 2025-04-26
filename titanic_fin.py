
import streamlit as st
import numpy as np
import pandas as pd
import pickle
from xgboost import XGBClassifier

# Load the trained model
with open("xgb_model.pkl", "rb") as file:
    model = pickle.load(file)

st.set_page_config(page_title="Titanic Survival Predictor", layout="centered")
st.title("🚢 Titanic Survival Predictor")
st.write("Enter the passenger details below to predict survival.")

# Input fields
pclass = st.selectbox("Passenger Class (1 = 1st, 2 = 2nd, 3 = 3rd)", [1, 2, 3])
sex = st.selectbox("Sex", ["male", "female"])
age = st.slider("Age", 0, 100, 25)
sibsp = st.number_input("No. of Siblings/Spouses Aboard", min_value=0, max_value=10, value=0)
parch = st.number_input("No. of Parents/Children Aboard", min_value=0, max_value=10, value=0)
fare = st.number_input("Fare Paid", min_value=0.0, max_value=600.0, value=50.0)
embarked = st.selectbox("Port of Embarkation", ["C", "Q", "S"])

# Encode categorical variables
sex_encoded = 1 if sex == "male" else 0
embarked_dict = {"C": 0, "Q": 1, "S": 2}
embarked_encoded = embarked_dict[embarked]

# Prediction
if st.button("Predict Survival"):
    input_data = np.array([[pclass, sex_encoded, age, sibsp, parch, fare, embarked_encoded]])
    prediction = model.predict(input_data)[0]
    if prediction == 1:
        st.success("🎉 The passenger is likely to **Survive**.")
    else:
        st.error("⚠️ The passenger is likely to **Not Survive**.")
