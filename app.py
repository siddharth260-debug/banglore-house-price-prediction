import streamlit as st
import pandas as pd
import pickle
import numpy as np

model = pickle.load(open("house_price_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
columns = pickle.load(open("columns.pkl", "rb"))
df = pd.read_csv("cleaned_data.csv")

st.title("Bangalore House Price Prediction")

location_list = sorted(df["location"].unique())
location = st.selectbox("Select Location", location_list)
total_sqft = st.number_input("Enter Total Sqft", min_value=300, max_value=10000, value=1000)
bhk = st.number_input("Enter BHK", min_value=1, max_value=10, value=2)
bath = st.number_input("Enter Bathrooms", min_value=1, max_value=10, value=2)

if st.button("Predict Price"):
    # Create empty DataFrame with all training columns
    input_data = pd.DataFrame(columns=columns)
    input_data.loc[0] = 0

    # One-hot encode location
    location_col = f"location_{location}"
    if location_col in input_data.columns:
        input_data.at[0, location_col] = 1

    # Prepare numeric features exactly as scaler expects
    num_cols = ["total_sqft", "BHK", "bath"]
    X_numeric = pd.DataFrame([[total_sqft, bhk, bath]], columns=num_cols)
    input_data[num_cols] = scaler.transform(X_numeric)

    # Predict
    prediction = model.predict(input_data)[0]
    st.success(f"Estimated Price: ₹ {prediction:.2f} Lakhs")

