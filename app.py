import streamlit as st
import pandas as pd
import numpy as np
import pickle

model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.title("Indian House Price Prediction (Bangalore Dataset)")

# User Inputs
sqft = st.number_input("Enter Total Square Feet", min_value=200, max_value=5000, step=10)
bath = st.number_input("Number of Bathrooms", min_value=1, max_value=10, step=1)
bhk = st.number_input("Number of BHK", min_value=1, max_value=10, step=1)

# Location dropdown
df = pd.read_csv("Final_df.csv")
locations = sorted(df["location"].unique())
location = st.selectbox("Select Location", locations)

# Convert location to one-hot encoding
input_df = pd.DataFrame([[sqft, bath, bhk, location]], 
                        columns=["total_sqft", "bath", "bhk", "location"])

# One-hot encoding
final_df = pd.get_dummies(input_df)

# Align with training columns
train_df = pd.get_dummies(df.drop("price", axis=1))
final_df = final_df.reindex(columns=train_df.columns, fill_value=0)

# Scaling numerical data
num_cols = ["total_sqft", "bath", "bhk"]
final_df[num_cols] = scaler.transform(final_df[num_cols])

# Predict
if st.button("Predict Price"):
    prediction = model.predict(final_df)[0]
    st.success(f"Estimated House Price: ₹ {prediction * 100000:.2f}")
