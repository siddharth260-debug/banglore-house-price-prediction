import streamlit as st
import pandas as pd
import numpy as np
import pickle

model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.title("Indian House Price Prediction (Bangalore Dataset)")

df = pd.read_csv("Final_df.csv")

sqft = st.number_input("Total Square Feet", min_value=200, max_value=5000, step=10)
bath = st.number_input("Number of Bathrooms", min_value=1, max_value=10, step=1)
bhk = st.number_input("Number of BHK", min_value=1, max_value=10, step=1)

locations = sorted(df["location"].unique())
location = st.selectbox("Select Location", locations)

input_data = pd.DataFrame([[sqft, bath, bhk, location]],
                          columns=["total_sqft", "bath", "bhk", "location"])

input_encoded = pd.get_dummies(input_data)
train_encoded = pd.get_dummies(df.drop("price", axis=1))
input_encoded = input_encoded.reindex(columns=train_encoded.columns, fill_value=0)

num_cols = ["total_sqft", "bath", "bhk"]
input_encoded[num_cols] = scaler.transform(input_encoded[num_cols])

if st.button("Predict Price"):
    result = model.predict(input_encoded)[0]
    st.success(f"Estimated Price: ₹ {result * 100000:.2f}")
