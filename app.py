import streamlit as st
import pandas as pd
import pickle

model = pickle.load(open("house_price_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
le = pickle.load(open("location_encoder.pkl", "rb"))
df = pd.read_csv("cleaned_data.csv")

st.title("Bangalore House Price Prediction")

location_list = sorted(df["location"].unique())
location = st.selectbox("Select Location", location_list)
total_sqft = st.number_input("Enter Total Sqft", min_value=300, max_value=10000, value=1000)
bhk = st.number_input("Enter BHK", min_value=1, max_value=10, value=2)
bath = st.number_input("Enter Bathrooms", min_value=1, max_value=10, value=2)

location_map = {loc: idx for idx, loc in enumerate(le.classes_)}

if st.button("Predict Price"):
    loc_encoded = location_map.get(location, -1)
    input_data = pd.DataFrame({
        "location": [loc_encoded],
        "total_sqft": [total_sqft],
        "BHK": [bhk],
        "bath": [bath]
    })

    num_cols = ["total_sqft", "BHK", "bath"]
    input_data[num_cols] = scaler.transform(input_data[num_cols])

    prediction = model.predict(input_data)[0]
    st.success(f"Estimated Price: ₹ {prediction:.2f} Lakhs")
