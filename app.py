import streamlit as st
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Muat model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

scaler = StandardScaler()

st.title('Prediksi Tarif Uber')

# Input Fitur
year = st.number_input('Year', min_value=2000, max_value=2025, value=2023)
month = st.number_input('Month', min_value=1, max_value=12, value=1)
day = st.number_input('Day', min_value=1, max_value=31, value=1)
hour = st.number_input('Hour', min_value=0, max_value=23, value=0)
minute = st.number_input('Minute', min_value=0, max_value=59, value=0)
second = st.number_input('Second', min_value=0, max_value=59, value=0)
pickup_longitude = st.number_input('Pickup Longitude')
pickup_latitude = st.number_input('Pickup Latitude')
dropoff_longitude = st.number_input('Dropoff Longitude')
dropoff_latitude = st.number_input('Dropoff Latitude')

if st.button('Predict Fare'):
    features = pd.DataFrame({
        'year': [year],
        'month': [month],
        'day': [day],
        'hour': [hour],
        'minute': [minute],
        'second': [second],
        'pickup_longitude': [pickup_longitude],
        'pickup_latitude': [pickup_latitude],
        'dropoff_longitude': [dropoff_longitude],
        'dropoff_latitude': [dropoff_latitude]
    })
    
    # Skalakan fitur
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)
    
    st.write(f'Predicted Fare: ${prediction[0]:.2f}')
