import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

# Memuat model dan scaler
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Judul aplikasi
st.title('Prediksi Harga/Tarif Uber')

# Input pengguna
pickup_longitude = st.number_input('Pickup Longitude', value=0.0)
pickup_latitude = st.number_input('Pickup Latitude', value=0.0)
dropoff_longitude = st.number_input('Dropoff Longitude', value=0.0)
dropoff_latitude = st.number_input('Dropoff Latitude', value=0.0)
passenger_count = st.number_input('Passenger Count', value=1)

# Membuat dataframe dari input pengguna
input_data = pd.DataFrame({
    'pickup_longitude': [pickup_longitude],
    'pickup_latitude': [pickup_latitude],
    'dropoff_longitude': [dropoff_longitude],
    'dropoff_latitude': [dropoff_latitude],
    'passenger_count': [passenger_count]
})

# Pra-pemrosesan input pengguna
input_data_scaled = scaler.transform(input_data)

# Prediksi
if st.button('Prediksi Tarif'):
    prediction = model.predict(input_data_scaled)
    st.write(f'Prediksi Tarif: ${prediction[0]:.2f}')
