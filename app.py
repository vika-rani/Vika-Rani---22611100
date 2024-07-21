import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

# Memuat model, scaler, dan fitur
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

with open('features.pkl', 'rb') as file:
    features = pickle.load(file)

# Judul aplikasi
st.title('Prediksi Harga/Tarif Uber')

# Input pengguna
pickup_longitude = st.number_input('Pickup Longitude', value=0.0)
pickup_latitude = st.number_input('Pickup Latitude', value=0.0)
dropoff_longitude = st.number_input('Dropoff Longitude', value=0.0)
dropoff_latitude = st.number_input('Dropoff Latitude', value=0.0)
passenger_count = st.number_input('Passenger Count', value=1)

# Input waktu
year = st.number_input('Year', value=2021)
month = st.number_input('Month', value=1, min_value=1, max_value=12)
day = st.number_input('Day', value=1, min_value=1, max_value=31)
hour = st.number_input('Hour', value=0, min_value=0, max_value=23)
minute = st.number_input('Minute', value=0, min_value=0, max_value=59)
second = st.number_input('Second', value=0, min_value=0, max_value=59)

# Membuat dataframe dari input pengguna
input_data = pd.DataFrame({
    'pickup_longitude': [pickup_longitude],
    'pickup_latitude': [pickup_latitude],
    'dropoff_longitude': [dropoff_longitude],
    'dropoff_latitude': [dropoff_latitude],
    'passenger_count': [passenger_count],
    'year': [year],
    'month': [month],
    'day': [day],
    'hour': [hour],
    'minute': [minute],
    'second': [second]
})

# Menambahkan kolom yang mungkin hilang dengan nilai default 0
for feature in features:
    if feature not in input_data.columns:
        input_data[feature] = 0

# Urutkan kolom sesuai dengan fitur yang digunakan saat pelatihan
input_data = input_data[features]

# Pra-pemrosesan input pengguna
input_data_scaled = scaler.transform(input_data)

# Prediksi
if st.button('Prediksi Tarif'):
    prediction = model.predict(input_data_scaled)
    st.write(f'Prediksi Tarif: ${prediction[0]:.2f}')
