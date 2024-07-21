import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import pickle

# Memuat data
data = pd.read_csv('C:/Users/ASUS/my_ml_project/uber.csv')

# Pra-pemrosesan
data['pickup_datetime'] = pd.to_datetime(data['pickup_datetime'])
data['year'] = data['pickup_datetime'].dt.year
data['month'] = data['pickup_datetime'].dt.month
data['day'] = data['pickup_datetime'].dt.day
data['hour'] = data['pickup_datetime'].dt.hour
data['minute'] = data['pickup_datetime'].dt.minute
data['second'] = data['pickup_datetime'].dt.second
data = data.drop(columns=['pickup_datetime', 'key'])
data.fillna(data.mean(), inplace=True)

# Memisahkan fitur dan target
X = data.drop(columns=['fare_amount'])
y = data['fare_amount']

# Standardisasi
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Pelatihan Model
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluasi
print('MAE:', mean_absolute_error(y_test, y_pred))
print('MSE:', mean_squared_error(y_test, y_pred))
print('RMSE:', np.sqrt(mean_squared_error(y_test, y_pred)))
print('R2 Score:', r2_score(y_test, y_pred))

# Menyimpan Model dan Scaler
with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)

with open('scaler.pkl', 'wb') as file:
    pickle.dump(scaler, file)

# Simpan fitur yang digunakan selama pelatihan
with open('features.pkl', 'wb') as file:
    pickle.dump(X.columns.tolist(), file)
