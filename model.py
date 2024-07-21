import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR, LinearSVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import pickle
import matplotlib.pyplot as plt

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

# Menyimpan scaler
with open('scaler.pkl', 'wb') as file:
    pickle.dump(scaler, file)

# Pelatihan Model
models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'SVR': SVR(C=0.1, epsilon=0.2, gamma=0.01),
    'Linear SVR': LinearSVR(C=1.0, max_iter=10000, random_state=42)
}

best_model_name = ''
best_rmse = float('inf')

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    print(f'{name}:')
    print('MAE:', mean_absolute_error(y_test, y_pred))
    print('MSE:', mean_squared_error(y_test, y_pred))
    print('RMSE:', rmse)
    print('R2 Score:', r2_score(y_test, y_pred))
    print('----------------------')
    
    if rmse < best_rmse:
        best_rmse = rmse
        best_model_name = name

best_model = models[best_model_name]

# Menyimpan model terbaik
with open('model.pkl', 'wb') as file:
    pickle.dump(best_model, file)

print(f'Model dengan performa terbaik adalah: {best_model_name} dengan RMSE: {best_rmse}')

# Visualisasi kinerja model
model_names = list(models.keys())
mae_scores = [mean_absolute_error(y_test, models[name].predict(X_test)) for name in model_names]
mse_scores = [mean_squared_error(y_test, models[name].predict(X_test)) for name in model_names]
rmse_scores = [np.sqrt(mean_squared_error(y_test, models[name].predict(X_test))) for name in model_names]
r2_scores = [r2_score(y_test, models[name].predict(X_test)) for name in model_names]

x = np.arange(len(model_names))

fig, axs = plt.subplots(2, 2, figsize=(12, 12))
fig.suptitle('Perbandingan Kinerja Model')

# MAE
axs[0, 0].bar(x, mae_scores, color='b')
axs[0, 0].set_xticks(x)
axs[0, 0].set_xticklabels(model_names)
axs[0, 0].set_title('Mean Absolute Error (MAE)')

# MSE
axs[0, 1].bar(x, mse_scores, color='r')
axs[0, 1].set_xticks(x)
axs[0, 1].set_xticklabels(model_names)
axs[0, 1].set_title('Mean Squared Error (MSE)')

# RMSE
axs[1, 0].bar(x, rmse_scores, color='g')
axs[1, 0].set_xticks(x)
axs[1, 0].set_xticklabels(model_names)
axs[1, 0].set_title('Root Mean Squared Error (RMSE)')

# R2 Score
axs[1, 1].bar(x, r2_scores, color='y')
axs[1, 1].set_xticks(x)
axs[1, 1].set_xticklabels(model_names)
axs[1, 1].set_title('R2 Score')

plt.show()
