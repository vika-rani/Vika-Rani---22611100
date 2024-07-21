import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

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
models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Support Vector Regressor': SVR(C=1.0, epsilon=0.2, kernel='rbf')
}

results = {}

for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Evaluasi
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    results[model_name] = {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'R2 Score': r2
    }
    
    print(f'{model_name} Evaluation:')
    print('MAE:', mae)
    print('MSE:', mse)
    print('RMSE:', rmse)
    print('R2 Score:', r2)
    print('---')

# Memilih Model Terbaik (RMSE terendah)
best_model_name = min(results, key=lambda k: results[k]['RMSE'])
best_model = models[best_model_name]

print(f'Model dengan performa terbaik adalah: {best_model_name}')

# Menyimpan Model Terbaik
with open('model_best.pkl', 'wb') as file:
    pickle.dump(best_model, file)

# Visualisasi Perbandingan Kinerja Model
metrics = ['MAE', 'MSE', 'RMSE', 'R2 Score']
metrics_values = {metric: [] for metric in metrics}

for model_name in models:
    for metric in metrics:
        metrics_values[metric].append(results[model_name][metric])

x = np.arange(len(models))

fig, axs = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Perbandingan Kinerja Model', fontsize=16)

# MAE
axs[0, 0].bar(x, metrics_values['MAE'], color='b')
axs[0, 0].set_xticks(x)
axs[0, 0].set_xticklabels(models, rotation=45)
axs[0, 0].set_title('Mean Absolute Error (MAE)')
axs[0, 0].set_ylabel('Value')

# MSE
axs[0, 1].bar(x, metrics_values['MSE'], color='r')
axs[0, 1].set_xticks(x)
axs[0, 1].set_xticklabels(models, rotation=45)
axs[0, 1].set_title('Mean Squared Error (MSE)')
axs[0, 1].set_ylabel('Value')

# RMSE
axs[1, 0].bar(x, metrics_values['RMSE'], color='g')
axs[1, 0].set_xticks(x)
axs[1, 0].set_xticklabels(models, rotation=45)
axs[1, 0].set_title('Root Mean Squared Error (RMSE)')
axs[1, 0].set_ylabel('Value')

# R2 Score
axs[1, 1].bar(x, metrics_values['R2 Score'], color='y')
axs[1, 1].set_xticks(x)
axs[1, 1].set_xticklabels(models, rotation=45)
axs[1, 1].set_title('R2 Score')
axs[1, 1].set_ylabel('Value')

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
