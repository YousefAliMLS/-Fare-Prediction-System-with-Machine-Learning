import pandas as pd
import numpy as np
import warnings
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

data_file = pd.read_csv("final_internship_data.csv")
data_file.columns = [col.strip().lower().replace(" ", "_") for col in data_file.columns]
data_file.drop(columns=['user_id', 'user_name', 'driver_name', 'key'], inplace=True, errors='ignore')

if 'pickup_datetime' in data_file.columns:
    data_file['pickup_datetime'] = pd.to_datetime(data_file['pickup_datetime'])
    data_file['hour'] = data_file['pickup_datetime'].dt.hour
    data_file['day'] = data_file['pickup_datetime'].dt.day
    data_file['month'] = data_file['pickup_datetime'].dt.month
    data_file['weekday'] = data_file['pickup_datetime'].dt.weekday
    data_file['year'] = data_file['pickup_datetime'].dt.year
    data_file.drop(columns=['pickup_datetime'], inplace=True)

data_file['is_weekend'] = data_file['weekday'].isin([5, 6]).astype(int)
data_file['rush_hours'] = data_file['hour'].between(7, 10) | data_file['hour'].between(16, 19)
data_file['rush_hours'] = data_file['rush_hours'].astype(int)

num_cols = data_file.select_dtypes(include=['int64', 'float64']).columns
for col in num_cols:
    data_file[col].fillna(data_file[col].median(), inplace=True)

cat_cols = data_file.select_dtypes(include='object').columns
for col in cat_cols:
    data_file[col].fillna(data_file[col].mode()[0], inplace=True)

data_file = pd.get_dummies(data_file, columns=['car_condition', 'weather', 'traffic_condition'], drop_first=True)

X = data_file.drop(columns=['fare_amount'])
y = data_file['fare_amount']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

with open('model_features.pkl', 'wb') as f:
    pickle.dump(list(X.columns), f)

X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
model.fit(X_train, Y_train)

preds = model.predict(X_test)
r2 = r2_score(Y_test, preds)
mae = mean_absolute_error(Y_test, preds)
accuracy = 100 * (1 - (mae / y.mean()))

print(f"R²: {r2:.4f}")
print(f"MAE: {mae:.2f}")
print(f"Accuracy: {accuracy:.2f}%")

with open('fare_model.pkl', 'wb') as f:
    pickle.dump(model, f)
