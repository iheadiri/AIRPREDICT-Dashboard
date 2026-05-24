import pandas as pd # type: ignore
import numpy as np # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.ensemble import RandomForestRegressor # type: ignore
import pickle
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
path = next((p for p in BASE_DIR.glob('*.csv') if 'Sunderland' in p.name and 'Silksworth' in p.name), None)
if path is None:
    raise FileNotFoundError(f"No Sunderland Silksworth CSV found in {BASE_DIR}")

print("Loading data...")
df = pd.read_csv(path, skiprows=7)
df.columns = ['Date', 'NO2', 'NO2_Status', 'Ozone', 'Ozone_Status', 'PM10', 'PM10_Status', 'PM2.5', 'PM2.5_Status']

df.replace('no data', np.nan, inplace=True)
pollutant_cols = ['NO2', 'Ozone', 'PM10', 'PM2.5']
for col in pollutant_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df['Date'] = pd.to_datetime(df['Date'], dayfirst=False)
df = df.sort_values('Date').set_index('Date')
df[pollutant_cols] = df[pollutant_cols].interpolate(method='time').ffill().bfill()
df = df.reset_index()

df['Month'] = df['Date'].dt.month
df['DayOfWeek'] = df['Date'].dt.dayofweek
df['DayOfYear'] = df['Date'].dt.dayofyear

features = ['Ozone', 'PM10', 'PM2.5', 'Month', 'DayOfWeek', 'DayOfYear']
X = df[features]
y = df['NO2']

print("Training model...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

model_path = BASE_DIR / "random_forest_no2_model.pkl"
print(f"Saving model to {model_path}...")
with open(model_path, 'wb') as f:
    pickle.dump(model, f)
    
print("Model saved successfully!")
