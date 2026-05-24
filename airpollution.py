import pandas as pd # type: ignore
import numpy as np # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.ensemble import RandomForestRegressor # type: ignore
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score # type: ignore
import matplotlib.pyplot as plt # type: ignore
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = next((p for p in BASE_DIR.glob('*.csv') if 'Sunderland' in p.name and 'Silksworth' in p.name), None)
if CSV_PATH is None:
    raise FileNotFoundError(f"No Sunderland Silksworth CSV found in {BASE_DIR}")
file_path = CSV_PATH

def load_and_clean_data(path):
    print(f"Loading data from {path}...")
    # Skip the first 7 lines of metadata
    df = pd.read_csv(path, skiprows=7)
    
    # Rename columns for easier access
    # Original headers: Date, Nitrogen dioxide, Status, Ozone, Status, PM10..., Status, PM2.5..., Status
    new_columns = [
        'Date', 
        'NO2', 'NO2_Status', 
        'Ozone', 'Ozone_Status', 
        'PM10', 'PM10_Status', 
        'PM2.5', 'PM2.5_Status'
    ]
    df.columns = new_columns
    
    # Convert 'no data' to NaN
    df.replace('no data', np.nan, inplace=True)
    
    # Convert pollutant columns to numeric
    pollutant_cols = ['NO2', 'Ozone', 'PM10', 'PM2.5']
    for col in pollutant_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Convert Date to datetime
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=False)
    
    # Interpolate missing values (time-based linear interpolation)
    df = df.sort_values('Date')
    df.set_index('Date', inplace=True)
    df[pollutant_cols] = df[pollutant_cols].interpolate(method='time')
    
    # Fill remaining NaNs (at the beginning of the series) with backfill/forwardfill
    df[pollutant_cols] = df[pollutant_cols].ffill().bfill()
    
    # Reset index to bring Date back as a column for feature engineering
    df.reset_index(inplace=True)
    
    return df

def feature_engineering(df):
    print("Performing feature engineering...")
    df['Month'] = df['Date'].dt.month
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['DayOfYear'] = df['Date'].dt.dayofyear
    return df

def train_and_evaluate(df):
    print("Training model...")
    # Features: Ozone, PM10, PM2.5, Month, DayOfWeek, DayOfYear
    # Target: NO2
    features = ['Ozone', 'PM10', 'PM2.5', 'Month', 'DayOfWeek', 'DayOfYear']
    target = 'NO2'
    
    X = df[features]
    y = df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print("\nModel Evaluation Metrics:")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"R² Score: {r2:.4f}")
    
    return model, X_test, y_test, y_pred

def plot_results(y_test, y_pred):
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual NO2')
    plt.ylabel('Predicted NO2')
    plt.title('Actual vs Predicted Nitrogen Dioxide (NO2) Levels')
    
    # Ensure directory exists for saving plot
    plot_dir = BASE_DIR
    output_path = plot_dir / "actual_vs_predicted_no2.png"
    plt.savefig(output_path)
    print(f"\nPlot saved to {output_path}")

if __name__ == "__main__":
    try:
        clean_df = load_and_clean_data(file_path)
        featured_df = feature_engineering(clean_df)
        model, X_test, y_test, y_pred = train_and_evaluate(featured_df)
        plot_results(y_test, y_pred)
    except Exception as e:
        print(f"An error occurred: {e}")
