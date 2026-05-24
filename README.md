# Air Pollution Prediction System using Machine Learning

## Project Overview

This project develops a machine learning-based predictive analytics system to forecast air pollution levels (NO₂) using environmental data. The system integrates data processing, model training, evaluation, and visualization within an interactive dashboard.

The project follows the CRISP-DM methodology, covering data understanding, preparation, modelling, evaluation, and deployment via an interactive dashboard.

The model provides insights into pollution trends and supports environmental monitoring and decision-making.

## Features

* Environmental data preprocessing and cleaning
* Feature engineering for time-series data
* Random Forest regression model for prediction
* Model evaluation using MAE, RMSE, and R²
* Interactive dashboard for visualization (Streamlit)
* Anomaly detection and spatial mapping

## Dataset

* Source: UK Defra Air Quality Data (Sunderland Silksworth station)
* Features include:
  * NO₂ concentration
  * Ozone levels
  * PM10 and PM2.5 particulate matter
  * Timestamp (daily measurements)
* Data type: Time-series environmental data
* File: `Sunderland Silksworth Air Pollution Datasets 2023 (Daily monitoring).csv`

## Project Structure

```
air-pollution-project/
│
├── Sunderland Silksworth Air Pollution Datasets 2023 (Daily monitoring).csv
├── app.py                           # Streamlit dashboard application
├── airpollution.py                  # Offline data processing and model training script
├── save_model.py                    # Script to train and save the ML model
├── requirements.txt.txt             # Python dependencies list
├── README.md                        # Project documentation
└── .venv/                           # Virtual environment (optional)
```

## Installation

1. Ensure Python 3.8+ is installed on your system.

2. Clone or download the project files to your local machine.

3. Navigate to the project directory:

   ```bash
   cd "c:\Users\excellen\OneDrive\Desktop\Air Pollution Datasets"
   ```

4. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

   Or, if using a virtual environment:

   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # On Windows
   pip install -r requirements.txt.txt
   ```

## How to Run (Local Setup)

1. Ensure Python is installed
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the app:

   ```bash
   streamlit run app.py
   ```

4. Open browser at:
   http://localhost:8501

### Alternative: Run Offline Scripts

- To train the model and generate plots without the dashboard:

  ```bash
  python airpollution.py
  ```

- To train and save the model only:

  ```bash
  python save_model.py
  ```

## Model Used

* **Random Forest Regressor**

The model was selected for its ability to capture complex nonlinear relationships in environmental data, handle missing values well, and provide feature importance insights.

### Model Configuration
- **Features**: Ozone, PM10, PM2.5, Month, DayOfWeek, DayOfYear
- **Target**: NO₂ concentration
- **Hyperparameters**: 100 estimators, random state 42

## Results

* **MAE**: 2.19
* **RMSE**: 3.11
* **R² Score**: 0.74

These results indicate good predictive performance for environmental forecasting, with the model explaining 74% of the variance in NO₂ levels.

## Additional Notes

- The dashboard includes 7 interactive tabs for comprehensive data exploration
- Anomaly detection uses Z-score thresholding (>2.5 standard deviations)
- SHAP values are computed for model explainability
- All visualizations are built with Plotly for interactivity

## Troubleshooting

- If the dashboard fails to load, ensure all dependencies are installed and the CSV file is in the project directory
- For path-related errors, the scripts automatically detect the CSV file location
- If Streamlit doesn't open automatically, manually navigate to `http://localhost:8501` in your browser

## Future Improvements

- Deploy dashboard to cloud
- Add real-time air quality API integration
- Implement deep learning models
- Extend monitoring to multiple UK stations