import streamlit as st # type: ignore
import pandas as pd # type: ignore
import numpy as np # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.ensemble import RandomForestRegressor # type: ignore
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score # type: ignore
import plotly.express as px # type: ignore
import plotly.graph_objects as go # type: ignore
import plotly.figure_factory as ff # type: ignore
import shap # type: ignore
import matplotlib.pyplot as plt # type: ignore
from pathlib import Path

# ----------------- UI / CONFIGURATION -----------------
st.set_page_config(page_title="Advanced Air Pollution Analytics", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    .main { background-color: #f4f6f9; }
    .stMetric { 
        background-color: #ffffff !important; 
        padding: 20px; 
        border-radius: 12px; 
        border: 1px solid #cce5ff !important;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
    }
    [data-testid="stMetricValue"] {
        color: #007bff !important; 
    }
    [data-testid="stMetricLabel"] p {
        color: #333333 !important;
    }
    h1, h2, h3 { color: #0056b3 !important; font-family: 'Inter', sans-serif; }
</style>
""", unsafe_allow_html=True)

BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = next((p for p in BASE_DIR.glob("*.csv") if "Sunderland" in p.name and "Silksworth" in p.name), None)
if CSV_PATH is None:
    raise FileNotFoundError(f"No Sunderland Silksworth CSV found in {BASE_DIR}")
FILE_PATH = str(CSV_PATH)

# ----------------- DATA PIPELINE -----------------
@st.cache_data
def load_and_clean_data(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        site_name = lines[1].split(',')[1].strip()
        lat = float(lines[4].split(',')[1].strip())
        lon = float(lines[5].split(',')[1].strip())
    
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
    
    z_scores = (df['NO2'] - df['NO2'].mean()) / df['NO2'].std()
    df['is_anomaly'] = np.abs(z_scores) > 2.5
    
    return df, {'site_name': site_name, 'lat': lat, 'lon': lon}

st.title("🌌 Air Pollution Neural-Visual Dashboard")
st.markdown("##### High-Impact Environmental Predictive Analytics System")

try:
    df, meta = load_and_clean_data(FILE_PATH)
    
    with st.sidebar:
        st.header("📍 Station Info")
        st.info(f"**Site:** {meta['site_name']}\n\n**Coordinates:** {meta['lat']}, {meta['lon']}")
        st.divider()
        st.header("⚙️ Model Settings")
        n_est = st.slider("Random Forest Estimators", 10, 200, 100)
        test_size = st.slider("Test Size (%)", 10, 40, 20) / 100

    # ----------------- MACHINE LEARNING -----------------
    features = ['Ozone', 'PM10', 'PM2.5', 'Month', 'DayOfWeek', 'DayOfYear']
    X, y = df[features], df['NO2']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    model = RandomForestRegressor(n_estimators=n_est, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("MAE", f"{mean_absolute_error(y_test, y_pred):.2f}")
    m2.metric("RMSE", f"{np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
    m3.metric("R² Score", f"{r2_score(y_test, y_pred):.3f}")
    m4.metric("Anomalies", f"{df['is_anomaly'].sum()}")

    # ----------------- PROTOTYPE TABS -----------------
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(["🚀 Model Precision", "📊 Statistical Analysis", "🛡️ Anomaly Tracking", "🌍 Spatial Map", "🧠 Explainable AI (SHAP)", "📋 Training Plan", "🧪 Advanced Visuals", "🔮 Future Improvements"])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Predicted vs Actual (NO2)")
            fig = px.scatter(x=y_test, y=y_pred, template="plotly_white", trendline="ols", color_discrete_sequence=['#007bff'])
            st.plotly_chart(fig, width='stretch')
        with c2:
            st.subheader("Residual Plot")
            residuals = y_test - y_pred
            fig_res = px.scatter(x=y_pred, y=residuals, template="plotly_white", color_discrete_sequence=['#ff4b4b'])
            fig_res.add_hline(y=0, line_dash="dash", line_color="#333333")
            st.plotly_chart(fig_res, width='stretch')

    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Correlation Heatmap")
            corr = df[['NO2', 'Ozone', 'PM10', 'PM2.5']].corr()
            fig_hm = px.imshow(corr, text_auto=".2f", aspect="auto", template="plotly_white", color_continuous_scale='RdBu_r')
            st.plotly_chart(fig_hm, width='stretch')
        with c2:
            st.subheader("NO2 Distribution")
            fig_dist = px.histogram(df, x="NO2", marginal="box", template="plotly_white", color_discrete_sequence=['#0056b3'])
            st.plotly_chart(fig_dist, width='stretch')

    with tab3:
        st.subheader("Automated Anomaly Detection")
        fig_anom = go.Figure()
        fig_anom.add_trace(go.Scatter(x=df['Date'], y=df['NO2'], mode='lines', name='Baseline Trend', line=dict(color='#cccccc')))
        anom_points = df[df['is_anomaly']]
        # Fix: using direct dict initialization instead of dict() to prevent Pyre type errors
        fig_anom.add_trace(go.Scatter(x=anom_points['Date'], y=anom_points['NO2'], mode='markers', name='Anomaly Event', marker={'color': '#ff4b4b', 'size': 10, 'symbol': 'x'}))
        fig_anom.update_layout(template="plotly_white")
        st.plotly_chart(fig_anom, width='stretch')

    with tab4:
        st.subheader("Geographical High-Impact Map")
        map_data = pd.DataFrame({'lat': [meta['lat']], 'lon': [meta['lon']], 'Site': [meta['site_name']], 'Avg NO2': [df['NO2'].mean()]})
        fig_map = px.scatter_map(map_data, lat="lat", lon="lon", size="Avg NO2", zoom=12, height=600, color_discrete_sequence=["#007bff"])
        fig_map.update_layout(mapbox_style="carto-positron", margin={"r":0,"t":0,"l":0,"b":0})
        st.plotly_chart(fig_map, width='stretch')

    with tab5:
        st.subheader("SHAP (SHapley Additive exPlanations)")
        with st.spinner("Computing SHAP values..."):
            explainer = shap.TreeExplainer(model)
            X_sample = X_test.sample(min(100, len(X_test)))
            shap_values = explainer.shap_values(X_sample)
            
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Global Feature Importance**")
                fig_shap_bar, ax = plt.subplots(figsize=(8, 5))
                shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
                plt.gcf().set_facecolor('#ffffff')
                ax.set_facecolor('#ffffff'); ax.tick_params(colors='#333333'); ax.xaxis.label.set_color('#333333')
                st.pyplot(fig_shap_bar)
            with c2:
                st.markdown("**Feature Impact Distribution**")
                fig_shap_sum, ax2 = plt.subplots(figsize=(8, 5))
                shap.summary_plot(shap_values, X_sample, show=False)
                plt.gcf().set_facecolor('#ffffff')
                ax2.set_facecolor('#ffffff'); ax2.tick_params(colors='#333333'); ax2.xaxis.label.set_color('#333333')
                st.pyplot(fig_shap_sum)

    with tab6:
        st.subheader("📋 System Architecture & Workflow")
        st.write("This outlines the exact end-to-end pipeline used to build this predictive dashboard:")
        
        st.markdown("---")
        
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        
        with c1:
            st.info("📊 **1. Data**\n\nFetched real-world CSV records from Defra UK-AIR.")
        with c2:
            st.warning("🧹 **2. Clean**\n\nSanitized string gaps and executed time-series interpolation.")
        with c3:
            st.error("🤖 **3. Train Model**\n\nEngineered features and ran a Random Forest Regressor.")
        with c4:
            st.success("📏 **4. Evaluate**\n\nChecked accuracy using MAE, RMSE, and R² scores.")
        with c5:
            st.info("🎯 **5. Predict**\n\nForecasted unseen Nitrogen Dioxide (NO2) targets.")
        with c6:
            st.success("📈 **6. Show Results**\n\nIntegrated interactive Plotly visuals into this UI.")
            
        st.markdown("---")
        st.write("By structuring the codebase into this linear sequence, raw environmental metrics are systematically transformed into actionable intelligence via the dashboard.")

    with tab7:
        st.subheader("🧪 Advanced Exploratory Visualizations")
        st.write("Deeper cuts of the data isolating temporal seasonality and multi-dimensional correlations.")
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### NO2 Seasonal Volatility")
            fig_box = px.box(df, x="Month", y="NO2", template="plotly_white", points="all", color="Month", color_discrete_sequence=px.colors.sequential.Blues_r)
            fig_box.update_layout(showlegend=False)
            st.plotly_chart(fig_box, width='stretch')
        with c2:
            st.markdown("#### 3D Pollutant Clustering")
            fig_3d = px.scatter_3d(df, x='PM2.5', y='Ozone', z='NO2', color='Month', template="plotly_white", opacity=0.7)
            fig_3d.update_layout(margin=dict(l=0, r=0, b=0, t=0))
            st.plotly_chart(fig_3d, width='stretch')
            
        st.markdown("#### Holistic Pollutant Comparison Overlay")
        fig_multi = px.line(df, x='Date', y=['NO2', 'PM10', 'PM2.5', 'Ozone'], template="plotly_white")
        fig_multi.update_layout(legend_title_text='Pollutants')
        st.plotly_chart(fig_multi, width='stretch')

    with tab8:
        st.subheader("🔮 Future Improvements")
        st.write("Potential upgrades to scale and enhance this predictive dashboard:")
        
        st.markdown("""
        - ☁️ **Deploy dashboard to cloud:** Host the application on Streamlit Community Cloud or AWS for global public access.
        - 🔌 **Add real-time air quality API integration:** Connect to live Defra API endpoints for daily, automated dataset updates instead of static CSVs.
        - 🧠 **Implement deep learning models:** Transition from Random Forest to advanced neural networks like LSTM (Long Short-Term Memory) for superior time-series forecasting.
        - 🗺️ **Extend monitoring to multiple UK stations:** Scale the spatial map and dataset to include nationwide monitoring stations beyond just Sunderland Silksworth.
        """)

except Exception as e:
    st.error(f"System Error: {e}")

    # =========================================
# PRODUCT USE CASES
# =========================================

st.subheader("Product Use Cases and User Functionality")

st.markdown("""
The AIRPREDICT platform supports multiple stakeholders involved in environmental monitoring,
public health analysis, and smart city decision-making.
""")

with st.expander("UC1: Pollution Prediction"):
    st.write("""
    Actor: Environmental Officer

    Purpose:
    Predict pollutant concentrations using machine learning.

    Functionality:
    - Random Forest prediction
    - Single prediction input
    - Batch prediction upload
    - Real-time pollutant forecasting
    """)

with st.expander("UC2: Explainable AI Analysis"):
    st.write("""
    Actor: Public Health Analyst

    Purpose:
    Understand prediction explanations using SHAP.

    Functionality:
    - SHAP feature importance
    - Explainable AI visualization
    - Prediction transparency
    - Feature contribution analysis
    """)

with st.expander("UC3: Environmental Monitoring"):
    st.write("""
    Actor: Policy Maker

    Purpose:
    Monitor pollution trends and anomalies.

    Functionality:
    - Pollution trend analysis
    - Time-series visualization
    - Interactive dashboard monitoring
    - Environmental risk observation
    """)

    with st.expander("UC4: Anomaly Detection"):
        st.write("""
    Actor: Environmental Monitoring Agency

    Purpose:
    Detect abnormal pollution spikes and environmental risks.

    Functionality:
    - Z-score anomaly detection
    - Pollution spike tracking
    - Risk monitoring
    """)
    with st.expander("UC5: Dashboard Monitoring"):
        st.write("""
    Actor: Citizens and Public Health Stakeholders

    Purpose:
    Monitor environmental conditions interactively.

    Functionality:
    - Interactive charts
    - Pollution trends
    - Spatial analysis
    - Visual environmental insights
    """)

    st.info("""
AIRPREDICT is a smart-city environmental analytics platform designed to support
pollution prediction, Explainable AI analysis, and environmental decision-making.
""")

st.subheader("Project Development Timeline (Gantt Chart)")

# st.subheader("Project Development Timeline (Gantt Chart)")

# st.image(
#     "gantt_chart.png",
#     caption="Figure C1: AIRPREDICT Project Development Timeline",
#     width="stretch"
# )

st.markdown("""
The Gantt chart illustrates the iterative Agile development lifecycle used during AIRPREDICT implementation, including:
- Dataset acquisition
- Data preprocessing
- Machine learning model development
- Dashboard implementation
- SHAP integration
- Testing and refinement

The chart supported project scheduling, sprint monitoring, milestone tracking, task coordination, and development management efficiency.
""")

# st.subheader("Project Development Timeline (Gantt Chart)")

# st.image(
#     "gantt_chart.png",
#     caption="Figure C1: AIRPREDICT Project Development Timeline",
#     width="stretch"
# )