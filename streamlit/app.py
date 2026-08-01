import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import sys
from PIL import Image

# Add src to path to import predict module
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))
from predict import make_prediction

# Set page config
st.set_page_config(page_title="Predictive Maintenance Dashboard", layout="wide")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select Page", ["Dashboard", "Prediction", "Explainability", "Analytics"])

# Load data and models
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
engineered_data_path = os.path.join(base_dir, "data/processed/engineered_data.csv")
shap_summary_path = os.path.join(base_dir, "models/shap_summary_plot.png")
shap_waterfall_path = os.path.join(base_dir, "models/shap_waterfall_plot_instance_0.png")

df = pd.read_csv(engineered_data_path)

# Page 1: Dashboard
if page == "Dashboard":
    st.title("Predictive Maintenance Dashboard")
    st.markdown("---")
    
    # Machine Health Overview
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Machines", len(df))
    with col2:
        failure_count = (df['Machine_failure'] == 1).sum()
        st.metric("Failures Detected", failure_count)
    with col3:
        failure_rate = (failure_count / len(df)) * 100
        st.metric("Failure Rate (%)", f"{failure_rate:.2f}")
    with col4:
        st.metric("Healthy Machines", len(df) - failure_count)
    
    st.markdown("---")
    
    # Failure Distribution
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Failure Distribution")
        failure_counts = df['Machine_failure'].value_counts()
        fig, ax = plt.subplots()
        ax.bar(['No Failure', 'Failure'], [failure_counts.get(0, 0), failure_counts.get(1, 0)], color=['green', 'red'])
        ax.set_ylabel("Count")
        st.pyplot(fig)
    
    with col2:
        st.subheader("Machine Type Distribution")
        type_counts = df['Type'].value_counts()
        fig, ax = plt.subplots()
        ax.pie(type_counts.values, labels=[f'Type {t}' for t in type_counts.index], autopct='%1.1f%%')
        st.pyplot(fig)
    
    st.markdown("---")
    
    # Top Risky Machines (based on temperature and torque)
    st.subheader("Top Risky Machines")
    df['Risk_Score'] = df['Torque_Nm'] * df['Process_temperature_K']
    top_risky = df.nlargest(5, 'Risk_Score')[['Type', 'Air_temperature_K', 'Process_temperature_K', 'Torque_Nm', 'Risk_Score']]
    st.dataframe(top_risky)

# Page 2: Prediction
elif page == "Prediction":
    st.title("Machine Failure Prediction")
    st.markdown("---")
    
    st.subheader("Enter Machine Parameters")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        machine_type = st.selectbox("Machine Type", ["L", "M", "H"])
        air_temp = st.number_input("Air Temperature (K)", value=298.0, min_value=290.0, max_value=310.0)
        torque = st.number_input("Torque (Nm)", value=42.0, min_value=0.0, max_value=100.0)
    
    with col2:
        process_temp = st.number_input("Process Temperature (K)", value=308.0, min_value=300.0, max_value=320.0)
        speed = st.number_input("Rotational Speed (rpm)", value=1500, min_value=1000, max_value=2500)
    
    with col3:
        tool_wear = st.number_input("Tool Wear (min)", value=180, min_value=0, max_value=300)
    
    if st.button("Predict"):
        prediction_data = {
            "Type": machine_type,
            "Air_temperature_K": air_temp,
            "Process_temperature_K": process_temp,
            "Rotational_speed_rpm": speed,
            "Torque_Nm": torque,
            "Tool_wear_min": tool_wear
        }
        result = make_prediction(prediction_data)
        
        st.markdown("---")
        st.subheader("Prediction Result")
        
        col1, col2 = st.columns(2)
        with col1:
            probability = result['failure_probability']
            prediction = result['prediction']
            
            # Display prediction with color
            if prediction == "Failure":
                st.error(f"⚠️ **{prediction}**")
                st.metric("Failure Probability", f"{probability*100:.2f}%")
            else:
                st.success(f"✅ **{prediction}**")
                st.metric("Failure Probability", f"{probability*100:.2f}%")
        
        with col2:
            # Gauge chart
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.barh(["Failure Risk"], [probability], color='red' if probability > 0.5 else 'green')
            ax.set_xlim(0, 1)
            ax.set_xlabel("Probability")
            st.pyplot(fig)

# Page 3: Explainability
elif page == "Explainability":
    st.title("Model Explainability (SHAP)")
    st.markdown("---")
    
    st.subheader("Global Feature Importance")
    if os.path.exists(shap_summary_path):
        st.image(shap_summary_path, caption="SHAP Summary Plot - Global Feature Importance")
    else:
        st.warning("SHAP summary plot not found. Please run explain.py first.")
    
    st.markdown("---")
    
    st.subheader("Instance-Level Explanation (Waterfall Plot)")
    if os.path.exists(shap_waterfall_path):
        st.image(shap_waterfall_path, caption="SHAP Waterfall Plot - Instance 0 Explanation")
    else:
        st.warning("SHAP waterfall plot not found. Please run explain.py first.")
    
    st.markdown("---")
    
    st.subheader("What do these plots mean?")
    st.markdown("""
    - **SHAP Summary Plot**: Shows which features have the most impact on the model's predictions. Red indicates high feature values pushing predictions toward failure, blue indicates low values.
    - **SHAP Waterfall Plot**: Shows how individual feature values contribute to the model's prediction for a specific instance. The base value is the average prediction, and each feature either increases or decreases the prediction.
    """)

# Page 4: Analytics
elif page == "Analytics":
    st.title("Detailed Analytics")
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Temperature Distribution")
        fig, ax = plt.subplots()
        ax.hist(df['Air_temperature_K'], bins=30, alpha=0.7, label='Air Temp', color='blue')
        ax.hist(df['Process_temperature_K'], bins=30, alpha=0.7, label='Process Temp', color='red')
        ax.set_xlabel("Temperature (K)")
        ax.set_ylabel("Frequency")
        ax.legend()
        st.pyplot(fig)
    
    with col2:
        st.subheader("Torque Distribution")
        fig, ax = plt.subplots()
        ax.hist(df['Torque_Nm'], bins=30, color='green', alpha=0.7)
        ax.set_xlabel("Torque (Nm)")
        ax.set_ylabel("Frequency")
        st.pyplot(fig)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Rotational Speed Distribution")
        fig, ax = plt.subplots()
        ax.hist(df['Rotational_speed_rpm'], bins=30, color='orange', alpha=0.7)
        ax.set_xlabel("Rotational Speed (rpm)")
        ax.set_ylabel("Frequency")
        st.pyplot(fig)
    
    with col2:
        st.subheader("Tool Wear Distribution")
        fig, ax = plt.subplots()
        ax.hist(df['Tool_wear_min'], bins=30, color='purple', alpha=0.7)
        ax.set_xlabel("Tool Wear (min)")
        ax.set_ylabel("Frequency")
        st.pyplot(fig)
    
    st.markdown("---")
    
    st.subheader("Correlation Matrix")
    correlation_cols = ['Air_temperature_K', 'Process_temperature_K', 'Rotational_speed_rpm', 'Torque_Nm', 'Tool_wear_min', 'Machine_failure']
    corr_matrix = df[correlation_cols].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', ax=ax)
    st.pyplot(fig)

st.markdown("---")
st.markdown("**Predictive Maintenance System** | Built with Streamlit, XGBoost, and SHAP")
