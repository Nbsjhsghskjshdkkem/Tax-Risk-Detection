import streamlit as st
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Set page configuration
st.set_page_config(page_title="Tax Risk Detection System", layout="wide")

@st.cache_data
def load_and_prep_data():
    # Load the dataset
    df = pd.read_csv("tax_risk_dataset.csv")
    
    # Use separate encoders for different categorical columns
    le_industry = LabelEncoder()
    le_risk = LabelEncoder()
    
    df['Industry'] = le_industry.fit_transform(df['Industry'])
    df['Risk_Label'] = le_risk.fit_transform(df['Risk_Label'])
    
    # Feature selection (Excluding ID and Label)
    X = df.drop(columns=['Taxpayer_ID', 'Risk_Label'])
    y = df['Risk_Label']
    
    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train the model
    dt = DecisionTreeClassifier(random_state=42)
    dt.fit(X_scaled, y)
    
    return dt, scaler, le_industry, le_risk, X.columns

# Initialize model and tools
model, scaler, le_industry, le_risk, feature_names = load_and_prep_data()

# App UI
st.title("📊 Tax Risk Detection System")
st.markdown("""
This application predicts the risk level of a taxpayer based on financial and compliance data.
""")

# Sidebar for User Inputs
st.sidebar.header("Input Taxpayer Data")

def get_user_input():
    inputs = {}
    inputs['Revenue'] = st.sidebar.number_input("Revenue", min_value=0.0, value=1000000.0)
    inputs['Expenses'] = st.sidebar.number_input("Expenses", min_value=0.0, value=800000.0)
    inputs['Tax_Liability'] = st.sidebar.number_input("Tax Liability", min_value=0.0, value=40000.0)
    inputs['Tax_Paid'] = st.sidebar.number_input("Tax Paid", min_value=0.0, value=35000.0)
    inputs['Late_Filings'] = st.sidebar.slider("Late Filings", 0, 10, 2)
    inputs['Compliance_Violations'] = st.sidebar.slider("Compliance Violations", 0, 10, 1)
    
    # Industry Selection using the industry-specific encoder classes
    industry_options = list(le_industry.classes_)
    selected_industry = st.sidebar.selectbox("Industry", industry_options)
    inputs['Industry'] = le_industry.transform([selected_industry])[0]
    
    # Calculated Fields logic
    inputs['Profit'] = inputs['Revenue'] - inputs['Expenses']
    inputs['Tax_Compliance_Ratio'] = inputs['Tax_Paid'] / inputs['Tax_Liability'] if inputs['Tax_Liability'] != 0 else 0
    inputs['Audit_Findings'] = st.sidebar.slider("Audit Findings", 0, 20, 0)
    inputs['Audit_to_Tax_Ratio'] = inputs['Audit_Findings'] / inputs['Tax_Liability'] if inputs['Tax_Liability'] != 0 else 0
    
    # Ensure the dataframe has columns in the correct order for the model
    return pd.DataFrame([inputs])[feature_names]

user_df = get_user_input()

# Display Input Data
st.subheader("Current Taxpayer Profile")
st.write(user_df)

# Prediction Logic
if st.button("Analyze Risk"):
    # Scale the input data
    user_scaled = scaler.transform(user_df)
    
    # Predict and decode using the risk encoder
    prediction_idx = model.predict(user_scaled)[0]
    prediction_label = le_risk.inverse_transform([prediction_idx])[0]
    
    # Visual Output
    st.divider()
    if prediction_label == 'High':
        st.error(f"### Result: {prediction_label} Risk")
    elif prediction_label == 'Medium':
        st.warning(f"### Result: {prediction_label} Risk")
    else:
        st.success(f"### Result: {prediction_label} Risk")
        
    st.info("Risk assessment is based on compliance ratios, late filings, and historical audit findings.")

# Footer
st.sidebar.markdown("---")
st.sidebar.write("Developed based on Tax Risk Detection Analysis.")