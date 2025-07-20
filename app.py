import streamlit as st
import numpy as np
import pickle
import pandas as pd

st.set_page_config(page_title="House Price Estimator 🏠", layout="wide")

# Background styling
st.markdown(
    """
    <style>
    .stApp {
        background-image: url('https://images.unsplash.com/photo-1564013799919-ab600027ffc6');
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        opacity: 0.95;
    }
    .stApp h1, .stApp h2, .stApp h3 {
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Load model
with open("house_model.pkl", "rb") as f:
    model_pipeline, features, cat_features = pickle.load(f)

# Dropdown values
MSZoning_values = ['RL', 'RM', 'C (all)', 'FV', 'RH']
Neighborhood_values = ['CollgCr', 'Veenker', 'Crawfor', 'NoRidge', 'Mitchel', 'Somerst', 'NWAmes',
                       'OldTown', 'BrkSide', 'Sawyer', 'NridgHt', 'NAmes', 'SawyerW', 'IDOTRR',
                       'MeadowV', 'Edwards', 'Timber', 'Gilbert', 'StoneBr', 'ClearCr', 'NPkVill',
                       'Blmngtn', 'BrDale', 'SWISU', 'Blueste']
HouseStyle_values = ['1Story', '2Story', '1.5Fin', '1.5Unf', 'SFoyer', 'SLvl', '2.5Unf', '2.5Fin']

# Title
st.title("🏡 House Price Estimator")
st.markdown("#### Enter the house details to predict its market price")

with st.form("input_form"):
    col1, col2 = st.columns(2)

    with col1:
        OverallQual = st.slider("Overall Quality (1-10)", 1, 10, 5)
        GrLivArea = st.number_input("Above ground living area (sq ft)", value=1500)
        GarageCars = st.slider("Garage capacity (cars)", 0, 5, 2)
        GarageArea = st.number_input("Garage area (sq ft)", value=400)
    
    with col2:
        TotalBsmtSF = st.number_input("Total basement area (sq ft)", value=800)
        FullBath = st.slider("Full bathrooms", 0, 4, 2)
        YearBuilt = st.slider("Year Built", 1900, 2025, 2000)
    
    st.markdown("---")
    col3, col4, col5 = st.columns(3)

    with col3:
        MSZoning = st.selectbox("Zoning", MSZoning_values)
    with col4:
        Neighborhood = st.selectbox("Neighborhood", Neighborhood_values)
    with col5:
        HouseStyle = st.selectbox("House Style", HouseStyle_values)

    submitted = st.form_submit_button("💵 Predict Price")

    if submitted:
        input_data = pd.DataFrame([{
            'OverallQual': OverallQual,
            'GrLivArea': GrLivArea,
            'GarageCars': GarageCars,
            'GarageArea': GarageArea,
            'TotalBsmtSF': TotalBsmtSF,
            'FullBath': FullBath,
            'YearBuilt': YearBuilt,
            'MSZoning': MSZoning,
            'Neighborhood': Neighborhood,
            'HouseStyle': HouseStyle
        }])

        prediction = model_pipeline.predict(input_data)[0]

        st.success(f"💰 Estimated Sale Price: ₹ {prediction:,.0f}")
