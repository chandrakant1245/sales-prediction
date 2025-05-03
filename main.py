import streamlit as st
import pickle 
import pandas as pd
import sklearn
import numpy as np
from xgboost import XGBRegressor

st.title("Big Mart Sales Prediction")

Item_Identifier = st.text_input("Item Identifier")
Item_weight = st.number_input("Item Weight", min_value=0.0, format="%.2f")
Item_Fat_Content = st.selectbox("Item Fat Content", ["Low Fat", "Regular", "Non-Edible"])  # Add valid options as needed
Item_visibility = st.number_input("Item Visibility", min_value=0.0, max_value=1.0, format="%.4f")
Item_Type = st.selectbox("Item Type", ["Dairy", "Soft Drinks", "Meat", "Fruits and Vegetables", "Household", "Baking Goods", "Snack Foods", "Frozen Foods", "Breakfast", "Health and Hygiene", "Hard Drinks", "Canned", "Breads", "Starchy Foods", "Others", "Seafood"])  # Replace with real categories
Item_MRP = st.number_input("Item MRP", min_value=0.0, format="%.2f")

Outlet_identifier = st.text_input("Outlet Identifier")
Outlet_established_year = st.number_input("Outlet Established Year", min_value=1900, max_value=2025, step=1)
Outlet_size = st.selectbox("Outlet Size", ["Small", "Medium", "High"])
Outlet_location_type = st.selectbox("Outlet Location Type", ["Tier 1", "Tier 2", "Tier 3"])
Outlet_type = st.selectbox("Outlet Type", ["Supermarket Type1", "Supermarket Type2", "Supermarket Type3", "Grocery Store"])



with open("./model.pkl","br") as path:
    model = pickle.load(path)

with open("./onehot_encoder.pkl","br") as path:
    encoder = pickle.load(path)

if st.button("Predict"):

    input_df = pd.DataFrame([{
        "Item_Identifier": Item_Identifier,
        "Item_Weight": Item_weight,
        "Item_Fat_Content": Item_Fat_Content,
        "Item_Visibility": Item_visibility,
        "Item_Type": Item_Type,
        "Item_MRP": Item_MRP,
        "Outlet_Identifier": Outlet_identifier,
        "Outlet_Establishment_Year": Outlet_established_year,
        "Outlet_Size": Outlet_size,
        "Outlet_Location_Type": Outlet_location_type,
        "Outlet_Type": Outlet_type
    }])

    input_df = encoder.transform(input_df)


    # Predict
    prediction = model.predict(input_df)

    st.success(f"Predicted Sales: {prediction[0]:.2f}")

