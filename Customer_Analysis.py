import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

st.set_page_config(page_title="K-Means Clustering Demo", layout="wide")

st.title("📊 Wholesale Customer Segmentation Analysis")

model = joblib.load("Kmeans_model.pkl") 
scaler = joblib.load("Scaler.pkl") 

st.sidebar.header("Enter Feature Values")
st.sidebar.subheader("Quantity of each product in Wholesale")
Fresh = st.sidebar.number_input("Fresh", min_value=0.0, value=12000.0)
Milk = st.sidebar.number_input("Milk", min_value=0.0, value=5000.0)
Grocery = st.sidebar.number_input("Grocery", min_value=0.0, value=7000.0)
Frozen = st.sidebar.number_input("Frozen", min_value=0.0, value=2000.0)
Detergents_Paper = st.sidebar.number_input("Detergents_Paper", min_value=0.0, value=2500.0)
Delicassen = st.sidebar.number_input("Delicassen", min_value=0.0, value=1000.0)
input_data = np.array([[Fresh, Milk, Grocery, Frozen, Detergents_Paper, Delicassen]])
scaled_input = scaler.transform(input_data)

if st.button("Predict Cluster"):
    
    input_data = np.array([[Fresh, Milk, Grocery, Frozen, Detergents_Paper, Delicassen]])
    scaled_input = scaler.transform(input_data)
    cluster = model.predict(scaled_input)[0]

    st.subheader("🔍 Predicted Customer Segment")
    st.write(f"### Segment: **{cluster}**")

    cluster_info = {
        0: "🟦 Cluster 0: Low/Medium  Spending Customers",
        1: "🟩 Cluster 1: High Spending Customers"
    }

    if cluster in cluster_info:
        st.success(cluster_info.get(cluster,"Unknown Cluster"))


