import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
# Set page to open in wide mode
st.set_page_config(layout="wide")
# Custom CSS to style the download button 
st.markdown("""
    <style>
    div[data-testid="stDownloadButton"] > button {
        padding: 10px 20px;
        font-size: 18px;
        font-weight: bold;
        color: white;
        background-color: #007BFF;
        border: 2px solid #007BFF;
        border-radius: 5px;
        cursor: pointer;
    }
    div[data-testid="stDownloadButton"] > button:hover {
        background-color: #0056b3;
    }
    div[data-testid="stDownloadButton"] {
        margin-bottom: 15px;
        padding: 5px;
    }
    </style>
""", unsafe_allow_html=True)

# Check if the cleaned DataFrame, selected features, and target variable are available in the session state
if 'df_final' in st.session_state and 'selected_features' in st.session_state and 'target_column' in st.session_state:
    df_final = st.session_state['df_final']
    selected_features = st.session_state['selected_features']
    target_column = st.session_state['target_column']
    
    st.title("Feature Scaling / Normalization")

    # Display selected features for scaling
    st.write(f"Scaling will be applied to the following features: {', '.join(selected_features)}")
    st.write(f"The target variable is: {target_column} (which will not be scaled).")

    # Choose scaling method
    scaling_method = st.radio("Choose a scaling method:", ("MinMaxScaler", "StandardScaler"))

    # Apply scaling when features are selected
    df_scaled = df_final.copy()

    if scaling_method == "MinMaxScaler":
        scaler = MinMaxScaler()
        st.write("MinMaxScaler scales the features to a fixed range, usually [0,1].")
    else:
        scaler = StandardScaler()
        st.write("StandardScaler standardizes the features by removing the mean and scaling to unit variance.")

    # Apply the scaler to the selected features only (not the target variable)
    df_scaled[selected_features] = scaler.fit_transform(df_final[selected_features])

    # Create a new DataFrame with only the scaled features and the target variable
    df_scaled_final = df_scaled[selected_features + [target_column]]

    # Show scaled data preview
    st.write("Scaled Data Preview (including only selected features and target variable):")
    st.dataframe(df_scaled_final)

    # Save the scaled DataFrame to the session state
    st.session_state['df_scaled'] = df_scaled_final

    # Provide a download button for the scaled dataset
    csv_scaled = df_scaled_final.to_csv(index=False).encode('utf-8')
    st.download_button(label="Download Scaled Data CSV", data=csv_scaled, file_name='scaled_data.csv', mime='text/csv')

else:
    st.warning("No cleaned data, selected features, or target variable available. Please preprocess the dataset first.")
