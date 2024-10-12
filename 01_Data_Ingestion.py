import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# Set page to open in wide mode
st.set_page_config(layout="wide")

st.title("Upload Data")


uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    # Save the dataframe to session state
    st.session_state['df'] = df

    st.write("Data preview:", df.head())