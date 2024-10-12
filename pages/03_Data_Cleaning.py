import streamlit as st
import pandas as pd
import numpy as np

# Set page to open in wide mode
st.set_page_config(layout="wide")

# Custom CSS to style the download button and checkbox
st.markdown("""
    <style>
        /* Style the checkbox input */
    div[data-testid="stCheckbox"] > div {
        display: flex;
        align-items: center;
    }

    /* Add some padding and margin around the checkbox */
    div[data-testid="stCheckbox"] {
        margin-bottom: 15px;
        padding: 5px;
        border: 2px solid #007BFF;
        border-radius: 5px;
        background-color: #f0f0f0;
    }
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

# Function to detect and remove outliers using Tukey's method
def TukeyOutliers(df_out, nameOfFeature):
    valueOfFeature = df_out[nameOfFeature]
    Q1 = np.percentile(valueOfFeature, 25.)
    Q3 = np.percentile(valueOfFeature, 75.)
    step = (Q3 - Q1) * 1.5
    outliers = valueOfFeature[~((valueOfFeature >= Q1 - step) & (valueOfFeature <= Q3 + step))].index.tolist()
    return outliers, len(outliers)

# Function to detect outliers using Z-score method
def ZScoreOutliers(df_out, nameOfFeature):
    valueOfFeature = df_out[nameOfFeature]
    mean = np.mean(valueOfFeature)
    std = np.std(valueOfFeature)
    threshold = 3  # Z-score threshold (3 is common, but can be adjusted)
    z_scores = (valueOfFeature - mean) / std
    outliers = valueOfFeature[abs(z_scores) > threshold].index.tolist()
    return outliers, len(outliers)

# Random Sample Imputation Function
def random_sample_imputation(df, feature):
    df_missing = df[df[feature].isnull()]
    df_non_missing = df[df[feature].notnull()]
    df_missing[feature] = df_missing[feature].apply(lambda x: np.random.choice(df_non_missing[feature]))
    df.loc[df[feature].isnull(), feature] = df_missing[feature]
    return df

# Mean Imputation Function
def mean_imputation(df, feature):
    mean_value = df[feature].mean()
    df[feature].fillna(mean_value, inplace=True)
    return df

# Median Imputation Function
def median_imputation(df, feature):
    median_value = df[feature].median()
    df[feature].fillna(median_value, inplace=True)
    return df

# Remove Missing Values Function
def remove_missing_values(df, feature):
    df = df[df[feature].notnull()]
    return df

# Replace with 0 Function
def replace_with_zero(df, feature):
    df[feature].fillna(0, inplace=True)
    return df

# Check if the dataset is available in the session state
if 'df' in st.session_state:
    df = st.session_state['df']
    df_modified = False  # Flag to track if the data is modified
    st.title("Data Cleaning")
    
    # Outlier Method Explanation
    st.subheader("Outlier Detection Methods")
    st.write("Outlier detection helps identify extreme values in the dataset. Below are two common methods:")

    st.markdown("""
    1. **Tukey's Method**: Uses the interquartile range (IQR) to detect outliers. It is robust to extreme values and identifies outliers outside the range [Q1 - 1.5*IQR, Q3 + 1.5*IQR].
    2. **Z-score Method**: Identifies outliers based on how many standard deviations a data point is from the mean. A common threshold is 3, meaning points more than 3 standard deviations away from the mean are flagged as outliers.
    """)

    # Outlier removal with method selection
    if st.checkbox("Remove Outliers"):
        df_t = df.copy()
        selected_features = st.multiselect("Select features to remove outliers from", df_t.columns)

        # Dropdown to select the outlier removal method
        outlier_method = st.selectbox("Select outlier removal method", ["Tukey's Method", "Z-score Method"])

        outliers_indices = set()
        outlier_summary = []

        for feature in selected_features:
            if outlier_method == "Tukey's Method":
                outliers, num_outliers = TukeyOutliers(df_t, feature)
            elif outlier_method == "Z-score Method":
                outliers, num_outliers = ZScoreOutliers(df_t, feature)

            outliers_indices.update(outliers)
            outlier_summary.append(f"Feature '{feature}': {num_outliers} outliers found using {outlier_method}.")

        if outliers_indices:
            df_cleaned = df_t.drop(outliers_indices).reset_index(drop=True)
            st.write(f"Total number of outliers removed: {len(outliers_indices)}")
            st.write(f"New dataset has {df_cleaned.shape[0]} samples and {df_cleaned.shape[1]} features.")
            
            for summary in outlier_summary:
                st.write(summary)

            st.session_state['df_cleaned'] = df_cleaned
            df = df_cleaned
            df_modified = True
        else:
            st.write("No outliers detected across the selected features.")

    # Missing value handling
    if st.checkbox("Handle Missing Values"):
        if 'df_cleaned' in st.session_state:
            df = st.session_state['df_cleaned']
        missing_features = df.columns[df.isnull().any()].tolist()
        if missing_features:
            st.write(f"Missing values found in: {missing_features}")
            selected_missing_features = st.multiselect("Select features to handle missing values", missing_features)

            if selected_missing_features:
                method = st.selectbox("Select method for handling missing values", 
                                       ["Random Sample Imputation", "Mean Imputation", "Median Imputation", 
                                        "Remove Missing Values", "Replace with 0"])

                for feature in selected_missing_features:
                    if method == "Random Sample Imputation":
                        df = random_sample_imputation(df, feature)
                        st.write(f"Missing values imputed for feature '{feature}' using Random Sample Imputation.")
                    elif method == "Mean Imputation":
                        df = mean_imputation(df, feature)
                        st.write(f"Missing values imputed for feature '{feature}' using Mean Imputation.")
                    elif method == "Median Imputation":
                        df = median_imputation(df, feature)
                        st.write(f"Missing values imputed for feature '{feature}' using Median Imputation.")
                    elif method == "Remove Missing Values":
                        df = remove_missing_values(df, feature)
                        st.write(f"Missing values removed for feature '{feature}'.")
                    elif method == "Replace with 0":
                        df = replace_with_zero(df, feature)
                        st.write(f"Missing values replaced with 0 for feature '{feature}'.")

                st.session_state['df_imputed'] = df
                df_modified = True
            else:
                st.write("No features selected for handling missing values.")
        else:
            st.write("No missing values found in the dataset.")

    # Data type conversion
    if st.checkbox("Data Type Conversion"):
        selected_conversion_features = st.multiselect("Select features for data type conversion", df.columns)
        for column in selected_conversion_features:
            current_dtype = df[column].dtype
            new_dtype = st.selectbox(f"Select new data type for '{column}'", ["int", "float", "string", "datetime"], key=column)
            if current_dtype == new_dtype:
                st.write(f"No need for data type conversion for '{column}'. It is already of type {current_dtype}.")
            else:
                if new_dtype == "int":
                    df[column] = df[column].astype(int)
                elif new_dtype == "float":
                    df[column] = df[column].astype(float)
                elif new_dtype == "string":
                    df[column] = df[column].astype(str)
                elif new_dtype == "datetime":
                    df[column] = pd.to_datetime(df[column], errors='coerce')
                st.write(f"Data type for '{column}' converted from {current_dtype} to {new_dtype}.")
        
        st.session_state['df_converted'] = df
        df_modified = True
    
    # Offer the download option only if any changes were made
    if df_modified:
        st.subheader("Cleaning Report")
        st.write('Original Data shape: {}  \nCleaned Data shape: {}  \nTotal rows lost: {}  \n Data loss is {}% of original data'.format(df_t.shape[0],df_cleaned.shape[0],
                                                              df_t.shape[0]-df_cleaned.shape[0],
                                                        (df_t.shape[0]-df_cleaned.shape[0])/df_t.shape[0]*100))
        csv_preprocessed = df.to_csv(index=False).encode('utf-8')
        st.download_button(label="Download Preprocessed Data CSV", data=csv_preprocessed, file_name='preprocessed_data.csv', mime='text/csv')
        st.session_state['df_final'] = df
        
else:
    st.warning("Please upload the dataset to begin preprocessing.")
