import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly.express as px
import io

# Set page to open in wide mode
st.set_page_config(layout="wide")

# Custom CSS to style both the checkbox 
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

    /* Align headers and text in correlation section */
    .correlation-section {
        display: flex;
        align-items: flex-start;
        justify-content: space-between;
        gap: 30px;
    }
    
    /* Reduce box plot size */
    .box-plot {
        width: 60%;
    }

    </style>
""", unsafe_allow_html=True)

# Function to create a heatmap
def HeatMap(df, x=True):
    correlations = df.corr()
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    heatmap_fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(correlations, cmap=cmap, vmax=1.0, center=0, fmt='.2f', square=True, 
                linewidths=.5, annot=x, cbar_kws={"shrink": .75})
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=10)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=10)
    plt.tight_layout()
    return heatmap_fig

# Custom function for plotting outliers
def OutLiersBox(df, nameOfFeature):
    trace0 = go.Box(
        y=df[nameOfFeature],
        name="All Points",
        jitter=0.3,
        pointpos=-1.8,
        boxpoints='all',
        marker=dict(color='rgb(7,40,89)'),
        line=dict(color='rgb(7,40,89)')
    )

    trace1 = go.Box(
        y=df[nameOfFeature],
        name="Only Whiskers",
        boxpoints=False,
        marker=dict(color='rgb(9,56,125)'),
        line=dict(color='rgb(9,56,125)')
    )

    trace2 = go.Box(
        y=df[nameOfFeature],
        name="Suspected Outliers",
        boxpoints='suspectedoutliers',
        marker=dict(color='rgb(8,81,156)', outliercolor='rgba(219, 64, 82, 0.6)',
                    line=dict(outliercolor='rgba(219, 64, 82, 0.6)', outlierwidth=2)),
        line=dict(color='rgb(8,81,156)')
    )

    trace3 = go.Box(
        y=df[nameOfFeature],
        name="Whiskers and Outliers",
        boxpoints='outliers',
        marker=dict(color='rgb(107,174,214)'),
        line=dict(color='rgb(107,174,214)')
    )

    data = [trace0, trace1, trace2, trace3]

    layout = go.Layout(
        margin=dict(t=40),  # Adjust the top margin for better spacing
        legend=dict(
            orientation="h",  # Set the legend orientation to horizontal
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        )
    )

    fig = go.Figure(data=data, layout=layout)
    return fig

st.title("Exploratory Data Analysis")

# Retrieve the dataframe from session state
if 'df' in st.session_state:
    df = st.session_state['df']

# Display Data Types and Null Values
if st.checkbox("Show Descriptive Statistics"):
    st.subheader("Data Description")
    st.write(df.describe())

    st.subheader("Data Types and Null Values")
    null_info = pd.DataFrame({
        'Data Type': df.dtypes,
        'Null Values': df.isnull().sum(),
        'Non-null Count': df.notnull().sum()
    })
    null_info['Null Values'] = null_info['Null Values'].apply(lambda x: f"No null values" if x == 0 else f"{x} null values")
    st.dataframe(null_info)

# Correlation Heatmap and Interpretation
if st.checkbox("Show Data Visualization"):
    st.subheader("Correlation Heatmap")
    heatmap_fig = HeatMap(df, x=True)
    st.pyplot(heatmap_fig)
    # Display highly correlated features and interpretation side by side
    st.subheader("Highly Correlated Features")

    col1, col2 = st.columns([3, 4], gap="large")  # Adjust column widths for better space

    with col1:
        correlations = df.corr().abs().unstack().sort_values(ascending=False)
        # Remove self-correlations (correlation of a feature with itself)
        correlations = correlations[correlations < 1].drop_duplicates()
        correlations = correlations.reset_index()
        correlations.columns = ['Feature 1', 'Feature 2', 'Correlation']
        st.write(correlations.head(10))  # Display the top 10 highly correlated pairs

    with col2:
        st.subheader("Interpretation of Correlations")
        st.markdown("""
        **Understanding Correlation:**
        
        Correlation values range from **-1** to **1**:
        
        - **Positive Correlation** (closer to 1): As one feature increases, the other feature tends to increase.
        Example: Higher study hours leading to better grades.
        
        - **Negative Correlation** (closer to -1): As one feature increases, the other feature tends to decrease.
        Example: More exercise might result in lower body fat percentage.
        
        - **No Correlation** (closer to 0): Minimal or no linear relationship between features.
        Example: Shoe size vs. exam scores.

        **Importance for Predictive Modeling:**
        
        - Strong correlations (values near ±1) indicate a significant relationship and are often key features for prediction.
        
        - High correlation between independent features can lead to multicollinearity, which may require addressing by removing or combining features to avoid redundancy and overfitting.
        """)

    # Adding CSS to ensure proper alignment
    st.markdown("""
    <style>
        .stColumn {
            display: inline-block;
            vertical-align: top;
        }
    </style>
    """, unsafe_allow_html=True)


# Outlier Investigation and Interpretation
if st.checkbox("Outlier Investigation"):
    st.subheader("Single Feature Outliers")
    feature = st.selectbox("Select feature for outlier detection", df.columns)

    # Create aligned sections for outlier box plot and interpretation with appropriate gap
    col1, col2 = st.columns([4, 3], gap="large")  # Adjust column widths for better space


    with col1:
        st.subheader(f"Boxplot for : {feature}")
        # Call the custom Plotly outlier function
        st.markdown("<br><br><br>", unsafe_allow_html=True)  # Adds two line breaks to push the plot down
        fig = OutLiersBox(df, feature)
        st.plotly_chart(fig, use_container_width=True)  # Use container width to make it responsive

    with col2:
        st.subheader("Understanding Box Plots and Outliers")
        st.markdown("""
        ****
        - **Box**: Represents the interquartile range (IQR), which contains the middle 50% of the data.Lower Edge: 
                    1st Quartile (Q1). Upper Edge: 3rd Quartile (Q3). Horizontal Line inside the Box: Median          
        - **Whiskers**: The lines that extend from the box to the smallest and largest values within 1.5 * IQR.
        - **Outliers**: Data points that lie outside the whisker range are considered outliers and are displayed as individual dots.
        - **All Points**: Every data point, including outliers.
        - **Only Whiskers**: Displays only the key data range within the whiskers, hiding outliers.
        - **Suspected Outliers**: Data points that fall outside 1.5 * IQR but aren't extreme enough to be definite outliers.
        - **Whiskers and Outliers**: Displays both the whiskers and any definite outliers.
        """)
