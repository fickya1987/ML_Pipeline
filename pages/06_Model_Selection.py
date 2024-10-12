import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, brier_score_loss
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier
# Set page to open in wide mode
st.set_page_config(layout="wide")
# Function to plot confusion matrix
def plot_confusion_matrix(cm, class_names):
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_title('Confusion Matrix')
    ax.set_ylabel('Actual')
    ax.set_xlabel('Predicted')
    return fig

# General model evaluation function
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    cm = confusion_matrix(y_test, y_pred)
    
    metrics = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1,
        "ROC AUC": auc
    }
    cm_plot = plot_confusion_matrix(cm, ['Not Diabetic', 'Diabetic'])
    
    return metrics, cm_plot

# Streamlit UI code for model selection
st.title('Model Selection & Baseline Algorithm Evaluation')

# Check if the cleaned DataFrame, selected features, and target variable are available in session state
if 'df_final' in st.session_state and 'selected_features' in st.session_state and 'target_column' in st.session_state:
    df_final = st.session_state['df_final']
    selected_features = st.session_state['selected_features']
    target_column = st.session_state['target_column']

    # Display selected features for scaling
    st.write(f"Model will be trained on features: {', '.join(selected_features)}")
    st.write(f"The target variable is: {target_column}")

    # Assign the selected features and target column to X and Y
    X = df_final[selected_features]  # Select the feature columns
    Y = df_final[target_column]  # Select the target column

    # Provide reasoning for choosing Binary Classification
    if Y.nunique() == 2:  # Binary classification problem
        st.subheader("Binary Classification")
        st.write(f"The target variable `{target_column}` has exactly two unique values, indicating a binary outcome. Therefore, the problem can be modeled as a binary classification task, where the goal is to predict whether an individual falls into one of two categories.")
        
        st.subheader("Suggested Models for Binary Classification")
        suggested_models = {
            "Logistic Regression": LogisticRegression(),
            "K-Nearest Neighbors": KNeighborsClassifier(),
            "Support Vector Machine": SVC(probability=True),
            "Decision Tree": DecisionTreeClassifier(),
            "AdaBoost": AdaBoostClassifier(),
            "Gradient Boosting": GradientBoostingClassifier(),
            "Random Forest": RandomForestClassifier(),
            "Extra Trees": ExtraTreesClassifier()
        }
        
        # Display the list of suggested models
        st.write("The following models are suggested for binary classification tasks:")
        for model_name in suggested_models:
            st.write(f"- {model_name}")
        
    else:
        st.error("This app currently only supports binary classification problems.")
        st.stop()

    # Train-test split
    X_train_sc, X_test_sc, y_train_sc, y_test_sc = train_test_split(X, Y, test_size=0.1, random_state=0, stratify=Y)

    # Evaluate all suggested models
    model_results = {}
    comparison_data = []

    for model_name, model in suggested_models.items():
        model.fit(X_train_sc, y_train_sc)
        metrics, cm_plot = evaluate_model(model, X_test_sc, y_test_sc)
        model_results[model_name] = (metrics, cm_plot, {})  # Store metrics and confusion matrix

    # Metric comparison table
    st.header("Metric Comparison Across Models")
    if model_results:
        for model_name, (metrics, cm_plot, extra_metrics) in model_results.items():
            comparison_data.append({
                "Model": model_name,
                "Accuracy": metrics['Accuracy'],
                "Precision": metrics['Precision'],
                "Recall": metrics['Recall'],
                "F1-score": metrics['F1-score'],
                "ROC AUC": metrics['ROC AUC']
            })

        comparison_df = pd.DataFrame(comparison_data).set_index('Model')

        # Display the DataFrame for metric comparison
        st.dataframe(comparison_df)

    # Convert DataFrame into long format for Plotly to handle multiple metrics
    st.header("Comparison Plot of Model Performance")
    comparison_df_long = comparison_df.reset_index().melt(id_vars="Model", var_name="Metric", value_name="Score")

    # Create a Plotly bar plot with metrics on the x-axis and models represented for each metric
    comparison_plot = px.bar(comparison_df_long, 
                             x='Metric', 
                             y='Score', 
                             color='Model', 
                             barmode='group',
                             title='Model Performance Comparison')
    # Display the Plotly plot
    st.plotly_chart(comparison_plot)

    # Model selection for next steps
    st.header("Select Models for Ensembling")
    selected_models = st.multiselect("Choose one or more models for further evaluation:", list(suggested_models.keys()))

    # Save selected models in session state
    if selected_models:
        st.session_state['selected_models'] = selected_models
        print(selected_models)
        st.success(f"Selected models saved: {', '.join(selected_models)}")
    else:
        st.warning("Please select at least one model for further evaluation.")

    # Display the selected models at the end
    if 'selected_models' in st.session_state:
        st.write("You have selected the following models for further evaluation:")
        st.write(', '.join(st.session_state['selected_models']))

















