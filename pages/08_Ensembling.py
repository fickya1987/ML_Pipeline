import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import (train_test_split, StratifiedKFold, 
                                     cross_val_score)
from sklearn.ensemble import (VotingClassifier, GradientBoostingClassifier, 
                              RandomForestClassifier, ExtraTreesClassifier, 
                              AdaBoostClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (roc_auc_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, classification_report,
                             roc_curve, precision_recall_curve, average_precision_score)
from sklearn.preprocessing import MinMaxScaler
from mlens.ensemble import SuperLearner

# Set page to open in wide mode
st.set_page_config(layout="wide")

# Ignore Warnings
warnings.filterwarnings('ignore')

# Set a seed for reproducibility
SEED = 7
np.random.seed(SEED)

@st.cache_resource
def get_models():
    """Generate a library of base learners."""
    models = {
        'Logistic Regression': LogisticRegression(C=0.7678243129497218, penalty='l2', solver='saga'),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=15),
        'Support Vector Machine': SVC(C=1.7, kernel='linear', probability=True),  # Ensure probability=True
        'Decision Tree': DecisionTreeClassifier(criterion='gini', max_depth=3, max_features=2, min_samples_leaf=3),
        'AdaBoost': AdaBoostClassifier(learning_rate=0.05, n_estimators=150),
        'Gradient Boosting': GradientBoostingClassifier(learning_rate=0.01, n_estimators=100),
        'Gaussian Naive Bayes': GaussianNB(),
        'Random Forest': RandomForestClassifier(),
        'Extra Trees': ExtraTreesClassifier()
    }
    return models

@st.cache_data
def load_and_preprocess_data():
    df_clean = pd.read_csv('df_clean_output.csv')
    df_unscaled = df_clean[['Glucose', 'BMI', 'Age', 'DiabetesPedigreeFunction', 'Outcome']]
    df_imp_scaled = MinMaxScaler().fit_transform(df_unscaled)
    
    X = df_imp_scaled[:, 0:4]
    Y = df_imp_scaled[:, 4]
    
    X_train_sc, X_test_sc, y_train_sc, y_test_sc = train_test_split(
        X, Y, test_size=0.1, random_state=0, stratify=df_imp_scaled[:, 4]
    )
    
    return X_train_sc, X_test_sc, y_train_sc, y_test_sc

@st.cache_resource
def create_and_train_ensemble(_X_train_sc, _y_train_sc, _X_test_sc, _y_test_sc):
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)
    ensemble = VotingClassifier(estimators=list(get_models().items()), voting='soft')  # Use soft voting for probabilities
    results = cross_val_score(ensemble, _X_train_sc, _y_train_sc, cv=kfold)
    
    ensemble_model = ensemble.fit(_X_train_sc, _y_train_sc)
    pred = ensemble_model.predict(_X_test_sc)
    pred_proba = ensemble_model.predict_proba(_X_test_sc)[:, 1]  # Now this should work correctly
    
    return ensemble_model, results.mean(), pred, pred_proba

@st.cache_data
def train_predict(_model_list, _xtrain, _xtest, _ytrain):
    """Fit models in list on training set and return predictions."""
    P = pd.DataFrame(np.zeros((_xtest.shape[0], len(_model_list))), columns=_model_list.keys())

    for name, model in _model_list.items():
        model.fit(_xtrain, _ytrain)
        P[name] = model.predict_proba(_xtest)[:, 1]

    return P

def plot_roc_curve(y_true, y_pred_proba):
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = roc_auc_score(y_true, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    st.pyplot(plt)

def plot_precision_recall_curve(y_true, y_pred_proba):
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    average_precision = average_precision_score(y_true, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.step(recall, precision, color='b', alpha=0.2, where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title(f'Precision-Recall curve: AP={average_precision:.2f}')
    st.pyplot(plt)

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    st.pyplot(plt)

def main():
    st.title("Ensemble Learning with Comprehensive Evaluation")

    # Load and preprocess data
    X_train_sc, X_test_sc, y_train_sc, y_test_sc = load_and_preprocess_data()

    st.session_state.base_model_created = True
    with st.spinner("Creating and training base model..."):
        ensemble_model, cv_accuracy, y_pred, y_pred_proba = create_and_train_ensemble(X_train_sc, y_train_sc, X_test_sc, y_test_sc)
    
    st.success("Base model created successfully!")
    st.write('Cross-validation accuracy: ', cv_accuracy)

    # Evaluate the ensemble model
    st.subheader("Ensemble Model Evaluation")
    
    # Classification report
    st.write("Classification Report:")
    report = classification_report(y_test_sc, y_pred, output_dict=True)
    st.table(pd.DataFrame(report).transpose())

    # Additional metrics
    st.write(f"Accuracy: {(y_test_sc == y_pred).mean():.4f}")
    st.write(f"Precision: {precision_score(y_test_sc, y_pred):.4f}")
    st.write(f"Recall: {recall_score(y_test_sc, y_pred):.4f}")
    st.write(f"F1-score: {f1_score(y_test_sc, y_pred):.4f}")
    st.write(f"ROC AUC: {roc_auc_score(y_test_sc, y_pred_proba):.4f}")

    # Plots
    st.subheader("Evaluation Plots")
    
    # Create three columns for the plots
    col1, col2, col3 = st.columns(3)
    
    # ROC Curve
    with col1:
        st.subheader("ROC Curve:")
        plot_roc_curve(y_test_sc, y_pred_proba)

    # Precision-Recall Curve
    with col2:
        st.subheader("Precision-Recall Curve:")
        plot_precision_recall_curve(y_test_sc, y_pred_proba)

    # Confusion Matrix
    with col3:
        st.subheader("Confusion Matrix:")
        plot_confusion_matrix(y_test_sc, y_pred)

    # Generate predictions and display correlation matrix
    models = get_models()
    P = train_predict(models, X_train_sc, X_test_sc, y_train_sc)

    # Super Learner section
    st.subheader("Super Learner Training")
    
    # Access the selected models from session state
    if 'selected_models' in st.session_state:
        # Checkbox to select all models
        select_all = st.checkbox("Select All Models", value=True)
        
        if select_all:
            selected_model_names = st.session_state['selected_models']  # Automatically select all models
        else:
            selected_model_names = []

        st.write("Selected models:", selected_model_names)

        if st.button("Train Super Learner"):
            if not st.session_state.get('base_model_created', False):
                st.warning("Please create the base model first before training the Super Learner.")
            elif selected_model_names:
                with st.spinner("Training Super Learner..."):
                    super_learner = SuperLearner(scorer=roc_auc_score, random_state=SEED)
                    
                    # Add models without the name argument
                    base_learners = {name: models[name] for name in selected_model_names}
                    super_learner.add(list(base_learners.values()))

                    # Add the meta learner
                    meta_learner = LogisticRegression(C=1.0, max_iter=1000, random_state=SEED)
                    super_learner.add(meta_learner)

                    super_learner.fit(X_train_sc, y_train_sc)
                    pred = super_learner.predict(X_test_sc)
                    pred_proba = super_learner.predict_proba(X_test_sc)

                    # Handle the case where pred_proba is one-dimensional
                    if pred_proba.ndim == 1:  # Only one class
                        pred_proba = np.column_stack((1 - pred_proba, pred_proba))

                    pred_proba = pred_proba[:, 1]  # Get probabilities for the positive class

                    st.success("Super Learner trained successfully!")

                    # Display metrics
                    st.write(f"Super Learner Accuracy: {(y_test_sc == pred).mean():.4f}")
                    st.write(f"Super Learner Precision: {precision_score(y_test_sc, pred):.4f}")
                    st.write(f"Super Learner Recall: {recall_score(y_test_sc, pred):.4f}")
                    st.write(f"Super Learner F1-score: {f1_score(y_test_sc, pred):.4f}")
                    st.write(f"Super Learner ROC AUC: {roc_auc_score(y_test_sc, pred_proba):.4f}")

                    # Super Learner Evaluation Plots
                    st.subheader("Super Learner Evaluation Plots")
                    
                    # Create three columns for the plots
                    col1, col2, col3 = st.columns(3)

                    # ROC Curve
                    with col1:
                        st.subheader("ROC Curve:")
                        plot_roc_curve(y_test_sc, pred_proba)

                    # Precision-Recall Curve
                    with col2:
                        st.subheader("Precision-Recall Curve:")
                        plot_precision_recall_curve(y_test_sc, pred_proba)

                    # Confusion Matrix
                    with col3:
                        st.subheader("Confusion Matrix:")
                        plot_confusion_matrix(y_test_sc, pred)

if __name__ == "__main__":
    main()
