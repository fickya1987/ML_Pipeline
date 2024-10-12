import random
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

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
    </style>
""", unsafe_allow_html=True)

# Check if required session states are available
if 'df_final' in st.session_state and 'selected_models' in st.session_state and 'selected_features' in st.session_state and 'target_column' in st.session_state:

    # Retrieve session state variables
    df = st.session_state['df_final']
    selected_models = st.session_state['selected_models']
    selected_features = st.session_state['selected_features']
    target_column = st.session_state['target_column']
    
    # Display the selected models
    st.write("### Selected Models for Hyperparameter Tuning")
    st.write(selected_models)

    # Provide user the choice for Auto-tuning or Manual-tuning using checkboxes
    st.write("### Hyperparameter Tuning Methods")
    auto_tuning = st.checkbox("Auto Tuning Hyperparameters")
    manual_tuning = st.checkbox("Manual Tuning Hyperparameters")

    # Ensure only one can be selected
    if auto_tuning and manual_tuning:
        st.warning("Please select only one tuning method: Auto-tuning or Manual-tuning.")
        auto_tuning = False  # Reset auto_tuning if both are selected
        manual_tuning = True  # Keep manual_tuning

    model_dict = {
        "Logistic Regression": LogisticRegression(),  
        "K-Nearest Neighbors": KNeighborsClassifier(),
        "Decision Tree": DecisionTreeClassifier(),
        "Gaussian Naive Bayes": GaussianNB(),
        "Support Vector Machine": SVC(probability=True),
        "AdaBoost": AdaBoostClassifier(),
        "Gradient Boosting": GradientBoostingClassifier(),
        "Random Forest": RandomForestClassifier(),
        "Extra Trees": ExtraTreesClassifier()
    }

    # Dictionary to store evaluation metrics
    evaluation_metrics = {}

    if auto_tuning:
        st.write("### Auto-Tuning Hyperparameters")

        # Setting the random seed for reproducibility
        SEED = 42
        
        # Dictionary to store the best parameters for each model
        best_params = {}

        # Iterate over selected models and perform tuning
        for model_name in selected_models:
            st.write(f"Tuning: {model_name}")
            model = model_dict[model_name]

            if model_name == "Logistic Regression":
                param_grid = {'C': [0.1, 1.0, 10], 'penalty': ['l2']}
                randomized_search = RandomizedSearchCV(model, param_grid, n_iter=100, cv=5, scoring='roc_auc', random_state=SEED)
                randomized_search.fit(df[selected_features], df[target_column])
                model = randomized_search.best_estimator_
                best_params[model_name] = randomized_search.best_params_

            elif model_name == "Decision Tree":
                param_grid = {'max_depth': [3, 5, 7, None], 'criterion': ['gini', 'entropy']}
                randomized_search = RandomizedSearchCV(model, param_grid, n_iter=100, cv=5, scoring='roc_auc', random_state=SEED)
                randomized_search.fit(df[selected_features], df[target_column])
                model = randomized_search.best_estimator_
                best_params[model_name] = randomized_search.best_params_

            else:
                param_grid = {}
                if model_name == "K-Nearest Neighbors":
                    param_grid = {'n_neighbors': [3, 5, 7], 'weights': ['uniform', 'distance']}
                elif model_name == "Support Vector Machine":
                    param_grid = {'C': [0.1, 1.0, 10], 'kernel': ['linear', 'rbf', 'poly']}
                elif model_name == "AdaBoost":
                    param_grid = {'n_estimators': [50, 100, 150], 'learning_rate': [0.1, 0.5, 1.0]}
                elif model_name == "Gradient Boosting":
                    param_grid = {'n_estimators': [100, 200], 'learning_rate': [0.01, 0.1, 0.2]}
                elif model_name == "Random Forest":
                    param_grid = {'n_estimators': [100, 200, 300], 'max_depth': [None, 10, 20]}
                elif model_name == "Extra Trees":
                    param_grid = {'n_estimators': [100, 200], 'max_depth': [None, 10, 20]}

                # Perform hyperparameter tuning using GridSearchCV
                grid_search = GridSearchCV(model, param_grid, cv=5, scoring='roc_auc')
                grid_search.fit(df[selected_features], df[target_column])
                model = grid_search.best_estimator_
                best_params[model_name] = grid_search.best_params_

        # Display all best parameters at once
        st.write("### Best Hyperparameters for Each Model:")
        for model_name, params in best_params.items():
            st.write(f"{model_name}: {params}")

        # Evaluate models with best parameters
        st.write("### Evaluation Metrics for Tuned Models:")
        cols = st.columns(4)  # Create 4 columns for layout
        plot_index = 0  # Track which column to place the plot in

        # Create a list to store the classification reports for later display
        reports = {}

        for model_name, params in best_params.items():
            model = model_dict[model_name].set_params(**params)

            # Fit the model
            X = df[selected_features]
            y = df[target_column]
            model.fit(X, y)

            # Predictions
            y_pred = model.predict(X)

            # Evaluation
            report = classification_report(y, y_pred, output_dict=True)
            evaluation_metrics[model_name] = report
            reports[model_name] = pd.DataFrame(report).transpose()  # Store the report for later

            # Plot confusion matrix
            cm = confusion_matrix(y, y_pred)
            fig, ax = plt.subplots(figsize=(3, 4))  # Adjust the figure size here
            sns.heatmap(cm, annot=True, fmt=".2f", cmap='Blues', square=True)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')

            # Display the plot in the appropriate column
            with cols[plot_index]:
                st.write(f'#### {model_name} Confusion Matrix')
                st.pyplot(fig)

            plot_index += 1  # Move to the next column
            if plot_index >= 4:  # Reset after three plots
                plot_index = 0
                cols = st.columns(4)  # Reset columns for new row

        # Now display all classification reports in a single section
        st.write("### Classification Reports:")
        for model_name, report_df in reports.items():
            st.write(f"#### Classification Report for {model_name}:")
            st.table(report_df)

    if manual_tuning:
        st.write("### Manual-Tuning Hyperparameters")

        # Create a list to store the confusion matrices for later display
        confusion_matrices = []
        reports = {}

        # Iterate over selected models for manual tuning
        for model_name in selected_models:
            st.write(f"#### Tune Hyperparameters for {model_name}")

            # Define manual hyperparameter widgets based on model
            if model_name == "Logistic Regression":
                C = st.number_input("C (Regularization Strength):", min_value=0.01, max_value=10.0, value=1.0, key=f"log_reg_C")
                penalty = st.selectbox("Penalty:", options=['l2', 'l1'], key=f"log_reg_penalty")
                tuned_model = LogisticRegression(C=C, penalty=penalty)
            
            elif model_name == "K-Nearest Neighbors":
                n_neighbors = st.slider("Number of Neighbors:", min_value=1, max_value=20, value=5, key=f"knn_neighbors")
                weights = st.selectbox(f"KNeighborsClassifier: Weights", ['uniform', 'distance'])
                tuned_model = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights)

            elif model_name == "Decision Tree":
                max_depth = st.slider("Max Depth:", min_value=1, max_value=10, value=3, key=f"dt_max_depth")
                criterion = st.selectbox(f"DecisionTreeClassifier: Criterion", ['gini', 'entropy'])
                tuned_model = DecisionTreeClassifier(max_depth=max_depth, criterion=criterion)

            elif model_name == "Gaussian Naive Bayes":
                tuned_model = GaussianNB()  # No hyperparameters to tune

            elif model_name == "Support Vector Machine":
                C = st.number_input("C (Regularization Strength):", min_value=0.01, max_value=10.0, value=1.0, key=f"svc_C")
                kernel = st.selectbox("Kernel:", options=['linear', 'rbf', 'poly'], key=f"svc_kernel")
                tuned_model = SVC(C=C, kernel=kernel, probability=True)

            elif model_name == "AdaBoost":
                n_estimators = st.slider("Number of Estimators:", min_value=50, max_value=500, value=100, key=f"ada_n_estimators")
                learning_rate = st.number_input("Learning Rate:", min_value=0.01, max_value=1.0, value=0.1, key=f"ada_learning_rate")
                tuned_model = AdaBoostClassifier(n_estimators=n_estimators, learning_rate=learning_rate)

            elif model_name == "Gradient Boosting":
                n_estimators = st.slider("Number of Estimators:", min_value=100, max_value=1000, value=200, key=f"gb_n_estimators")
                learning_rate = st.number_input("Learning Rate:", min_value=0.01, max_value=1.0, value=0.1, key=f"gb_learning_rate")
                tuned_model = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate)

            elif model_name == "Random Forest":
                n_estimators = st.slider("Number of Estimators:", min_value=100, max_value=500, value=200, key=f"rf_n_estimators")
                max_depth = st.slider("Max Depth:", min_value=1, max_value=20, value=10, key=f"rf_max_depth")
                tuned_model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)

            elif model_name == "Extra Trees":
                n_estimators = st.slider("Number of Estimators:", min_value=100, max_value=500, value=200, key=f"et_n_estimators")
                max_depth = st.slider("Max Depth:", min_value=1, max_value=20, value=10, key=f"et_max_depth")
                tuned_model = ExtraTreesClassifier(n_estimators=n_estimators, max_depth=max_depth)

            # Fit the tuned model
            X = df[selected_features]
            y = df[target_column]
            tuned_model.fit(X, y)

            # Predictions
            y_pred = tuned_model.predict(X)

            # Evaluation
            report = classification_report(y, y_pred, output_dict=True)
            evaluation_metrics[model_name] = report
            reports[model_name] = pd.DataFrame(report).transpose()  # Store the report for later

            # Collect confusion matrix data
            cm = confusion_matrix(y, y_pred)
            confusion_matrices.append((model_name, cm))  # Store the model name and confusion matrix for later

        # Display the confusion matrices in a grid format
        st.write("### Confusion Matrices:")
        cols = st.columns(4)  # Create 4 columns for layout
        plot_index = 0  # Track which column to place the plot in

        for model_name, cm in confusion_matrices:
            fig, ax = plt.subplots(figsize=(3, 4))  # Adjust the figure size here
            sns.heatmap(cm, annot=True, fmt=".2f", cmap='Blues', square=True)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')

            # Display the plot in the appropriate column
            with cols[plot_index]:
                st.write(f'#### {model_name}')
                st.pyplot(fig)

            plot_index += 1  # Move to the next column
            if plot_index >= 4:  # Reset after four plots
                plot_index = 0
                cols = st.columns(4)  # Reset columns for new row

        # Now display all classification reports in a single section
        st.write("### Classification Reports:")
        for model_name, report_df in reports.items():
            st.write(f"#### Classification Report for {model_name}:")
            st.table(report_df)

else:
    st.error("Required session state variables are not found.")
