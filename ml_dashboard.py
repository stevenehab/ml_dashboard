import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, mean_squared_error, roc_auc_score, classification_report, confusion_matrix, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
from imblearn.over_sampling import SMOTE
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPClassifier, MLPRegressor
import plotly.express as px

# **Constants and Helper Functions**
def show_error(message):
    st.error(message)

def show_warning(message):
    st.warning(message)

def load_dataset(file):
    if file.name.endswith(".csv"):
        return pd.read_csv(file)
    elif file.name.endswith(".xlsx"):
        return pd.read_excel(file, engine="openpyxl")
    else:
        show_error("Unsupported file type. Please upload a CSV or Excel file.")
        st.stop()

def encode_categoricals(df, encoding):
    categoricals = df.select_dtypes(include="object").columns
    if encoding == "Label Encoding":
        le = LabelEncoder()
        for col in categoricals:
            df[col] = le.fit_transform(df[col])
    elif encoding == "One-Hot Encoding":
        df = pd.get_dummies(df, columns=categoricals)
    return df

# **Main Application**
st.title("ML-Powered Data Dashboard")

with st.expander("How to Use This App"):
    st.write("1. Upload your dataset (.csv or.xlsx).")
    st.write("2. Preview and understand your dataset.")
    st.write("3. Select preprocessing options.")
    st.write("4. Choose a machine learning task and algorithm.")
    st.write("5. Evaluate your model's performance.")
    st.write("6. Make predictions using the model.")

# **Upload Dataset**
uploaded_file = st.file_uploader("Upload your dataset (CSV or Excel)", type=["csv", "xlsx"])

if uploaded_file is not None:
    try:
        df = load_dataset(uploaded_file)
        
        # **Dataset Preview & Info**
        st.subheader("Dataset Preview")
        st.dataframe(df.head())
        st.subheader("Dataset Info")
        col1, col2, col3 = st.columns(3)
        col1.write(f"Shape: {df.shape}")
        col2.write(f"Columns: {list(df.columns)}")
        col3.write("Data Types:")
        col3.write(df.dtypes)

        # **Data Preprocessing**
        st.subheader("Data Preprocessing")
        preprocessing_ops = st.container()
        with preprocessing_ops:
            if st.checkbox("Drop rows with missing values"):
                df = df.dropna()
                st.write("Rows with missing values have been dropped.")
                st.write(f"Updated Shape: {df.shape}")
            elif st.checkbox("Fill missing values with mean (numerical only)"):
                df = df.fillna(df.mean())
                st.write("Missing values have been filled with mean.")
                st.write(f"Updated Shape: {df.shape}")

            encoding = st.selectbox("Select Encoding for Categorical Variables", ["Label Encoding", "One-Hot Encoding"])
            df = encode_categoricals(df, encoding)
            st.write("Encoded Dataset:")
            st.dataframe(df.head())

            # **Handle Imbalanced Data for Classification**
            handle_imbalance = st.checkbox("Handle Class Imbalance (Classification only)")

        # **Machine Learning Model Builder**
        st.subheader("Machine Learning Model Builder")

        target = st.selectbox("Select the target variable", df.columns)

        task = st.radio("Select Task Type", ["Classification", "Regression"])
        if task == "Classification":
            if pd.api.types.is_numeric_dtype(df[target]) and len(df[target].unique()) > 10:
                show_error(
                    f"Selected task is Classification, but the target variable '{target}' appears to be continuous. "
                    "Please select a categorical target variable or choose Regression."
                )
                st.stop()

            elif pd.api.types.is_numeric_dtype(df[target]):
                st.write(f"Target variable '{target}' is continuous. Binning it into discrete categories for Classification...")
                df[target] = pd.cut(df[target], bins=3, labels=[0, 1, 2])  # Example: 3 bins
                st.write("Binned target variable:", df[target].value_counts())

            if handle_imbalance:
                st.write("Handling class imbalance using SMOTE...")
                X = df.drop(columns=[target])
                y = df[target]
                smote = SMOTE()
                X_res, y_res = smote.fit_resample(X, y)
                st.write(f"Resampled Dataset Shape: {X_res.shape}")
                X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)
            else:
                X = df.drop(columns=[target])
                y = df[target]
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        elif task == "Regression":
            if not pd.api.types.is_numeric_dtype(df[target]):
                show_error(
                    f"Selected task is Regression, but the target variable '{target}' is not numeric. "
                    "Please select a numeric target variable or choose Classification."
                )
                st.stop()

            X = df.drop(columns=[target])
            y = df[target]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # **Remove Outliers in Target for Regression**
            if st.checkbox("Remove Outliers in Target"):
                Q1 = y_train.quantile(0.25)
                Q3 = y_train.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                inliers = (y_train >= lower_bound) & (y_train <= upper_bound)
                X_train, y_train = X_train[inliers], y_train[inliers]
                st.write(f"Removed outliers from target. New training data size: {len(y_train)}")

            # **Apply Log Transformation to Target for Regression**
            if st.checkbox("Apply Log Transformation to Target"):
                y_train = np.log1p(y_train)  
                y_test = np.log1p(y_test)
                st.write("Log transformation applied to target variable.")

        # **Scaling**
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # **Algorithm Selection**
        algorithm = st.radio("Select Algorithm", {
            "Classification": ["Random Forest", "XGBoost", "SVM", "Neural Network"],
            "Regression": ["Random Forest", "XGBoost", "SVM", "Neural Network"]
        }[task])

        # **Model Training**
        if task == "Classification":
            st.write(f"You selected: Classification with {algorithm}")
            if algorithm == "Random Forest":
                model = RandomForestClassifier()
            elif algorithm == "XGBoost":
                model = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
            elif algorithm == "SVM":
                model = SVC()
            elif algorithm == "Neural Network":
                model = MLPClassifier()

            # **Basic Hyperparameter Tuning for RandomForest**
            if algorithm == "Random Forest":
                params = {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [None, 5, 10]
                }
                grid_search = GridSearchCV(model, params, cv=3, n_jobs=-1)
                grid_search.fit(X_train, y_train)
                model = grid_search.best_estimator_
                st.write(f"Best Parameters: {grid_search.best_params_}")

            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            st.write(f"Model Accuracy: {accuracy:.2f}")
            st.write("Classification Report:")
            st.write(classification_report(y_test, y_pred))
            st.write("Confusion Matrix:")
            st.write(confusion_matrix(y_test, y_pred))
            if algorithm in ["Random Forest", "XGBoost"]:
                st.subheader("Feature Importance")
                feature_importance = pd.Series(model.feature_importances_, index=df.drop(columns=[target]).columns)
                fig = px.bar(feature_importance, title="Feature Importance")
                st.plotly_chart(fig)
            if algorithm in ["Random Forest", "XGBoost", "Neural Network"]:
                st.write(f"ROC-AUC Score: {roc_auc_score(y_test, model.predict_proba(X_test)[:,1]):.2f}")

        elif task == "Regression":
            st.write(f"You selected: Regression with {algorithm}")
            if algorithm == "Random Forest":
                model = RandomForestRegressor()
            elif algorithm == "XGBoost":
                model = XGBRegressor()
            elif algorithm == "SVM":
                model = SVR()
            elif algorithm == "Neural Network":
                model = MLPRegressor()
            
            # **Basic Hyperparameter Tuning for RandomForest**
            if algorithm == "Random Forest":
                params = {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [None, 5, 10]
                }
                grid_search = GridSearchCV(model, params, cv=3, n_jobs=-1)
                grid_search.fit(X_train, y_train)
                model = grid_search.best_estimator_
                st.write(f"Best Parameters: {grid_search.best_params_}")

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            rmse = mean_squared_error(y_test, y_pred, squared=False)
            mse = mean_squared_error(y_test, y_pred)
            st.write(f"Model RMSE: {rmse:.2f}")
            st.write(f"Model MSE: {mse:.2f}")
            st.write(f"Model R-Squared: {r2_score(y_test, y_pred):.2f}")

            if algorithm in ["Random Forest", "XGBoost"]:
                st.subheader("Feature Importance")
                feature_importance = pd.Series(model.feature_importances_, index=df.drop(columns=[target]).columns)
                fig = px.bar(feature_importance, title="Feature Importance")
                st.plotly_chart(fig)

            # **Make Predictions**
            st.subheader("Make Predictions")
            st.write("Input data for prediction:")
            user_input = {}
            for col in df.drop(columns=[target]).columns:
                user_input[col] = st.number_input(f"Value for {col}", value=0.0)    
            input_df = pd.DataFrame([user_input])

            if st.button("Predict"):
                if task == "Classification":
                    prediction = model.predict(input_df)
                    st.write(f"Prediction (Class): {prediction[0]}")
                    if algorithm in ["Random Forest", "XGBoost", "Neural Network"]:
                        prediction_proba = model.predict_proba(input_df)
                        st.write(f"Prediction Probabilities: {prediction_proba[0]}")
                elif task == "Regression":
                    prediction = model.predict(input_df)
                    st.write(f"Prediction (Value): {prediction[0]}")

    except Exception as e:
        show_error(f"Error processing file: {e}")
        st.stop()