import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor

st.title("ML-Powered Data Dashboard")
st.write("Upload a dataset, visualize it, and build machine learning models!")

uploaded_file = st.file_uploader("Upload your dataset (CSV or Excel)", type=["csv", "xlsx"])

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(".xlsx"):
            df = pd.read_excel(uploaded_file, engine="openpyxl")
        else:
            st.error("Unsupported file type. Please upload a CSV or Excel file.")
            st.stop()
    except Exception as e:
        st.error(f"Error loading file: {e}")
        st.stop()

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Dataset Info")
    st.write("Shape:", df.shape)
    st.write("Columns:", list(df.columns))
    st.write("Data Types:")
    st.write(df.dtypes)

    if st.checkbox("Drop rows with missing values"):
        df = df.dropna()
        st.write("Rows with missing values have been dropped.")
        st.write("Updated Shape:", df.shape)
    elif st.checkbox("Fill missing values with mean (numerical only)"):
        df = df.fillna(df.mean())
        st.write("Missing values have been filled with mean.")
        st.write("Updated Shape:", df.shape)

    st.subheader("Machine Learning Model Builder")

    target = st.selectbox("Select the target variable", df.columns)

    task = st.radio("Select Task Type", ["Classification", "Regression"])
    if task == "Classification":
        if pd.api.types.is_numeric_dtype(df[target]) and len(df[target].unique()) > 10:
            st.error(
                f"Selected task is Classification, but the target variable '{target}' appears to be continuous. "
                "Please select a categorical target variable or choose Regression."
            )
            st.stop()

        elif pd.api.types.is_numeric_dtype(df[target]):
            st.write(f"Target variable '{target}' is continuous. Binning it into discrete categories for Classification...")
            df[target] = pd.cut(df[target], bins=3, labels=[0, 1, 2])  # Example: 3 bins
            st.write("Binned target variable:", df[target].value_counts())

    elif task == "Regression":
        if not pd.api.types.is_numeric_dtype(df[target]):
            st.error(
                f"Selected task is Regression, but the target variable '{target}' is not numeric. "
                "Please select a numeric target variable or choose Classification."
            )
            st.stop()

    if df[target].dtype == "object":
        st.write(f"Target variable '{target}' is categorical. Encoding it...")
        label_encoder = LabelEncoder()
        df[target] = label_encoder.fit_transform(df[target])
        st.write("Encoded target variable classes:", list(label_encoder.classes_))

    features = df.drop(columns=[target])

    if features.select_dtypes(include="object").shape[1] > 0:
        features = pd.get_dummies(features)
        st.write("Encoded categorical features into numerical values.")

    X_train, X_test, y_train, y_test = train_test_split(
        features, df[target], test_size=0.2, random_state=42
    )

    if task == "Regression":
        if st.checkbox("Remove Outliers in Target"):
            Q1 = y_train.quantile(0.25)
            Q3 = y_train.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            inliers = (y_train >= lower_bound) & (y_train <= upper_bound)
            X_train, y_train = X_train[inliers], y_train[inliers]
            st.write(f"Removed outliers from target. New training data size: {len(y_train)}")

        if st.checkbox("Apply Log Transformation to Target"):
            y_train = np.log1p(y_train)  
            y_test = np.log1p(y_test)
            st.write("Log transformation applied to target variable.")

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    algorithm = st.radio("Select Algorithm", ["Random Forest", "XGBoost"])

    if task == "Classification":
        st.write(f"You selected: Classification with {algorithm}")
        if algorithm == "Random Forest":
            model = RandomForestClassifier()
        elif algorithm == "XGBoost":
            if y_train.min() != 0:
                y_train = y_train - y_train.min()
                y_test = y_test - y_test.min()

            model = XGBClassifier(use_label_encoder=False, eval_metric="logloss")

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        st.write(f"Model Accuracy: {accuracy:.2f}")

        st.subheader("Feature Importance")
        feature_importance = pd.Series(model.feature_importances_, index=features.columns)
        st.bar_chart(feature_importance)

    elif task == "Regression":
        st.write(f"You selected: Regression with {algorithm}")
        if algorithm == "Random Forest":
            model = RandomForestRegressor()
        elif algorithm == "XGBoost":
            model = XGBRegressor()

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        if st.checkbox("Reverse Log Transformation"):
            y_pred = np.expm1(y_pred)  
            y_test = np.expm1(y_test)

        rmse = mean_squared_error(y_test, y_pred, squared=False)
        mse = mean_squared_error(y_test, y_pred)
        st.write(f"Model RMSE: {rmse:.2f}")
        st.write(f"Model MSE: {mse:.2f}")

        st.subheader("Feature Importance")
        feature_importance = pd.Series(model.feature_importances_, index=features.columns)
        st.bar_chart(feature_importance)

    st.subheader("Make Predictions")
    st.write("Input data for prediction:")
    user_input = {}
    for col in features.columns:
        user_input[col] = st.number_input(f"Value for {col}", value=0.0)
    input_df = pd.DataFrame([user_input])

    if st.button("Predict"):
        prediction = model.predict(input_df)
        st.write(f"Prediction: {prediction[0]}")

else:
    st.write("Upload a dataset to get started.")