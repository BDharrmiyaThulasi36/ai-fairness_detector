# streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    accuracy_score,
    mean_squared_error,
    r2_score,
)
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

st.set_page_config(page_title="Bias Detection & Fairness Auditor", page_icon="⚖️", layout="wide")

# ---------------------------
# Sidebar Navigation
# ---------------------------
st.sidebar.title("⚙️ Audit Settings")
st.sidebar.write("Configure your fairness audit.")

# ---------------------------
# Main Title
# ---------------------------
st.title("⚖️ Bias Detection & Fairness Auditor")
st.markdown("A Responsible AI tool to **detect, measure, and report bias** in ML models.")

# File upload
uploaded_file = st.file_uploader("📂 Upload your dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("✅ File uploaded successfully!")
    st.write("### 🔎 Data Preview")
    st.dataframe(df.head())

    # Select target + sensitive attribute
    target_col = st.selectbox("🎯 Select target variable (label):", df.columns)
    sensitive_col = st.selectbox("🧑‍🤝‍🧑 Select sensitive attribute to audit:", df.columns)

    # Model Choice
    model_choice = st.sidebar.radio("Select Model", ["Logistic Regression", "Decision Tree", "Linear Regression"])

    if st.button("🚀 Run Fairness Audit"):
        # Prepare data
        X = df.drop(columns=[target_col])
        y = df[target_col]

        # --- FIX 1: Generalize label encoding for any categorical target ---
        if y.dtype == "object" or y.dtype.name == "category":
            le = LabelEncoder()
            y = le.fit_transform(y)
            st.info(f"ℹ️ Target column '{target_col}' encoded as numeric using LabelEncoder.")

        # Encode features (dummy variables for categorical)
        X = pd.get_dummies(X, drop_first=True)

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        # --- FIX 2: Ensure y_train and y_test are pandas Series with indices ---
        y_train = pd.Series(y_train, index=X_train.index)
        y_test = pd.Series(y_test, index=X_test.index)

        # ---------------------------
        # Detect classification vs regression
        # ---------------------------
        unique_vals = len(np.unique(y))
        is_classification = unique_vals <= 10  # heuristic: 10 or fewer categories → classification

        if is_classification:
            if model_choice == "Logistic Regression":
                model = LogisticRegression(max_iter=1000)
            else:
                model = DecisionTreeClassifier(max_depth=5)
        else:
            model = LinearRegression()

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # ---------------------------
        # Overall Performance
        # ---------------------------
        st.subheader("📊 Overall Model Performance")

        if is_classification:
            y_pred_labels = (
                np.round(y_pred).astype(int) if y_pred.ndim == 1 else np.argmax(y_pred, axis=1)
            )

            overall_metrics = {
                "Accuracy": accuracy_score(y_test, y_pred_labels),
                "Precision": precision_score(y_test, y_pred_labels, average="weighted", zero_division=0),
                "Recall": recall_score(y_test, y_pred_labels, average="weighted", zero_division=0),
            }
        else:
            overall_metrics = {
                "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
                "R²": r2_score(y_test, y_pred),
            }

        st.write(overall_metrics)

        # ---------------------------
        # Fairness Audit by Group
        # ---------------------------
        st.subheader(f"🔍 Fairness Audit by {sensitive_col}")
        results = []

        # Align predictions with X_test indices
        y_pred_series = pd.Series(y_pred, index=X_test.index)

        for group in df[sensitive_col].unique():
            idx = X_test.index[df.loc[X_test.index, sensitive_col] == group]

            y_true_group = y_test.loc[idx]
            y_pred_group = y_pred_series.loc[idx]

            if is_classification:
                y_pred_group = np.round(y_pred_group).astype(int)
                acc = accuracy_score(y_true_group, y_pred_group)
                prec = precision_score(y_true_group, y_pred_group, average="weighted", zero_division=0)
                rec = recall_score(y_true_group, y_pred_group, average="weighted", zero_division=0)
                positive_rate = np.mean(y_pred_group)
                results.append([group, acc, prec, rec, positive_rate])
            else:
                mse = mean_squared_error(y_true_group, y_pred_group)
                r2 = r2_score(y_true_group, y_pred_group)
                avg_pred = np.mean(y_pred_group)
                results.append([group, mse, r2, avg_pred])

        if is_classification:
            fairness_df = pd.DataFrame(
                results,
                columns=[sensitive_col, "Accuracy", "Precision", "Recall", "Positive Rate"],
            )
        else:
            fairness_df = pd.DataFrame(
                results,
                columns=[sensitive_col, "MSE", "R²", "Average Prediction"],
            )

        st.dataframe(fairness_df)

        # ---------------------------
        # Advanced Fairness Metrics (classification only)
        # ---------------------------
        if is_classification and not fairness_df.empty:
            st.subheader("⚖️ Advanced Fairness Metrics")

            groups = fairness_df[sensitive_col].tolist()
            pos_rates = fairness_df["Positive Rate"].tolist()
            recalls = fairness_df["Recall"].tolist()

            dp_diff = max(pos_rates) - min(pos_rates)
            eo_diff = max(recalls) - min(recalls)
            sp_ratio = min(pos_rates) / max(pos_rates) if max(pos_rates) > 0 else 0

            adv_metrics = {
                "Demographic Parity Difference": round(dp_diff, 3),
                "Equal Opportunity Difference": round(eo_diff, 3),
                "Statistical Parity Ratio": round(sp_ratio, 3),
            }
            st.write(adv_metrics)

        # ---------------------------
        # Visualizations
        # ---------------------------
        st.subheader("📈 Group Comparison Charts")

        if not fairness_df.empty:
            if is_classification:
                st.bar_chart(
                    fairness_df.set_index(sensitive_col)[["Accuracy", "Precision", "Recall"]]
                )
            else:
                st.bar_chart(
                    fairness_df.set_index(sensitive_col)[["MSE", "R²", "Average Prediction"]]
                )

        # ---------------------------
        # Export Report
        # ---------------------------
        st.download_button(
            "💾 Download Audit Report (CSV)",
            fairness_df.to_csv(index=False).encode("utf-8"),
            file_name="fairness_audit.csv",
            mime="text/csv",
        )

