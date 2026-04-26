import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    confusion_matrix, accuracy_score,
    mean_absolute_error, mean_squared_error, r2_score
)

# ──────────────────────────────────────────────
# Page Config
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Osteoporosis Risk Predictor",
    page_icon="🦴",
    layout="wide",
)

st.title("🦴 Osteoporosis Risk Prediction")
st.markdown(
    "Upload the **osteoporosis.csv** dataset to train models and predict patient risk."
)

# ──────────────────────────────────────────────
# Sidebar – File Upload
# ──────────────────────────────────────────────
st.sidebar.header("📂 Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload osteoporosis.csv", type=["csv"])

# ──────────────────────────────────────────────
# Helper: encode + split
# ──────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def preprocess(df_raw: pd.DataFrame):
    df = df_raw.copy()

    # Drop ID column if present
    for col in ["Id", "ID", "id"]:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    # Fill missing values
    df.fillna("None", inplace=True)

    # Label-encode categoricals
    le = LabelEncoder()
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = le.fit_transform(df[col])

    X = df.drop("Osteoporosis", axis=1)
    y = df["Osteoporosis"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=101
    )
    return df, X, y, X_train, X_test, y_train, y_test


@st.cache_data(show_spinner=False)
def train_models(X_train, y_train):
    lr = LogisticRegression(
        C=0.1, max_iter=100, penalty="l2",
        random_state=0, solver="liblinear", multi_class="auto"
    )
    rfc = RandomForestClassifier(
        criterion="gini", max_depth=10,
        min_samples_leaf=2, min_samples_split=2, random_state=42
    )
    dtree = DecisionTreeClassifier(
        criterion="entropy", max_depth=10,
        min_samples_leaf=10, min_samples_split=2, random_state=0
    )
    svc = SVC(C=0.1, degree=2, gamma="auto", random_state=0, kernel="linear")

    lr.fit(X_train, y_train)
    rfc.fit(X_train, y_train)
    dtree.fit(X_train, y_train)
    svc.fit(X_train, y_train)

    return {"Logistic Regression": lr,
            "Random Forest": rfc,
            "Decision Tree": dtree,
            "SVC": svc}


# ──────────────────────────────────────────────
# Main Logic
# ──────────────────────────────────────────────
if uploaded_file is None:
    st.info("👈  Upload **osteoporosis.csv** in the sidebar to get started.")
    st.stop()

df_raw = pd.read_csv(uploaded_file)

tabs = st.tabs([
    "📊 Data Overview",
    "🔍 EDA",
    "🤖 Model Training & Evaluation",
    "🔮 Predict Risk",
])

# ══════════════════════════════════════════════
# TAB 1 – Data Overview
# ══════════════════════════════════════════════
with tabs[0]:
    st.subheader("Raw Data")
    st.dataframe(df_raw.head(20), use_container_width=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", df_raw.shape[0])
    col2.metric("Columns", df_raw.shape[1])
    col3.metric("Missing Values", int(df_raw.isnull().sum().sum()))

    st.subheader("Descriptive Statistics")
    st.dataframe(df_raw.describe(), use_container_width=True)

    st.subheader("Missing Value %")
    missing = (df_raw.isnull().sum() / len(df_raw) * 100).reset_index()
    missing.columns = ["Column", "Missing %"]
    missing = missing[missing["Missing %"] > 0]
    if missing.empty:
        st.success("No missing values found.")
    else:
        st.dataframe(missing, use_container_width=True)

# ══════════════════════════════════════════════
# TAB 2 – EDA
# ══════════════════════════════════════════════
with tabs[1]:
    df_raw_filled = df_raw.copy()
    for col in ["Id", "ID", "id"]:
        if col in df_raw_filled.columns:
            df_raw_filled.drop(columns=[col], inplace=True)
    df_raw_filled.fillna("None", inplace=True)

    # Target distribution
    st.subheader("Target Variable Distribution")
    fig, ax = plt.subplots(figsize=(4, 4))
    df_raw_filled["Osteoporosis"].value_counts().plot.pie(
        autopct="%1.1f%%", ax=ax, labels=["No", "Yes"]
    )
    ax.set_title("Osteoporosis Distribution")
    ax.set_ylabel("")
    st.pyplot(fig)
    plt.close()

    # Age distribution
    st.subheader("Age vs Osteoporosis")
    fig, ax = plt.subplots(figsize=(8, 4))
    df_raw_filled[df_raw_filled["Osteoporosis"] == 1]["Age"].plot.hist(
        bins=30, alpha=0.5, color="steelblue", label="Osteoporosis = Yes", ax=ax
    )
    df_raw_filled[df_raw_filled["Osteoporosis"] == 0]["Age"].plot.hist(
        bins=30, alpha=0.5, color="salmon", label="Osteoporosis = No", ax=ax
    )
    ax.legend(); ax.set_xlabel("Age"); ax.set_title("Osteoporosis by Age")
    st.pyplot(fig); plt.close()

    # Categorical variables
    cat_pairs = [
        ("Gender", "Gender vs Osteoporosis"),
        ("Hormonal Changes", "Hormonal Changes vs Osteoporosis"),
        ("Family History", "Family History vs Osteoporosis"),
        ("Race/Ethnicity", "Race/Ethnicity vs Osteoporosis"),
        ("Body Weight", "Body Weight vs Osteoporosis"),
        ("Physical Activity", "Physical Activity vs Osteoporosis"),
        ("Smoking", "Smoking vs Osteoporosis"),
        ("Alcohol Consumption", "Alcohol Consumption vs Osteoporosis"),
        ("Medical Conditions", "Medical Conditions vs Osteoporosis"),
        ("Medications", "Medications vs Osteoporosis"),
        ("Prior Fractures", "Prior Fractures vs Osteoporosis"),
    ]

    # Nutrition side-by-side
    st.subheader("Nutrition vs Osteoporosis")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax_i, col in zip(axes, ["Calcium Intake", "Vitamin D Intake"]):
        if col in df_raw_filled.columns:
            sns.countplot(x=col, data=df_raw_filled, hue="Osteoporosis", ax=ax_i)
            ax_i.set_title(f"{col} vs Osteoporosis")
            ax_i.tick_params(axis="x", rotation=15)
    st.pyplot(fig); plt.close()

    # All other categoricals
    st.subheader("Other Feature Distributions")
    available = [(c, t) for c, t in cat_pairs if c in df_raw_filled.columns
                 and c not in ["Calcium Intake", "Vitamin D Intake"]]
    for i in range(0, len(available), 2):
        cols = st.columns(2)
        for j, (col, title) in enumerate(available[i:i+2]):
            with cols[j]:
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.countplot(x=col, data=df_raw_filled, hue="Osteoporosis", ax=ax)
                ax.set_title(title)
                ax.tick_params(axis="x", rotation=15)
                st.pyplot(fig); plt.close()

    # Correlation heatmap
    st.subheader("Correlation Matrix")
    df_encoded = df_raw_filled.copy()
    le = LabelEncoder()
    for col in df_encoded.select_dtypes(include=["object"]).columns:
        df_encoded[col] = le.fit_transform(df_encoded[col])
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(df_encoded.corr(), annot=True, fmt=".2f", ax=ax, cmap="coolwarm")
    st.pyplot(fig); plt.close()

# ══════════════════════════════════════════════
# TAB 3 – Model Training & Evaluation
# ══════════════════════════════════════════════
with tabs[2]:
    st.subheader("Training Models")

    with st.spinner("Preprocessing data and training 4 models…"):
        df_enc, X, y, X_train, X_test, y_train, y_test = preprocess(df_raw)
        models = train_models(X_train, y_train)

    st.success("✅ All models trained successfully!")

    # Predictions
    preds = {name: m.predict(X_test) for name, m in models.items()}

    # Accuracy bar chart
    st.subheader("Model Accuracy Comparison")
    acc = {name: accuracy_score(y_test, p) for name, p in preds.items()}
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(x=list(acc.keys()), y=list(acc.values()), ax=ax, palette="Blues_d")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Accuracy")
    ax.set_title("Model Accuracy")
    for i, v in enumerate(acc.values()):
        ax.text(i, v + 0.01, f"{v:.2%}", ha="center", fontweight="bold")
    st.pyplot(fig); plt.close()

    # Confusion matrices
    st.subheader("Confusion Matrices")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for ax_i, (name, p) in zip(axes.flatten(), preds.items()):
        cm = confusion_matrix(y_test, p)
        sns.heatmap(cm, annot=True, fmt="g", ax=ax_i, cmap="Blues")
        ax_i.set_title(name)
        ax_i.set_xlabel("Predicted"); ax_i.set_ylabel("Actual")
    plt.tight_layout()
    st.pyplot(fig); plt.close()

    # Metrics table
    st.subheader("Regression-Style Error Metrics")
    rows = []
    for name, p in preds.items():
        rows.append({
            "Model": name,
            "Accuracy": f"{accuracy_score(y_test, p):.4f}",
            "MAE": f"{mean_absolute_error(y_test, p):.4f}",
            "MSE": f"{mean_squared_error(y_test, p):.4f}",
            "RMSE": f"{np.sqrt(mean_squared_error(y_test, p)):.4f}",
            "R² Score": f"{r2_score(y_test, p):.4f}",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    # Feature importance
    st.subheader("Feature Importance")
    feature_cols = list(X.columns)
    importance_models = {
        "Logistic Regression": models["Logistic Regression"].coef_[0],
        "Random Forest": models["Random Forest"].feature_importances_,
        "Decision Tree": models["Decision Tree"].feature_importances_,
        "SVC": models["SVC"].coef_[0],
    }
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    for ax_i, (name, coeff) in zip(axes.flatten(), importance_models.items()):
        feat_df = pd.DataFrame({"Feature": feature_cols, "Importance": coeff})
        feat_df.sort_values("Importance", inplace=True)
        ax_i.barh(feat_df["Feature"], feat_df["Importance"])
        ax_i.set_title(f"Feature Importance – {name}")
        ax_i.set_xlabel("Importance")
    plt.tight_layout()
    st.pyplot(fig); plt.close()

# ══════════════════════════════════════════════
# TAB 4 – Predict Risk
# ══════════════════════════════════════════════
with tabs[3]:
    st.subheader("🔮 Predict Osteoporosis Risk for a New Patient")
    st.markdown(
        "Fill in the patient details below. The app will run all four trained "
        "models and show you the risk prediction."
    )

    # Ensure models are trained
    if "models" not in dir():
        with st.spinner("Training models…"):
            df_enc, X, y, X_train, X_test, y_train, y_test = preprocess(df_raw)
            models = train_models(X_train, y_train)

    # Build input widgets that mirror the dataset columns (after encoding)
    # We recreate label encoders so we can encode user inputs consistently
    df_tmp = df_raw.copy()
    for col_drop in ["Id", "ID", "id"]:
        if col_drop in df_tmp.columns:
            df_tmp.drop(columns=[col_drop], inplace=True)
    df_tmp.fillna("None", inplace=True)

    encoders = {}
    for col in df_tmp.select_dtypes(include=["object"]).columns:
        le = LabelEncoder()
        le.fit(df_tmp[col])
        encoders[col] = le

    feature_columns = [c for c in df_tmp.columns if c != "Osteoporosis"]

    with st.form("patient_form"):
        cols_left, cols_right = st.columns(2)
        user_inputs = {}

        for i, col in enumerate(feature_columns):
            target_col = cols_left if i % 2 == 0 else cols_right
            with target_col:
                if col == "Age":
                    user_inputs[col] = st.number_input("Age", min_value=1, max_value=120, value=50)
                elif col in encoders:
                    options = list(encoders[col].classes_)
                    user_inputs[col] = st.selectbox(col, options)
                else:
                    user_inputs[col] = st.number_input(col, value=0)

        submitted = st.form_submit_button("🔍 Predict", use_container_width=True)

    if submitted:
        # Build input vector
        input_row = {}
        for col in feature_columns:
            val = user_inputs[col]
            if col in encoders:
                val = int(encoders[col].transform([val])[0])
            input_row[col] = val

        input_df = pd.DataFrame([input_row])

        st.markdown("---")
        st.subheader("Prediction Results")

        result_cols = st.columns(4)
        for col_widget, (name, model) in zip(result_cols, models.items()):
            pred = model.predict(input_df)[0]
            label = "⚠️ At Risk" if pred == 1 else "✅ Low Risk"
            color = "#ff4b4b" if pred == 1 else "#21c55d"
            col_widget.markdown(
                f"""
                <div style='text-align:center; padding:16px; border-radius:10px;
                            border: 2px solid {color};'>
                  <p style='font-size:13px; color:gray; margin:0'>{name}</p>
                  <p style='font-size:22px; font-weight:bold; color:{color}; margin:4px 0'>{label}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Majority vote summary
        all_preds = [int(m.predict(input_df)[0]) for m in models.values()]
        majority = "⚠️ **HIGH RISK** of Osteoporosis" if sum(all_preds) >= 2 else "✅ **LOW RISK** of Osteoporosis"
        st.markdown(f"### Consensus: {majority}")
        st.caption(
            f"{sum(all_preds)} out of {len(all_preds)} models predict positive osteoporosis risk."
        )
