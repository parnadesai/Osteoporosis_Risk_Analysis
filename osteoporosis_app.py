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
        random_state=0, solver="liblinear"
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
# ══════════════════════════════════════════════
# TAB 4 – Predict Risk
# ══════════════════════════════════════════════
with tabs[3]:
    st.subheader("🔮 Predict Osteoporosis Risk for a New Patient")
    st.markdown(
        "Fill in the patient details below. The app will run all four trained "
        "models and return a detailed risk assessment."
    )

    # Ensure models are trained
    if "models" not in dir():
        with st.spinner("Training models…"):
            df_enc, X, y, X_train, X_test, y_train, y_test = preprocess(df_raw)
            models = train_models(X_train, y_train)

    # ── Rebuild encoders ──────────────────────────────────────────
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

    # ── Gender watch (outside form so it re-renders reactively) ───
    gender_options = list(encoders["Gender"].classes_) if "Gender" in encoders else ["Female", "Male"]
    selected_gender = st.selectbox("Gender", gender_options, key="gender_selector")

    # Gender-conditional hormonal options
    if selected_gender == "Female":
        hormonal_options = ["Postmenopausal", "Premenopausal", "None"]
    else:
        hormonal_options = ["Low Testosterone", "Normal", "None"]

    # Filter to only valid classes the encoder knows
    if "Hormonal Changes" in encoders:
        known = list(encoders["Hormonal Changes"].classes_)
        hormonal_options = [o for o in hormonal_options if o in known] or known

    with st.form("patient_form"):

        # ── Section 1: Demographics ───────────────────────────────
        st.markdown("#### 👤 Demographics")
        d1, d2, d3 = st.columns(3)
        user_inputs = {}

        user_inputs["Gender"] = selected_gender   # captured above

        with d1:
            user_inputs["Age"] = st.number_input(
                "Age", min_value=1, max_value=120, value=50
            )
        with d2:
            if "Race/Ethnicity" in encoders:
                user_inputs["Race/Ethnicity"] = st.selectbox(
                    "Race / Ethnicity", list(encoders["Race/Ethnicity"].classes_)
                )
        with d3:
            if "Body Weight" in encoders:
                user_inputs["Body Weight"] = st.selectbox(
                    "Body Weight", list(encoders["Body Weight"].classes_)
                )

        # ── Section 2: Hormonal & Family ─────────────────────────
        st.markdown("#### 🧬 Hormonal & Genetic Factors")
        h1, h2 = st.columns(2)
        with h1:
            if "Hormonal Changes" in encoders:
                user_inputs["Hormonal Changes"] = st.selectbox(
                    "Hormonal Changes", hormonal_options,
                    help="Options change based on the selected gender above."
                )
        with h2:
            if "Family History" in encoders:
                user_inputs["Family History"] = st.selectbox(
                    "Family History of Osteoporosis",
                    list(encoders["Family History"].classes_)
                )

        # ── Section 3: Lifestyle ──────────────────────────────────
        st.markdown("#### 🏃 Lifestyle")
        l1, l2, l3 = st.columns(3)
        with l1:
            if "Physical Activity" in encoders:
                user_inputs["Physical Activity"] = st.selectbox(
                    "Physical Activity", list(encoders["Physical Activity"].classes_)
                )
        with l2:
            if "Smoking" in encoders:
                user_inputs["Smoking"] = st.selectbox(
                    "Smoking", list(encoders["Smoking"].classes_)
                )
        with l3:
            if "Alcohol Consumption" in encoders:
                user_inputs["Alcohol Consumption"] = st.selectbox(
                    "Alcohol Consumption", list(encoders["Alcohol Consumption"].classes_)
                )

        # ── Section 4: Medical ────────────────────────────────────
        st.markdown("#### 🏥 Medical History")
        m1, m2, m3 = st.columns(3)
        with m1:
            if "Medical Conditions" in encoders:
                user_inputs["Medical Conditions"] = st.selectbox(
                    "Medical Conditions", list(encoders["Medical Conditions"].classes_)
                )
        with m2:
            if "Medications" in encoders:
                user_inputs["Medications"] = st.selectbox(
                    "Medications", list(encoders["Medications"].classes_)
                )
        with m3:
            if "Prior Fractures" in encoders:
                user_inputs["Prior Fractures"] = st.selectbox(
                    "Prior Fractures", list(encoders["Prior Fractures"].classes_)
                )

        # ── Section 5: Nutrition ──────────────────────────────────
        st.markdown("#### 🥗 Nutrition")
        n1, n2 = st.columns(2)
        with n1:
            if "Calcium Intake" in encoders:
                user_inputs["Calcium Intake"] = st.selectbox(
                    "Calcium Intake", list(encoders["Calcium Intake"].classes_)
                )
        with n2:
            if "Vitamin D Intake" in encoders:
                user_inputs["Vitamin D Intake"] = st.selectbox(
                    "Vitamin D Intake", list(encoders["Vitamin D Intake"].classes_)
                )

        # Catch any remaining feature columns not yet handled
        remaining = [c for c in feature_columns if c not in user_inputs]
        if remaining:
            st.markdown("#### 📋 Other Factors")
            rem_cols = st.columns(min(len(remaining), 3))
            for i, col in enumerate(remaining):
                with rem_cols[i % 3]:
                    if col in encoders:
                        user_inputs[col] = st.selectbox(col, list(encoders[col].classes_))
                    else:
                        user_inputs[col] = st.number_input(col, value=0)

        submitted = st.form_submit_button("🔍 Run Risk Assessment", use_container_width=True)

    # ── Results ───────────────────────────────────────────────────
    if submitted:
        # Build encoded input vector
        input_row = {}
        for col in feature_columns:
            val = user_inputs.get(col, 0)
            if col in encoders:
                val = int(encoders[col].transform([val])[0])
            input_row[col] = val
        input_df = pd.DataFrame([input_row])

        all_preds  = {name: int(m.predict(input_df)[0]) for name, m in models.items()}
        risk_votes = sum(all_preds.values())
        total      = len(all_preds)
        risk_pct   = risk_votes / total * 100

        st.markdown("---")
        st.subheader("📋 Risk Assessment Results")

        # ── Overall verdict ───────────────────────────────────────
        if risk_votes == 0:
            verdict_color = "#21c55d"; verdict_label = "✅ LOW RISK"
            verdict_msg   = "All models agree: low probability of osteoporosis."
        elif risk_votes <= total // 2:
            verdict_color = "#f59e0b"; verdict_label = "⚡ BORDERLINE"
            verdict_msg   = "Mixed signals — consider a clinical bone-density scan."
        else:
            verdict_color = "#ef4444"; verdict_label = "⚠️ HIGH RISK"
            verdict_msg   = "Majority of models flag elevated osteoporosis risk."

        st.markdown(
            f"""
            <div style='padding:20px 24px; border-radius:12px; border:2px solid {verdict_color};
                        background:{"rgba(239,68,68,0.07)" if risk_votes > total//2 else "rgba(33,197,93,0.07)"};
                        margin-bottom:16px'>
              <h2 style='color:{verdict_color}; margin:0'>{verdict_label}</h2>
              <p style='margin:6px 0 0; color:#555'>{verdict_msg}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ── Vote bar ──────────────────────────────────────────────
        st.markdown("**Model Consensus**")
        bar_html = ""
        for name, pred in all_preds.items():
            col_c = "#ef4444" if pred == 1 else "#21c55d"
            seg_label = "At Risk" if pred == 1 else "Low Risk"
            bar_html += (
                f"<div style='flex:1; background:{col_c}; color:white; text-align:center;"
                f"padding:8px 4px; font-size:12px; font-weight:600;'>"
                f"{name}<br>{seg_label}</div>"
            )
        st.markdown(
            f"<div style='display:flex; border-radius:8px; overflow:hidden; gap:2px'>{bar_html}</div>",
            unsafe_allow_html=True,
        )
        st.caption(f"{risk_votes} of {total} models predict positive osteoporosis risk ({risk_pct:.0f}%)")

        st.markdown("---")

        # ── Per-model cards ───────────────────────────────────────
        st.markdown("**Individual Model Verdicts**")
        card_cols = st.columns(4)
        model_notes = {
            "Logistic Regression": "Linear boundary — good baseline",
            "Random Forest":       "Ensemble of trees — robust to noise",
            "Decision Tree":       "Rule-based — highly interpretable",
            "SVC":                 "Margin classifier — strong on structure",
        }
        for col_widget, (name, pred) in zip(card_cols, all_preds.items()):
            color = "#ef4444" if pred == 1 else "#21c55d"
            icon  = "⚠️" if pred == 1 else "✅"
            label = "At Risk" if pred == 1 else "Low Risk"
            note  = model_notes.get(name, "")
            col_widget.markdown(
                f"""
                <div style='border:1.5px solid {color}; border-radius:10px;
                            padding:14px 12px; text-align:center; height:130px'>
                  <div style='font-size:11px; color:#888; margin-bottom:4px'>{name}</div>
                  <div style='font-size:26px'>{icon}</div>
                  <div style='font-size:15px; font-weight:700; color:{color}'>{label}</div>
                  <div style='font-size:10px; color:#aaa; margin-top:6px'>{note}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # ── Clinical summary ──────────────────────────────────────
        st.markdown("---")
        st.markdown("**📌 Patient Summary**")

        display_map = {
            "Age": user_inputs.get("Age"),
            "Gender": user_inputs.get("Gender"),
            "Hormonal Changes": user_inputs.get("Hormonal Changes"),
            "Family History": user_inputs.get("Family History"),
            "Body Weight": user_inputs.get("Body Weight"),
            "Physical Activity": user_inputs.get("Physical Activity"),
            "Calcium Intake": user_inputs.get("Calcium Intake"),
            "Vitamin D Intake": user_inputs.get("Vitamin D Intake"),
            "Prior Fractures": user_inputs.get("Prior Fractures"),
            "Smoking": user_inputs.get("Smoking"),
            "Alcohol Consumption": user_inputs.get("Alcohol Consumption"),
        }
        summary_df = pd.DataFrame(
            [(k, v) for k, v in display_map.items() if v is not None],
            columns=["Factor", "Value"]
        )
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

        st.info(
            "⚕️ **Disclaimer:** This tool is for educational/research purposes only. "
            "Predictions are not a substitute for professional medical diagnosis. "
            "Consult a physician or bone-density specialist for clinical evaluation."
        )
        
