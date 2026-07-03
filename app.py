import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Churn Predictor", page_icon="📡", layout="centered")

@st.cache_resource
def load_models():
    preprocessor = joblib.load("models/preprocessor.pkl")
    lgbm         = joblib.load("models/lightgbm_balanced.pkl")
    lr           = joblib.load("models/logistic_regression_balanced.pkl")
    return preprocessor, lgbm, lr

preprocessor, lgbm, lr = load_models()

REQUIRED_COLS = [
    'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
    'InternetService', 'OnlineSecurity', 'OnlineBackup',
    'DeviceProtection', 'TechSupport', 'Contract',
    'PaperlessBilling', 'PaymentMethod', 'MonthlyCharges', 'auto_payment'
]

def predict(df, model):
    return model.predict_proba(preprocessor.transform(df[REQUIRED_COLS]))[:, 1]

def risk_label(p):
    if p > 0.7:   return "🔴 High risk"
    elif p > 0.4: return "🟡 Medium risk"
    else:         return "🟢 Low risk"

# ── Header ────────────────────────────────────────────────────────────────────
st.title("Churn Predictor")
st.caption("Telco Customer Churn  · Kaggle dataset · scikit-learn + LightGBM")
st.divider()

# ── Model selection ───────────────────────────────────────────────────────────
st.subheader("Model")

c1, c2 = st.columns(2)
with c1:
    with st.container(border=True):
        st.markdown("**LightGBM**")
        st.markdown("Better F1 (metric) : catches most churners without too many false alarms. " \
        "Use this if you want to prioritize the highest-risk customers only.")
        st.caption("Recall: The percentage of all actual churners the model catches")

with c2:
    with st.container(border=True):
        st.markdown("**Logistic Regression**")
        st.markdown("Higher recall — flags more customers as at-risk, including some who wouldn't have churned. Use this if missing a churner is the worst outcome.")
        st.caption("F1: A balanced rating combining Recall (catching churners) and Precision (avoiding false alarms) into one score.")

model_choice = st.radio(
    "Pick one",
    ["LightGBM", "Logistic Regression"],
    horizontal=True,
    label_visibility="collapsed"
)
model = lgbm if model_choice == "LightGBM" else lr

st.caption(
    "Risk thresholds: > 70% = high | 40–70% = medium | < 40% = low. " \
    "These cutoffs are a starting point, in practice they'd be calibrated to the cost of a retention action vs. losing a customer."
)

st.divider()

# ── Input mode ────────────────────────────────────────────────────────────────
st.subheader("Customer data")
tab_manual, tab_csv = st.tabs(["Manual entry", "Upload CSV"])

# ── MANUAL ───────────────────────────────────────────────────────────────────
with tab_manual:
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("**Contract**")
        contract        = st.selectbox("Contract type",     ["Month-to-month", "One year", "Two year"])
        payment         = st.selectbox("Payment method",    ["Electronic check", "Mailed check",
                                                             "Bank transfer (automatic)", "Credit card (automatic)"])
        paperless       = st.selectbox("Paperless billing", ["Yes", "No"])
        monthly_charges = st.slider("Monthly charges ($)",  18, 120, 65)

    with c2:
        st.markdown("**Services**")
        internet        = st.selectbox("Internet",          ["DSL", "Fiber optic", "No"])
        online_security = st.selectbox("Online security",   ["Yes", "No", "No internet service"])
        online_backup   = st.selectbox("Online backup",     ["Yes", "No", "No internet service"])
        device_prot     = st.selectbox("Device protection", ["Yes", "No", "No internet service"])
        tech_support    = st.selectbox("Tech support",      ["Yes", "No", "No internet service"])

    with c3:
        st.markdown("**Profile**")
        tenure          = st.slider("Tenure (months)",      0, 72, 12)
        senior          = st.selectbox("Senior citizen",    [0, 1], format_func=lambda x: "Yes" if x else "No")
        partner         = st.selectbox("Partner",           ["Yes", "No"])
        dependents      = st.selectbox("Dependents",        ["Yes", "No"])

    if st.button("Run prediction", type="primary"):
        input_df = pd.DataFrame([{
            'SeniorCitizen':    senior,
            'Partner':          partner,
            'Dependents':       dependents,
            'tenure':           tenure,
            'InternetService':  internet,
            'OnlineSecurity':   online_security,
            'OnlineBackup':     online_backup,
            'DeviceProtection': device_prot,
            'TechSupport':      tech_support,
            'Contract':         contract,
            'PaperlessBilling': paperless,
            'PaymentMethod':    payment,
            'MonthlyCharges':   monthly_charges,
            'auto_payment':     1 if "automatic" in payment else 0
        }])

        proba = predict(input_df, model)[0]
        st.divider()

        _, center, _ = st.columns([1, 2, 1])
        with center:
            st.metric("Churn probability", f"{proba:.1%}")
            st.markdown(f"### {risk_label(proba)}")
 
            if proba > 0.7:
                st.error("This customer matches the high-churn profile closely — short tenure, high bill, or no long-term contract.")
            elif proba > 0.4:
                st.warning("Some churn signals present but not conclusive.")
            else:
                st.success("No strong churn signals detected.")

# ── CSV ───────────────────────────────────────────────────────────────────────
with tab_csv:
    st.markdown("Upload a CSV with one row per customer to score a batch.")

    with st.expander("Required columns"):
        st.code(", ".join(REQUIRED_COLS))

    uploaded = st.file_uploader("CSV file", type=["csv"])

    if uploaded:
        df_up   = pd.read_csv(uploaded)
        missing = [c for c in REQUIRED_COLS if c not in df_up.columns]

        if missing:
            st.error(f"Missing columns: {missing}")
        else:
            probas = predict(df_up, model)
            df_up["Churn probability"] = (probas * 100).round(1).astype(str) + "%"
            df_up["Risk"]              = [risk_label(p) for p in probas]

            high   = (probas > 0.7).sum()
            medium = ((probas > 0.4) & (probas <= 0.7)).sum()
            low    = (probas <= 0.4).sum()

            m1, m2, m3 = st.columns(3)
            m1.metric("High risk",   high)
            m2.metric("Medium risk", medium)
            m3.metric("Low risk",    low)

            st.dataframe(
                df_up[["Churn probability", "Risk"] + REQUIRED_COLS],
                use_container_width=True
            )

st.divider()
st.caption("Thresholds: > 70% high · 40–70% medium · < 40% low")
