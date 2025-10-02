import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ------------------ Page Configuration ------------------
st.set_page_config(
    page_title="Financial Fraud Detection",
    layout="wide"
)

# ------------------ Styling (Dark Mode + Background Image Overlay) ------------------
page_bg = """
<style>
[data-testid="stAppViewContainer"] {
    position: relative;
    z-index: 0;
}
[data-testid="stAppViewContainer"]::before {
    content: "";
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background-image: url("https://i.pinimg.com/1200x/5f/64/17/5f641707a86728ad7ec25ac23d188384.jpg");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    opacity: 0.08; /* Very dark image */
    z-index: -1;
}
[data-testid="stSidebar"], [data-testid="stHeader"] {
    background: rgba(17, 24, 39, 0.9);
    color: #f8fafc;
    backdrop-filter: blur(6px);
}
[data-testid="stMarkdownContainer"] h1,
[data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3,
[data-testid="stMarkdownContainer"] h4,
[data-testid="stMarkdownContainer"] h5 {
    color: #E0E7FF;
    font-weight: 700;
}
label, .stTextInput, .stNumberInput, .stSelectbox, .stFileUploader {
    color: #f1f5f9 !important;
}
input, textarea {
    background-color: #1f2937 !important;
    color: #f8fafc !important;
    border: 1px solid #374151 !important;
}
.stButton>button {
    background: linear-gradient(90deg, #1E3A8A, #3B82F6);
    color: white;
    font-size: 16px;
    font-weight: 600;
    border-radius: 10px;
    padding: 0.6em 1.2em;
    border: none;
}
.stButton>button:hover {
    background: linear-gradient(90deg, #3B82F6, #60A5FA);
    transform: scale(1.02);
}
[data-testid="stDataFrame"] {
    background-color: rgba(30, 41, 59, 0.9);
    color: #f8fafc;
    border-radius: 10px;
}
section[data-testid="stSidebar"] .css-1v0mbdj {
    color: #f8fafc !important;
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# ------------------ Load Model ------------------
try:
    if os.path.exists("catboost.pkl"):
        best_model = joblib.load("catboost.pkl")
        MODEL_LOADED = True
    else:
        st.warning("⚠️ Model file 'catboost.pkl' not found. A dummy model will be used.")
        class DummyModel:
            def predict_proba(self, X):
                return np.array([[0.99, 0.01]] * len(X))
        best_model = DummyModel()
        MODEL_LOADED = False
except Exception as e:
    st.error(f"❌ Failed to load model: {e}")
    best_model = None
    MODEL_LOADED = False

# ------------------ Page Header ------------------
st.markdown("<h1 style='text-align: center;'>Financial Fraud Detection Tool</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size:18px; color:#cbd5e1;'>Evaluate individual or batch transactions for potential fraud.</p>", unsafe_allow_html=True)
st.markdown("---")

# ------------------ Input Preprocessing ------------------
def preprocess_input(df: pd.DataFrame) -> pd.DataFrame:
    required_columns = ['step', 'type', 'amount', 'oldbalanceOrg', 'newbalanceOrig',
                        'oldbalanceDest', 'newbalanceDest', 'nameDest']
    for col in required_columns:
        if col not in df.columns:
            df[col] = 0.0 if col not in ['type', 'nameDest'] else ('TRANSFER' if col == 'type' else 'C00000')

    df['balanceDiffOrig'] = df['oldbalanceOrg'] - df['newbalanceOrig']
    df['balanceDiffDest'] = df['oldbalanceDest'] - df['newbalanceDest']
    df['destIsMerchant'] = df['nameDest'].astype(str).str.startswith("M").astype(int)
    df['senderTxnCount'] = 0
    df['receiverTxnCount'] = 0

    df = pd.get_dummies(df, columns=["type"], prefix="type")

    expected_features = [
        "step", "amount", "balanceDiffOrig", "balanceDiffDest",
        "senderTxnCount", "receiverTxnCount", "destIsMerchant",
        "type_CASH_IN", "type_CASH_OUT", "type_DEBIT", "type_PAYMENT", "type_TRANSFER"
    ]
    for col in expected_features:
        if col not in df.columns:
            df[col] = 0

    return df[expected_features]

# ------------------ Sidebar ------------------
st.sidebar.title("Configuration")
input_mode = st.sidebar.radio("Select input mode:", ['Single Transaction', 'Batch Prediction'])

# ------------------ Single Transaction Mode ------------------
if input_mode == 'Single Transaction':
    st.subheader("Analyze a Single Transaction")

    col1, col2 = st.columns(2)
    with col1:
        step = st.number_input("Step (Time Unit)", min_value=1, max_value=744, value=1)
        transaction_type = st.selectbox("Transaction Type", ['TRANSFER', 'CASH_OUT', 'PAYMENT', 'CASH_IN', 'DEBIT'])
        amount = st.number_input("Amount (USD)", min_value=0.01, value=5000.00, step=50.0)
        nameOrig = st.text_input("Sender ID", value="C123456789")
        nameDest = st.text_input("Recipient ID", value="M987654321")

    with col2:
        oldbalanceOrg = st.number_input("Sender Balance (Before)", min_value=0.00, value=10000.00, step=100.0)
        newbalanceOrig = st.number_input("Sender Balance (After)", min_value=0.00, value=5000.00, step=100.0)
        oldbalanceDest = st.number_input("Recipient Balance (Before)", min_value=0.00, value=500.00, step=100.0)
        newbalanceDest = st.number_input("Recipient Balance (After)", min_value=0.00, value=5500.00, step=100.0)

    if st.button("🔍 Evaluate Transaction"):
        if not MODEL_LOADED:
            st.warning("⚠️ Model is not available. Using dummy prediction.")
        try:
            input_df = pd.DataFrame([{
                'step': step,
                'type': transaction_type,
                'amount': amount,
                'nameOrig': nameOrig,
                'oldbalanceOrg': oldbalanceOrg,
                'newbalanceOrig': newbalanceOrig,
                'nameDest': nameDest,
                'oldbalanceDest': oldbalanceDest,
                'newbalanceDest': newbalanceDest
            }])
            processed = preprocess_input(input_df)
            prob = best_model.predict_proba(processed)[0, 1]
            if prob >= 0.5:
                st.error(f"🔴 High Fraud Risk Detected ({prob:.2%})")
            else:
                st.success(f"🟢 Transaction Likely Legitimate ({(1 - prob):.2%} confidence)")
        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")

# ------------------ Batch Prediction Mode ------------------
elif input_mode == 'Batch Prediction':
    st.subheader("Evaluate Multiple Transactions")
    uploaded_file = st.file_uploader("Upload a CSV file", type=['csv'])

    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.write("### Preview of Uploaded Data", df.head())

            df_clean = df.drop(columns=['isFraud'], errors='ignore')

            if st.button("Run Batch Prediction"):
                if not MODEL_LOADED:
                    st.warning("⚠️ Model is not available. Dummy results will be shown.")
                processed = preprocess_input(df_clean)
                probs = best_model.predict_proba(processed)[:, 1]
                preds = (probs >= 0.5).astype(int)

                df_clean.insert(0, 'Prediction', preds)
                df_clean.drop(columns=[
                    'balanceDiffOrig', 'balanceDiffDest',
                    'destIsMerchant', 'senderTxnCount', 'receiverTxnCount'
                ], inplace=True, errors='ignore')

                st.success("Batch Prediction Complete")
                st.dataframe(df_clean.head(20))

                fraud_count = np.sum(preds)
                legit_count = len(preds) - fraud_count

                st.info(f"🔴 Fraudulent: {fraud_count}\n\n🟢 Legitimate: {legit_count}")
        except Exception as e:
            st.error(f"❌ Error reading or processing CSV: {e}")

# ------------------ Footer ------------------
st.markdown("---")
st.caption("📌 Disclaimer: This tool is for educational and risk assessment purposes only. Not a replacement for certified financial fraud systems.")
