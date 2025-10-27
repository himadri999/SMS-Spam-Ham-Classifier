import streamlit as st
import joblib

# Load model
model = joblib.load("sms_spam_model.joblib")

# Streamlit UI
st.title("📩 SMS Spam/Ham Classifier")
st.write("Enter a message and find out if it's Spam or Ham!")

msg = st.text_area("Enter your message:")

if st.button("Classify"):
    pred = model.predict([msg])[0]
    if pred == 'spam':
        st.error("🚨 Spam Message Detected!")
    else:
        st.success("✅ Ham (Not Spam) Message!")

st.markdown("---")
st.caption("Built using Streamlit + Scikit-learn")
