import streamlit as st
import requests

st.set_page_config(page_title="Titanic Survival Predictor", layout="centered")

st.title("🚢 Titanic Survival Prediction")
st.markdown("Enter passenger details to predict survival probability.")

# Input fields
pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.selectbox("Sex", ["male", "female"])
age = st.number_input("Age", min_value=0, max_value=100, value=25)
sibsp = st.number_input("Siblings/Spouses Aboard", min_value=0, value=0)
parch = st.number_input("Parents/Children Aboard", min_value=0, value=0)
fare = st.number_input("Fare Paid", min_value=0.0, value=10.0)

if st.button("Predict"):
    payload = {
        "Pclass": pclass,
        "Sex": 0 if sex == "male" else 1,
        "Age": age,
        "SibSp": sibsp,
        "Parch": parch,
        "Fare": fare,
    }

    try:
        response = requests.post("http://backend:8000/predict", json=payload)

        result = response.json()

        if result["survived"] == 1:
            st.success(f"🎉 Survived (Probability: {result['probability']})")
        else:
            st.error(f"💀 Did Not Survive (Probability: {result['probability']})")

    except Exception as e:
        st.error("Backend not reachable. Make sure FastAPI is running.")
