import streamlit as st
import pickle
import numpy as np

def main():
    st.set_page_config(page_title="Churn Prediction", layout="centered")
    st.title("📊 Customer Churn Prediction App")
    st.markdown("Enter customer details below to predict churn.")

    # Load model and scaler
    try:
        with open("churn_model.pkl", "rb") as f:
            model = pickle.load(f)
        with open("scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        st.success("Model and scaler loaded successfully.")
    except Exception as e:
        st.error(f"❌ Error loading model or scaler: {e}")
        return

    # Input fields
    credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=650)
    geography = st.selectbox("Geography", ["France", "Spain", "Germany"])
    gender = st.radio("Gender", ["Male", "Female"])
    age = st.number_input("Age", min_value=18, max_value=100, value=35)
    tenure = st.number_input("Tenure (years)", min_value=0, max_value=10, value=3)
    balance = st.number_input("Balance", min_value=0.0, value=50000.0)
    num_of_products = st.selectbox("Number of Products", [1, 2, 3, 4])
    has_cr_card = st.radio("Has Credit Card?", ["Yes", "No"])
    is_active_member = st.radio("Is Active Member?", ["Yes", "No"])
    estimated_salary = st.number_input("Estimated Salary", min_value=0.0, value=60000.0)

    # Encode categorical variables
    geography_encoded = [1 if geography == "Germany" else 0,
                         1 if geography == "Spain" else 0]  # France is baseline
    gender_encoded = [1 if gender == "Male" else 0]

    # Combine all inputs
    input_data = np.array([[credit_score] + geography_encoded + gender_encoded +
                           [age, tenure, balance, num_of_products,
                            1 if has_cr_card == "Yes" else 0,
                            1 if is_active_member == "Yes" else 0,
                            estimated_salary]])

    # Predict button
    if st.button("🔍 Predict Churn"):
        try:
            input_scaled = scaler.transform(input_data)
            prediction = model.predict(input_scaled)

            if prediction[0] == 1:
                st.error("⚠️ This customer is likely to churn.")
            else:
                st.success("✅ This customer is likely to stay.")
        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")

if __name__ == "__main__":
    main()