import streamlit as st
import pandas as pd
import joblib

# -------------------------------
# Load models
# -------------------------------
heart_model = joblib.load("models/heart_pipeline.pkl")
kidney_model = joblib.load("models/kidney_pipeline.pkl")
liver_model = joblib.load("models/liver_pipeline.pkl")

# -------------------------------
# AI Precautions
# -------------------------------
def get_precautions(disease):
    data = {
        "Heart Disease": [
            "Reduce salt, sugar, and fatty foods.",
            "Exercise 30 min daily.",
            "Avoid smoking and alcohol.",
            "Monitor blood pressure regularly.",
            "Maintain healthy weight."
        ],
        "Kidney Disease": [
            "Drink 3-4 liters of water daily.",
            "Avoid painkillers and excess salt.",
            "Control diabetes and blood pressure.",
            "Limit protein intake.",
            "Avoid processed foods."
        ],
        "Liver Disease": [
            "Avoid alcohol completely.",
            "Reduce oily and spicy foods.",
            "Eat more fruits and vegetables.",
            "Maintain healthy weight.",
            "Take medications only as prescribed."
        ],
        "Healthy": [
            "Maintain balanced diet.",
            "Exercise regularly.",
            "Sleep 7-8 hours daily.",
            "Drink enough water.",
            "Avoid junk food."
        ]
    }
    return data.get(disease, ["No precautions available."])

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="AI Disease Prediction", layout="wide")
st.title("  Multi-Disease Prediction System")
st.write("Predict Heart, Kidney, and Liver Diseases with AI-generated precautions")

tabs = st.tabs([" Heart", " Kidney", " Liver"])

# ---------------- Heart Prediction ----------------
with tabs[0]:
    st.subheader("Heart Disease Prediction")
    # Input features
    heart_inputs = {
        "age": st.number_input("Age", 1, 120, key="heart_age"),
        "sex": st.selectbox("Sex", ["Male","Female"], key="heart_sex"),
        "cp": st.selectbox("Chest Pain Type (0-3)", [0,1,2,3], key="heart_cp"),
        "trestbps": st.number_input("Resting BP", key="heart_trestbps"),
        "chol": st.number_input("Cholesterol", key="heart_chol"),
        "fbs": st.selectbox("Fasting Blood Sugar >120 mg/dl", ["Yes","No"], key="heart_fbs"),
        "restecg": st.selectbox("Resting ECG (0-2)", [0,1,2], key="heart_restecg"),
        "thalach": st.number_input("Max Heart Rate", key="heart_thalach"),
        "exang": st.selectbox("Exercise Induced Angina", ["Yes","No"], key="heart_exang"),
        "oldpeak": st.number_input("ST Depression", key="heart_oldpeak"),
        "slope": st.selectbox("Slope of ST Segment (0-2)", [0,1,2], key="heart_slope"),
        "ca": st.number_input("Major Vessels Colored (0-3)", key="heart_ca"),
        "thal": st.selectbox("Thalassemia (1=normal,2=fixed,3=reversable)", [1,2,3], key="heart_thal")
    }

    if st.button("Predict Heart Disease", key="heart_predict"):
        # Convert categorical to numeric for pipeline
        heart_inputs['sex'] = 1 if heart_inputs['sex']=="Male" else 0
        heart_inputs['fbs'] = 1 if heart_inputs['fbs']=="Yes" else 0
        heart_inputs['exang'] = 1 if heart_inputs['exang']=="Yes" else 0

        df_heart = pd.DataFrame([heart_inputs])
        pred = heart_model.predict(df_heart)[0]
        disease = "Heart Disease" if pred==1 else "Healthy"
        st.success(f"Prediction: **{disease}**")
        st.info("### Precautions:")
        for p in get_precautions(disease):
            st.write("- ", p)

# ---------------- Kidney Prediction ----------------
with tabs[1]:
    st.subheader("Kidney Disease Prediction")
    kidney_inputs = {
        "age": st.number_input("Age", 1, 120, key="kidney_age"),
        "bp": st.number_input("Blood Pressure", key="kidney_bp"),
        "sg": st.number_input("Specific Gravity", key="kidney_sg"),
        "al": st.number_input("Albumin", key="kidney_al"),
        "su": st.number_input("Sugar", key="kidney_su"),
        "rbc": st.selectbox("Red Blood Cells", ["normal","abnormal"], key="kidney_rbc"),
        "pc": st.selectbox("Pus Cell", ["normal","abnormal"], key="kidney_pc"),
        "pcc": st.selectbox("Pus Cell Clumps", ["present","notpresent"], key="kidney_pcc"),
        "ba": st.selectbox("Bacteria", ["present","notpresent"], key="kidney_ba"),
        "bgr": st.number_input("Blood Glucose Random", key="kidney_bgr"),
        "bu": st.number_input("Blood Urea", key="kidney_bu"),
        "sc": st.number_input("Serum Creatinine", key="kidney_sc"),
        "sod": st.number_input("Sodium", key="kidney_sod"),
        "pot": st.number_input("Potassium", key="kidney_pot"),
        "hemo": st.number_input("Hemoglobin", key="kidney_hemo")
    }

    if st.button("Predict Kidney Disease", key="kidney_predict"):
        # Convert categorical to numeric
        kidney_inputs['rbc'] = 0 if kidney_inputs['rbc']=="normal" else 1
        kidney_inputs['pc'] = 0 if kidney_inputs['pc']=="normal" else 1
        kidney_inputs['pcc'] = 1 if kidney_inputs['pcc']=="present" else 0
        kidney_inputs['ba'] = 1 if kidney_inputs['ba']=="present" else 0

        df_kidney = pd.DataFrame([kidney_inputs])
        pred = kidney_model.predict(df_kidney)[0]
        disease = "Kidney Disease" if pred==1 else "Healthy"
        st.success(f"Prediction: **{disease}**")
        st.info("### Precautions:")
        for p in get_precautions(disease):
            st.write("- ", p)

# ---------------- Liver Prediction ----------------
with tabs[2]:
    st.subheader("Liver Disease Prediction")
    liver_inputs = {
        "age": st.number_input("Age", 1, 120, key="liver_age"),
        "gender": st.selectbox("Gender", ["Male","Female"], key="liver_gender"),
        "tb": st.number_input("Total Bilirubin", key="liver_tb"),
        "db": st.number_input("Direct Bilirubin", key="liver_db"),
        "alkphos": st.number_input("Alkaline Phosphotase", key="liver_alkphos"),
        "sgot": st.number_input("SGOT", key="liver_sgot"),
        "sgpt": st.number_input("SGPT", key="liver_sgpt"),
        "tp": st.number_input("Total Proteins", key="liver_tp"),
        "alb": st.number_input("Albumin", key="liver_alb"),
        "ag_ratio": st.number_input("Albumin/Globulin Ratio", key="liver_ag_ratio")
    }

    if st.button("Predict Liver Disease", key="liver_predict"):
        liver_inputs['gender'] = 1 if liver_inputs['gender']=="Male" else 0
        df_liver = pd.DataFrame([liver_inputs])
        pred = liver_model.predict(df_liver)[0]
        disease = "Liver Disease" if pred==1 else "Healthy"
        st.success(f"Prediction: **{disease}**")
        st.info("### Precautions:")
        for p in get_precautions(disease):
            st.write("- ", p)
