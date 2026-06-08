import streamlit as st
import pandas as pd
import joblib

# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Heart Disease Prediction System",
    page_icon="❤️",
    layout="wide"
)

# =====================================================
# CUSTOM CSS
# =====================================================

st.markdown("""
<style>
.main {
    padding-top: 1rem;
}

.big-font {
    font-size:20px !important;
    font-weight:bold;
}

.result-box {
    padding:15px;
    border-radius:10px;
    margin-top:10px;
}
</style>
""", unsafe_allow_html=True)

# =====================================================
# LOAD MODEL & SCALER
# =====================================================

@st.cache_resource
def load_model():
    model = joblib.load("final_heart_model.pkl")
    scaler = joblib.load("final_scaler.pkl")
    return model, scaler

try:
    model, scaler = load_model()

except Exception as e:
    st.error("❌ Unable to load model files.")
    st.error(str(e))
    st.stop()

# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.title("❤️ Heart Disease Predictor")

st.sidebar.markdown("""
### About Project

This application predicts the likelihood of heart disease using a Machine Learning model trained on a heart disease dataset.

### Model Used

✅ Logistic Regression

### Model Accuracy

**80.33%**

### Dataset Features

- Age
- Sex
- Chest Pain Type
- Blood Pressure
- Cholesterol
- Blood Sugar
- ECG Results
- Heart Rate
- Angina
- Old Peak
- Slope
- Major Vessels
- Thalassemia

### Disclaimer

This tool is for educational purposes only.

It should NOT replace professional medical advice.
""")

# =====================================================
# TITLE
# =====================================================

st.title("❤️ Heart Disease Prediction System")

st.markdown("""
Predict the probability of heart disease using patient clinical information.

Fill all values and click **Predict**.
""")

# =====================================================
# FEATURE EXPLANATION
# =====================================================

with st.expander("📚 Understanding Each Feature"):

    st.markdown("""
### Age
Age of the patient in years.

### Sex
0 = Female  
1 = Male

### Chest Pain Type (cp)

- 0 = Typical Angina
- 1 = Atypical Angina
- 2 = Non-anginal Pain
- 3 = Asymptomatic

### Resting Blood Pressure (trestbps)

Blood pressure measured at rest (mm Hg).

### Cholesterol (chol)

Serum cholesterol in mg/dl.

### Fasting Blood Sugar (fbs)

- 0 = ≤120 mg/dl
- 1 = >120 mg/dl

### Rest ECG (restecg)

Electrocardiographic results.

### Maximum Heart Rate (thalach)

Highest heart rate achieved.

### Exercise Induced Angina (exang)

- 0 = No
- 1 = Yes

### Old Peak

ST depression induced by exercise.

### Slope

Slope of peak exercise ST segment.

### CA

Number of major vessels colored by fluoroscopy.

### Thal

Thalassemia category.
""")

# =====================================================
# INPUT FORM
# =====================================================

col1, col2 = st.columns(2)

with col1:

    age = st.number_input(
        "Age",
        min_value=1,
        max_value=120,
        value=50
    )

    sex = st.selectbox(
        "Sex",
        [0, 1],
        format_func=lambda x: "Female" if x == 0 else "Male"
    )

    cp = st.selectbox(
        "Chest Pain Type",
        [0, 1, 2, 3]
    )

    trestbps = st.number_input(
        "Resting Blood Pressure",
        min_value=50,
        max_value=250,
        value=120
    )

    chol = st.number_input(
        "Cholesterol",
        min_value=100,
        max_value=700,
        value=200
    )

    fbs = st.selectbox(
        "Fasting Blood Sugar > 120 mg/dl",
        [0, 1]
    )

    restecg = st.selectbox(
        "Rest ECG",
        [0, 1, 2]
    )

with col2:

    thalach = st.number_input(
        "Maximum Heart Rate",
        min_value=50,
        max_value=250,
        value=150
    )

    exang = st.selectbox(
        "Exercise Induced Angina",
        [0, 1]
    )

    oldpeak = st.number_input(
        "Old Peak",
        min_value=0.0,
        max_value=10.0,
        value=1.0,
        step=0.1
    )

    slope = st.selectbox(
        "Slope",
        [0, 1, 2]
    )

    ca = st.selectbox(
        "Number of Major Vessels",
        [0, 1, 2, 3, 4]
    )

    thal = st.selectbox(
        "Thal",
        [0, 1, 2, 3]
    )

# =====================================================
# PREDICTION BUTTON
# =====================================================

if st.button("🔍 Predict Heart Disease"):

    try:

        patient_data = pd.DataFrame({
            "age": [age],
            "sex": [sex],
            "cp": [cp],
            "trestbps": [trestbps],
            "chol": [chol],
            "fbs": [fbs],
            "restecg": [restecg],
            "thalach": [thalach],
            "exang": [exang],
            "oldpeak": [oldpeak],
            "slope": [slope],
            "ca": [ca],
            "thal": [thal]
        })

        scaled_data = scaler.transform(patient_data)

        prediction = model.predict(scaled_data)[0]

        probability = model.predict_proba(scaled_data)[0]

        risk_percentage = probability[1] * 100

        st.divider()

        st.subheader("📊 Prediction Results")

        st.metric(
            label="Heart Disease Probability",
            value=f"{risk_percentage:.2f}%"
        )

        st.progress(min(int(risk_percentage), 100))

        if risk_percentage < 30:
            risk_level = "🟢 Low Risk"

        elif risk_percentage < 70:
            risk_level = "🟡 Moderate Risk"

        else:
            risk_level = "🔴 High Risk"

        st.write(f"### Risk Category: {risk_level}")

        if prediction == 1:

            st.error(
                "Prediction: Heart Disease Likely Detected"
            )

        else:

            st.success(
                "Prediction: No Heart Disease Detected"
            )

        st.divider()

        st.subheader("📋 Patient Summary")

        st.dataframe(
            patient_data,
            use_container_width=True
        )

        st.divider()

        st.subheader("🩺 Interpretation")

        if risk_percentage < 30:

            st.info("""
            The model estimates a relatively low risk.
            Continue healthy lifestyle practices and regular checkups.
            """)

        elif risk_percentage < 70:

            st.warning("""
            Moderate risk detected.
            Consider discussing results with a healthcare professional.
            """)

        else:

            st.error("""
            High risk detected.
            Professional medical evaluation is strongly recommended.
            """)

    except Exception as e:

        st.error("Prediction failed.")
        st.error(str(e))

# =====================================================
# MODEL INFORMATION
# =====================================================

st.divider()

with st.expander("🤖 Model Information"):

    st.markdown("""
### Final Model

**Logistic Regression**

### Dataset

Heart Disease Dataset (Deduplicated)

### Features Used

1. age
2. sex
3. cp
4. trestbps
5. chol
6. fbs
7. restecg
8. thalach
9. exang
10. oldpeak
11. slope
12. ca
13. thal

### Performance

Accuracy: **80.33%**
""")

# =====================================================
# FOOTER
# =====================================================

st.divider()

st.caption(
    "Built with Streamlit • Scikit-Learn • Logistic Regression"
)
