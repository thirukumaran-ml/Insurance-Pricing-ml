
# ==========================================
# HEALTH INSURANCE PREMIUM PREDICTOR
# PRODUCTION STREAMLIT APPLICATION
# ==========================================

# -----------------------------
# IMPORT LIBRARIES
# -----------------------------

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -----------------------------
# PAGE CONFIGURATION
# -----------------------------

st.set_page_config(

    page_title="Insurance Premium Predictor",

    page_icon="💡",

    layout="centered"

)

# -----------------------------
# LOAD TRAINED MODEL
# -----------------------------

model = joblib.load(
    'best_insurance_model.pkl'
)

# -----------------------------
# APPLICATION TITLE
# -----------------------------

st.title(
    "🏥 Health Insurance Premium Predictor"
)

st.markdown("""

This AI-powered application predicts
medical insurance premiums using
Machine Learning.

### Features:
- Real-time premium prediction
- Risk assessment
- Advanced ML model
- Production-grade pipeline

""")

# -----------------------------
# SIDEBAR
# -----------------------------

st.sidebar.header("About")

st.sidebar.info("""

This project uses:

- LightGBM / Ensemble ML
- Feature Engineering
- Optuna Optimization
- SHAP Explainability
- Streamlit Deployment

""")

# -----------------------------
# USER INPUT SECTION
# -----------------------------

st.subheader("Enter Customer Information")

# Age
age = st.slider(

    "Age",

    min_value=18,

    max_value=64,

    value=30

)

# Gender
sex = st.selectbox(

    "Gender",

    ['male', 'female']

)

# BMI
bmi = st.slider(

    "BMI",

    min_value=15.0,

    max_value=50.0,

    value=25.0

)

# Children
children = st.slider(

    "Number of Children",

    min_value=0,

    max_value=5,

    value=0

)

# Smoking Status
smoker = st.selectbox(

    "Smoking Status",

    ['yes', 'no']

)

# Region
region = st.selectbox(

    "Region",

    [

        'southwest',

        'southeast',

        'northwest',

        'northeast'

    ]

)

# -----------------------------
# FEATURE ENGINEERING
# -----------------------------

def bmi_category(bmi):

    if bmi < 18.5:

        return 'Underweight'

    elif bmi < 25:

        return 'Normal'

    elif bmi < 30:

        return 'Overweight'

    else:

        return 'Obese'


def age_group(age):

    if age < 30:

        return 'Young'

    elif age < 50:

        return 'Adult'

    else:

        return 'Senior'

# -----------------------------
# CREATE INPUT DATAFRAME
# -----------------------------

input_df = pd.DataFrame({

    'age': [age],

    'sex': [sex],

    'bmi': [bmi],

    'children': [children],

    'smoker': [smoker],

    'region': [region],

    'bmi_category': [

        bmi_category(bmi)

    ],

    'age_group': [

        age_group(age)

    ],

    'bmi_smoker_interaction': [

        bmi * (
            1 if smoker == 'yes'
            else 0
        )

    ],

    'is_obese': [

        1 if bmi >= 30
        else 0

    ],

    'high_risk': [

        1 if (
            smoker == 'yes'
            and bmi >= 30
        )
        else 0

    ]

})

# -----------------------------
# DISPLAY INPUT SUMMARY
# -----------------------------

st.subheader("Customer Summary")

st.dataframe(input_df)

# -----------------------------
# PREDICTION BUTTON
# -----------------------------

if st.button("Predict Insurance Premium"):

    # Prediction
    prediction_log = model.predict(
        input_df
    )

    # Reverse log transformation
    prediction = np.expm1(
        prediction_log
    )[0]

    # -------------------------
    # DISPLAY PREDICTION
    # -------------------------

    st.success(

        f"Estimated Insurance Premium: "
        f"${prediction:,.2f}"

    )

    # -------------------------
    # RISK LEVEL
    # -------------------------

    st.subheader("Risk Assessment")

    if prediction < 10000:

        st.success(
            "🟢 Low Insurance Risk"
        )

    elif prediction < 30000:

        st.warning(
            "🟠 Moderate Insurance Risk"
        )

    else:

        st.error(
            "🔴 High Insurance Risk"
        )

    # -------------------------
    # ADDITIONAL INSIGHTS
    # -------------------------

    st.subheader("Prediction Insights")

    if smoker == 'yes':

        st.write(
            "- Smoking significantly "
            "increases insurance costs."
        )

    if bmi >= 30:

        st.write(
            "- High BMI contributes "
            "to increased medical risk."
        )

    if age >= 50:

        st.write(
            "- Higher age generally "
            "increases premium values."
        )

# -----------------------------
# FOOTER
# -----------------------------

st.markdown("---")

st.markdown(

    """
    Developed using:
    Machine Learning • Streamlit • SHAP • LightGBM
    """

)
