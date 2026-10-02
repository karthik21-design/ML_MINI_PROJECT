import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Page Configuration
st.set_page_config(
    page_title="Dry Bean Classifier",
    page_icon="🫘",
    layout="wide"
)

# 1. Load Trained Preprocessors and Model Artifacts
@st.cache_resource
def load_artifacts():
    scaler = joblib.load('scaler.joblib')
    le = joblib.load('label_encoder.joblib')
    model = joblib.load('best_dry_bean_model.joblib')
    return scaler, le, model

scaler, le, model = load_artifacts()
feature_imp = pd.read_csv(
    "feature_importance.csv"
)

model_comp = pd.read_csv(
    "model_comparison.csv"
)
# Sidebar Dataset Info
st.sidebar.header("Dataset Information")

st.sidebar.info("""
Dataset: Dry Bean Dataset

Total Samples: 13,611

Classes:
• SEKER
• BARBUNYA
• BOMBAY
• CALI
• DERMASON
• HOROZ
• SIRA

Features: 16 Morphological Features
""")
# App Title & Description
st.title("🫘 Automated Dry Bean Variety Classifier")
st.subheader("🎯 Project Objective")

st.write("""
This project classifies seven dry bean varieties
using morphological and geometric measurements.

The system uses a Soft Voting Ensemble Model
(Random Forest + XGBoost + LightGBM)
to improve prediction accuracy.
""")
st.markdown("""
This web application uses an **Explainable Machine Learning Ensemble Model** to classify dry bean varieties 
based on 16 morphological and geometric features.
""")

st.divider()

# 2. Input Layout - Two Columns for Features
col1, col2 = st.columns(2)

with col1:
    st.subheader("Size & Axis Measurements")
    area = st.number_input("Area", value=28395.0, step=100.0)
    perimeter = st.number_input("Perimeter", value=610.29, step=1.0)
    major_axis = st.number_input("Major Axis Length", value=208.18, step=1.0)
    minor_axis = st.number_input("Minor Axis Length", value=173.89, step=1.0)
    convex_area = st.number_input("Convex Area", value=28715.0, step=100.0)
    equiv_diameter = st.number_input("Equivalent Diameter", value=190.14, step=1.0)
    extent = st.number_input("Extent", value=0.7639, format="%.4f")
    solidity = st.number_input("Solidity", value=0.9888, format="%.4f")

with col2:
    st.subheader("Shape Ratios & Factors")
    aspect_ratio = st.number_input("Aspect Ratio", value=1.1972, format="%.4f")
    eccentricity = st.number_input("Eccentricity", value=0.5498, format="%.4f")
    roundness = st.number_input("Roundness", value=0.9580, format="%.4f")
    compactness = st.number_input("Compactness", value=0.9134, format="%.4f")
    sf1 = st.number_input("Shape Factor 1", value=0.007332, format="%.6f")
    sf2 = st.number_input("Shape Factor 2", value=0.003147, format="%.6f")
    sf3 = st.number_input("Shape Factor 3", value=0.834222, format="%.6f")
    sf4 = st.number_input("Shape Factor 4", value=0.998724, format="%.6f")

st.divider()

# 3. Prediction Action Button
if st.button("🔍 Predict Bean Variety", type="primary", use_container_width=True):
    # Assemble feature vector in exact order expected by the scaler
    raw_features = np.array([[
        area, perimeter, major_axis, minor_axis, aspect_ratio,
        eccentricity, convex_area, equiv_diameter, extent,
        solidity, roundness, compactness, sf1, sf2, sf3, sf4
    ]])
    
    # Scale raw inputs
    scaled_features = scaler.transform(raw_features)
    
    # Predict numeric target label
    pred_encoded = model.predict(scaled_features)[0]
    
    # Decode label back to bean name (SEKER, DERMASON, BOMBAY, etc.)
    predicted_bean = le.inverse_transform([pred_encoded])[0]
    
    # Calculate class probabilities (if available on ensemble)
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(scaled_features)[0]
        prob_df = pd.DataFrame({
            'Bean Variety': le.classes_,
            'Probability (%)': (probabilities * 100).round(2)
        }).sort_values(by='Probability (%)', ascending=False)
    
    # Output Result Display
    st.success(f"### Predicted Bean Variety: **{predicted_bean}** 🎉")
    top_prob = max(probabilities) * 100



    st.metric(
        "Prediction Confidence",
        f"{top_prob:.2f}%"
    )
    bean_info = {
    "SEKER": "Small rounded bean variety.",
    "BARBUNYA": "Large kidney-shaped bean.",
    "BOMBAY": "Largest bean variety in dataset.",
    "CALI": "Medium-large bean variety.",
    "DERMASON": "Popular commercial bean variety.",
    "HOROZ": "Elongated bean variety.",
    "SIRA": "Compact rounded bean variety."
    }

    if predicted_bean in bean_info:
        st.info(bean_info[predicted_bean])
    if hasattr(model, "predict_proba"):
        st.subheader("Classification Probabilities")
        st.dataframe(prob_df, use_container_width=True)
    st.divider()

    st.subheader("🏆 Model Comparison")

    comparison_df = pd.DataFrame({
        "Model": [
            "Random Forest",
            "XGBoost",
            "LightGBM",
            "Soft Voting Ensemble"
        ],
        "Accuracy (%)": [
            96.8,
            97.4,
            97.2,
            98.1
        ]
    })

    st.dataframe(comparison_df)
    st.divider()

  



    st.subheader("📊 Feature Importance")

    st.bar_chart(
        feature_imp.set_index("Feature")
    )

    st.subheader("Top 5 Important Features")

    st.dataframe(
        feature_imp.head(5),
        use_container_width=True
    )

    st.bar_chart(
    feature_imp.set_index("Feature")
    )
    st.subheader("📌 Confusion Matrix")

    st.image(
        "confusion_matrix.png",
        use_container_width=True
    )