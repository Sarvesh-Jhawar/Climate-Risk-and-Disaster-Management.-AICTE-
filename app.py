# app.py
import streamlit as st
import pandas as pd
import joblib

# --- Configuration ---
st.set_page_config(
    page_title="Disaster Prediction AI",
    page_icon="🌍",
    layout="wide"
)

# --- Load Model and Encoder ---
@st.cache_resource
def load_model_and_encoder():
    """Load the pre-trained model and label encoder."""
    try:
        model = joblib.load('random_forest_model.joblib')
        label_encoder = joblib.load('label_encoder.joblib')
        return model, label_encoder
    except FileNotFoundError:
        st.error("Model or Label Encoder not found. Please run `week3_project.py` to train and save the model first.")
        return None, None

model, label_encoder = load_model_and_encoder()

# --- Streamlit UI ---
st.title("🌍 Climate Risk & Disaster Prediction AI")
st.markdown(
    "This application predicts the type of a natural disaster based on its geographical coordinates and time. "
    "The prediction is made using a pre-trained **Random Forest** model."
)
st.markdown("---")

# --- Sidebar for User Input ---
st.sidebar.header("📍 Enter Disaster Event Details")

longitude = st.sidebar.number_input(
    "Longitude (-180 to 180)",
    min_value=-180.0,
    max_value=180.0,
    value=-118.5,
    step=0.1,
    format="%.4f"
)
latitude = st.sidebar.number_input(
    "Latitude (-90 to 90)",
    min_value=-90.0,
    max_value=90.0,
    value=34.5,
    step=0.1,
    format="%.4f"
)
month = st.sidebar.slider(
    "Month of the Year",
    min_value=1,
    max_value=12,
    value=7
)
dayofweek = st.sidebar.slider(
    "Day of the Week (0=Monday, 6=Sunday)",
    min_value=0,
    max_value=6,
    value=3
)

# --- Prediction Logic ---
if model is not None and label_encoder is not None:
    if st.sidebar.button("🔮 Predict Disaster Category", type="primary"):
        # Prepare input data
        input_data = pd.DataFrame({
            'Longitude': [longitude],
            'Latitude': [latitude],
            'Month': [month],
            'DayOfWeek': [dayofweek]
        })

        # Make prediction
        prediction_encoded = model.predict(input_data)[0]
        prediction_label = label_encoder.inverse_transform([prediction_encoded])[0]

        # Get prediction probabilities
        prediction_proba = model.predict_proba(input_data)
        
        # Display results
        st.subheader("✨ Prediction Results")
        col1, col2 = st.columns(2)
        
        with col1:
            st.success(f"**Predicted Disaster Category:**")
            st.metric(label="Category", value=prediction_label)
            
            st.write("### 🗺️ Event Location")
            map_df = pd.DataFrame({'lat': [latitude], 'lon': [longitude]})
            st.map(map_df, zoom=5)

        with col2:
            st.write("### 🎯 Prediction Confidence")
            proba_df = pd.DataFrame(
                prediction_proba.T,
                index=label_encoder.classes_,
                columns=['Probability']
            )
            st.bar_chart(proba_df)

st.markdown("---")
st.info(
    "**How it works:** The model was trained on a dataset of global natural disasters. "
    "It uses the location (Longitude, Latitude) and time (Month, Day of Week) to classify the event. "
    "The current model is a Random Forest Classifier."
)