import streamlit as st
from PIL import Image
import pandas as pd
from inference import load_model, analyze_mri, publish_mqtt_result

st.set_page_config(page_title="AI Medical Imaging Platform", layout="wide")

st.title("Brain Tumor Medical Imaging AI Engine")
st.write("Upload an MRI scan to run MobileNetV3 classification, generate Grad-CAM visual attention maps, and dispatch real-time diagnosis payloads over MQTT.")

# Load Model with caching
@st.cache_resource
def get_cached_model():
    return load_model()

model = get_cached_model()

# Sidebar Setup
st.sidebar.header("Configuration")
enable_mqtt = st.sidebar.checkbox("Publish results to MQTT Broker", value=True)
target_layer = st.sidebar.selectbox("Grad-CAM Target Layer", ["features.12", "features.11", "features.10"])

# File Upload Section
uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    raw_img = Image.open(uploaded_file).convert("RGB")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Input MRI Scan")
        st.image(raw_img, use_container_width=True)

    with st.spinner("Executing Inference & Building Visual Attention Map..."):
        # Run processing
        results = analyze_mri(raw_img, model, target_layer_name=target_layer)

    with col2:
        st.subheader("Grad-CAM Explainability Map")
        st.image(results["cam_overlay"], use_container_width=True)

    # Diagnostic Metrics Panel
    st.markdown("---")
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    m_col1.metric("Predicted Diagnosis", results["pred_class"])
    m_col2.metric("Confidence Score", f"{results['confidence']:.2f}%")
    m_col3.metric("Inference Latency", f"{results['inf_time_ms']:.2f} ms")
    m_col4.metric("Grad-CAM Compute Time", f"{results['cam_time_ms']:.2f} ms")

    # Class Probabilities Breakdown
    st.write("### Prediction Breakdown")
    prob_df = pd.DataFrame(
        results["probabilities"].items(), columns=["Class", "Probability"]
    )
    st.bar_chart(prob_df.set_index("Class"))

    # Optional MQTT Publishing
    if enable_mqtt:
        mqtt_status = publish_mqtt_result(results, image_name=uploaded_file.name)
        if mqtt_status:
            st.success("Successfully transmitted diagnosis telemetry to MQTT broker topic `brain_tumor/diagnosis`")
        else:
            st.error("Failed to transmit metrics to local MQTT broker.")