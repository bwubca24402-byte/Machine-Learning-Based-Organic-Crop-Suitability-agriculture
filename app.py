import streamlit as st
from preprocessing import FEATURES
from model import train_model, predict_suitability
from utils import suitability_message


st.set_page_config(page_title="Organic Crop Suitability", page_icon="🌱", layout="wide")
st.title("🌱 Machine Learning-Based Organic Crop Suitability Assessment")
st.write("Enter soil nutrient values to get a model prediction. This is an educational project, not an official agricultural recommendation.")

@st.cache_resource
def get_trained_model():
    return train_model()

try:
    model, metrics = get_trained_model()
except Exception as err:
    st.error(f"Model or dataset loading failed: {err}")
    st.info("Keep app.py, model.py, preprocessing.py, utils.py, and the CSV dataset in the same folder.")
    st.stop()

with st.expander("Model evaluation", expanded=False):
    st.metric("Test accuracy", f"{metrics['accuracy']:.1%}")
    st.text(metrics["classification_report"])

st.subheader("Soil information")
col1, col2, col3 = st.columns(3)
with col1:
    ph = st.number_input("Soil pH", 0.0, 14.0, 6.5, 0.1)
    ec = st.number_input("EC", 0.0, 20.0, 0.5, 0.1, help="Electrical conductivity (dataset units)")
    organic_c = st.number_input("Organic Carbon", 0.0, 20.0, 1.0, 0.1)
    nitrogen = st.number_input("Nitrogen", 0.0, 5000.0, 250.0, 1.0)
with col2:
    phosphorus = st.number_input("Phosphorus", 0.0, 5000.0, 30.0, 1.0)
    potassium = st.number_input("Potassium", 0.0, 5000.0, 150.0, 1.0)
    sulphur = st.number_input("Sulphur", 0.0, 1000.0, 20.0, 1.0)
    zinc = st.number_input("Zinc", 0.0, 1000.0, 1.0, 0.1)
with col3:
    iron = st.number_input("Iron", 0.0, 1000.0, 5.0, 0.1)
    manganese = st.number_input("Manganese", 0.0, 1000.0, 2.0, 0.1)
    copper = st.number_input("Copper", 0.0, 1000.0, 0.5, 0.1)
    boron = st.number_input("Boron", 0.0, 1000.0, 0.5, 0.1)

if st.button("Assess Soil Suitability", type="primary"):
    sample = {"pH": ph, "EC": ec, "Organic_C": organic_c, "Nitrogen": nitrogen,
              "Phosphorus": phosphorus, "Potassium": potassium, "Sulphur": sulphur,
              "Zinc": zinc, "Iron": iron, "Manganese": manganese, "Copper": copper, "Boron": boron}
    label, confidence = predict_suitability(model, sample)
    st.success(f"Prediction: {label}")
    if confidence is not None:
        st.write(f"Model confidence (largest class probability): {confidence:.1%}")
    st.write(suitability_message(label))
    st.caption("The CSV labels are synthetic and rubric-derived for student learning; this output is not a certification.")
