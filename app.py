
from pathlib import Path
import math
import joblib
import numpy as np
import streamlit as st

st.set_page_config(
    page_title="Diabetes Risk Screening",
    page_icon="🩺",
    layout="centered"
)

MODEL_PATH = Path("model/diabetes_model.pkl")

# -----------------------------
# Theme / Styling
# -----------------------------
st.markdown("""
<style>
.stApp{
    background:
        radial-gradient(circle at 12% 0%, rgba(14,165,233,.14), transparent 28%),
        radial-gradient(circle at 100% 10%, rgba(16,185,129,.10), transparent 24%),
        #07111f;
    color:#ffffff !important;
}

[data-testid="stHeader"]{
    background:transparent;
}

.block-container{
    max-width:980px;
    padding-top:2rem;
    padding-bottom:3rem;
}

.hero{
    border:1px solid #1f3550;
    background:linear-gradient(180deg, rgba(15,23,42,.96), rgba(8,17,31,.96));
    border-radius:24px;
    padding:28px;
    margin-bottom:18px;
    box-shadow:0 18px 55px rgba(0,0,0,.25);
}

.badges{
    display:flex;
    gap:10px;
    flex-wrap:wrap;
    margin-bottom:16px;
}

.badge{
    background:#10243a;
    border:1px solid #24496f;
    color:#22d3ee;
    padding:7px 12px;
    border-radius:9px;
    font-weight:800;
    font-size:.76rem;
    letter-spacing:.04em;
}

.hero h1{
    color:#ffffff !important;
    margin:0 0 8px;
    font-size:2.2rem;
}

.hero p{
    color:#cbd5e1 !important;
    line-height:1.65;
    margin:0;
    font-size:1.02rem;
}

.section-title{
    color:#ffffff !important;
    font-size:1.2rem;
    font-weight:800;
    margin:14px 0 8px;
}

.info-card{
    background:#0d1728;
    border:1px solid #223652;
    border-radius:18px;
    padding:18px;
    margin:10px 0;
}

.metric-label{
    color:#a8bdd6 !important;
    font-size:.82rem;
}

.metric-value{
    color:#ffffff !important;
    font-size:1.6rem;
    font-weight:800;
}

.safe-note{
    background:#0f1f34;
    border:1px solid #2c4a6f;
    color:#dbeafe !important;
    border-radius:14px;
    padding:14px 16px;
    margin:12px 0 18px;
}

[data-testid="stMarkdownContainer"] *{
    color:#ffffff !important;
}

label, p, span, div{
    color:#ffffff;
}

[data-testid="stNumberInput"] input,
[data-testid="stTextInput"] input{
    background:#0b1626 !important;
    color:#ffffff !important;
    border:1px solid #34506f !important;
}

.stButton>button{
    width:100%;
    border:0;
    border-radius:12px;
    font-weight:800;
    background:linear-gradient(135deg,#2563eb,#06b6d4);
    color:white !important;
    padding:.7rem 1rem;
}

[data-testid="stAlert"]{
    border-radius:14px;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Header
# -----------------------------
st.markdown("""
<div class="hero">
  <div class="badges">
    <span class="badge">MACHINE LEARNING</span>
    <span class="badge">STREAMLIT</span>
    <span class="badge">PYTHON</span>
  </div>
  <h1>Diabetes Risk Screening App</h1>
  <p>
    A portfolio-ready machine learning interface for diabetes risk screening.
    It is designed for demonstration and educational use, with high-contrast text and a clean clinical-style theme.
  </p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="safe-note">
<b>Important:</b> This app is not a medical diagnosis tool and should not be used to make treatment decisions.
A trained model must be clinically validated before real-world use.
</div>
""", unsafe_allow_html=True)

# -----------------------------
# Model loader
# -----------------------------
@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    try:
        return joblib.load(MODEL_PATH)
    except Exception:
        return None

model = load_model()

if model is None:
    st.info("Demo mode is active. Add `model/diabetes_model.pkl` to use a trained machine-learning model.")
else:
    st.success("Trained model loaded successfully.")

# -----------------------------
# Inputs
# -----------------------------
st.markdown('<div class="section-title">Patient Inputs</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1, step=1)
    glucose = st.number_input("Glucose", min_value=0, max_value=300, value=110, step=1)
    blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=200, value=72, step=1)
    skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20, step=1)

with col2:
    insulin = st.number_input("Insulin", min_value=0, max_value=1000, value=80, step=1)
    bmi = st.number_input("BMI", min_value=0.0, max_value=80.0, value=25.0, step=0.1)
    pedigree = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.47, step=0.01)
    age = st.number_input("Age", min_value=1, max_value=120, value=30, step=1)

features = np.array([[
    pregnancies, glucose, blood_pressure, skin_thickness,
    insulin, bmi, pedigree, age
]], dtype=float)

# -----------------------------
# Demo fallback
# -----------------------------
def demo_probability(x):
    """
    Illustrative, non-clinical deterministic demo only.
    This is intentionally labeled as demo output and is not a medical score.
    """
    vals = x.flatten().astype(float)
    scaled = np.array([
        vals[0] / 20.0,
        vals[1] / 300.0,
        vals[2] / 200.0,
        vals[3] / 100.0,
        vals[4] / 1000.0,
        vals[5] / 80.0,
        vals[6] / 3.0,
        vals[7] / 120.0,
    ])
    weighted = (
        scaled[1]*0.28 +
        scaled[5]*0.18 +
        scaled[7]*0.16 +
        scaled[0]*0.10 +
        scaled[4]*0.08 +
        scaled[6]*0.08 +
        scaled[2]*0.06 +
        scaled[3]*0.06
    )
    return float(max(0.02, min(0.98, weighted)))

# -----------------------------
# Prediction
# -----------------------------
if st.button("Run Screening"):
    if model is not None and hasattr(model, "predict_proba"):
        prob = float(model.predict_proba(features)[0][1])
        source = "Trained model"
    else:
        prob = demo_probability(features)
        source = "Illustrative demo"

    colA, colB = st.columns(2)
    with colA:
        st.markdown(
            f'<div class="info-card"><div class="metric-label">Estimated Risk</div>'
            f'<div class="metric-value">{prob*100:.1f}%</div></div>',
            unsafe_allow_html=True
        )
    with colB:
        label = "Higher" if prob >= 0.5 else "Lower"
        st.markdown(
            f'<div class="info-card"><div class="metric-label">Screening Category</div>'
            f'<div class="metric-value">{label}</div></div>',
            unsafe_allow_html=True
        )

    st.progress(prob)
    st.caption(f"Prediction source: {source}")

    if model is None:
        st.warning(
            "This is a visual demo result only. It is not generated by a clinically validated model."
        )
    else:
        st.info(
            "Even with a trained model, this result is only a screening estimate and not a diagnosis."
        )

with st.expander("How to use a real ML model"):
    st.write(
        "Train the included model using a suitable diabetes dataset, then save it as "
        "`model/diabetes_model.pkl`. Restart the Streamlit app after adding the model file."
    )

st.caption("Diabetes Risk Screening • Python • Streamlit • Machine Learning")
