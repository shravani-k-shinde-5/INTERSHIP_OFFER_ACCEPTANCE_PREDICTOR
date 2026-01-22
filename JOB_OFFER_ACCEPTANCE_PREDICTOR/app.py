import streamlit as st
import pickle
import pandas as pd

# =====================================================
# 1. PAGE CONFIGURATION
# =====================================================
st.set_page_config(
    page_title="Internship Acceptance Predictor",
    page_icon="🎓",
    layout="centered"
)

# =====================================================
# 2. CUSTOM CSS (UI STRUCTURE & STYLING)
# =====================================================
st.markdown("""
<style>
    body {
        background-color: #f5f7fa;
    }

    .main-title {
        text-align: center;
        color: #4B0082;
        font-size: 38px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .sub-title {
        text-align: center;
        color: #666;
        font-size: 18px;
        margin-bottom: 30px;
    }

    .section-title {
        font-size: 20px;
        font-weight: 600;
        color: #4B0082;
        margin-bottom: 15px;
        border-bottom: 2px solid #eee;
        padding-bottom: 5px;
    }

    .card {
        background-color: #ffffff;
        padding: 25px;
        border-radius: 14px;
        box-shadow: 0px 6px 16px rgba(0,0,0,0.08);
        margin-bottom: 25px;
    }

    .result-card {
        background-color: #ffffff;
        padding: 22px;
        border-radius: 14px;
        box-shadow: 0px 6px 16px rgba(0,0,0,0.08);
        margin-top: 25px;
    }

    .predict-btn {
        display: flex;
        justify-content: center;
        margin-top: 10px;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# =====================================================
# 3. LOAD MODEL
# =====================================================
@st.cache_resource
def load_model():
    with open("offer.pkl", "rb") as file:
        return pickle.load(file)

model = load_model()

# =====================================================
# 4. HEADER
# =====================================================
st.markdown('<div class="main-title">🎓 Internship Offer Acceptance Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Predict whether a student will accept an internship offer</div>', unsafe_allow_html=True)

# =====================================================
# 5. INPUT SECTION
# =====================================================
with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.markdown('<div class="section-title">📌 Internship Details</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        stipend = st.slider("💰 Monthly Stipend (INR)", 0, 40000, 15000, step=1000)
        duration = st.selectbox("📅 Duration (months)", [1, 2, 3, 4, 5, 6])

    with col2:
        distance = st.slider("📍 Distance from Home (km)", 0, 100, 10)

    st.markdown('<div class="section-title">🏠 Work Preferences</div>', unsafe_allow_html=True)
    col3, col4 = st.columns(2)

    with col3:
        wfh = st.radio("Work From Home?", ["Yes", "No"])

    with col4:
        exam_time = st.radio("During Exam Time?", ["Yes", "No"])

    st.markdown('</div>', unsafe_allow_html=True)

# =====================================================
# 6. INPUT PREPROCESSING
# =====================================================
def preprocess_inputs(stipend, duration, wfh, distance, exam_time):
    return pd.DataFrame({
        "stipend": [stipend],
        "duration": [duration],
        "wfh": [1 if wfh == "Yes" else 0],
        "distance": [distance],
        "exam_time": [1 if exam_time == "Yes" else 0]
    })

input_df = preprocess_inputs(stipend, duration, wfh, distance, exam_time)

# =====================================================
# 7. PREDICT BUTTON
# =====================================================
st.markdown('<div class="predict-btn">', unsafe_allow_html=True)
predict = st.button("🔮 Predict Acceptance")
st.markdown('</div>', unsafe_allow_html=True)

# =====================================================
# =====================================================
# 8. PREDICTION OUTPUT (PROBABILITY-BASED DECISION)
# =====================================================
if predict:
    probability = model.predict_proba(input_df)[0][1]  # P(accepted)
    confidence = probability * 100

    # Custom decision threshold
    THRESHOLD = 0.64

    st.markdown('<div class="result-card">', unsafe_allow_html=True)

    if probability >= THRESHOLD:
        st.success(
            f"✅ **Offer will likely be ACCEPTED**\n\n"
            f"📊 Acceptance Probability: **{confidence:.1f}%**"
        )
    else:
        st.error(
            f"❌ **Offer will likely be REJECTED**\n\n"
            f"📊 Acceptance Probability: **{confidence:.1f}%**"
        )

    st.markdown("#### 🔍 Input Summary")
    st.write(f"""
    - 💰 Stipend: ₹{stipend}
    - 📅 Duration: {duration} months
    - 🏠 Work From Home: {wfh}
    - 📍 Distance: {distance} km
    - 📝 During Exam Time: {exam_time}
    """)

    st.markdown('</div>', unsafe_allow_html=True)

# =====================================================
# 9. FOOTER
# =====================================================
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(
    "<center><small>Built with Machine Learning & Streamlit</small></center>",
    unsafe_allow_html=True
)
