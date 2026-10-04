import streamlit as st
import pickle
import numpy as np

# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(
    page_title="Parkinson's AI Detection",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------------
with open("parkinsons_model.pkl", "rb") as file:
    model = pickle.load(file)

with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

# ---------------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------------
st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

* {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 10% 10%, rgba(91, 75, 255, 0.12), transparent 28%),
        radial-gradient(circle at 90% 20%, rgba(0, 190, 255, 0.10), transparent 28%),
        #080b14;
    color: #f5f7ff;
}

/* Hide default Streamlit elements */
#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header {
    background: transparent !important;
}

/* Main container */
.block-container {
    max-width: 1250px;
    padding-top: 2rem;
    padding-bottom: 4rem;
}

/* Hero */
.hero {
    padding: 42px 45px;
    border-radius: 28px;
    background:
        linear-gradient(135deg,
        rgba(35, 45, 100, 0.95),
        rgba(15, 23, 52, 0.92));
    border: 1px solid rgba(130, 150, 255, 0.20);
    box-shadow: 0 25px 70px rgba(0,0,0,0.35);
    margin-bottom: 30px;
}

.hero-content {
    display: flex;
    align-items: center;
    gap: 25px;
}

.brain {
    font-size: 65px;
    filter: drop-shadow(0 0 20px rgba(120,130,255,0.45));
}

.hero-title {
    font-size: 42px;
    font-weight: 800;
    margin: 0;
    background: linear-gradient(90deg, #ffffff, #9eb7ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-subtitle {
    color: #aeb8d4;
    font-size: 16px;
    margin-top: 8px;
}

.badge {
    display: inline-block;
    margin-top: 17px;
    padding: 7px 14px;
    border-radius: 30px;
    background: rgba(77, 126, 255, 0.15);
    border: 1px solid rgba(110, 145, 255, 0.25);
    color: #8fb2ff;
    font-size: 13px;
    font-weight: 600;
}

/* Section cards */
.section-card {
    background: rgba(17, 22, 38, 0.82);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 22px;
    padding: 25px;
    margin-bottom: 22px;
    box-shadow: 0 15px 45px rgba(0,0,0,0.18);
}

.section-title {
    font-size: 21px;
    font-weight: 700;
    color: #f2f5ff;
    margin-bottom: 5px;
}

.section-description {
    font-size: 13px;
    color: #8f9ab5;
    margin-bottom: 20px;
}

/* Patient card */
.patient-icon {
    font-size: 25px;
}

input {
    border-radius: 12px !important;
}

/* Input labels */
label {
    color: #bfc8df !important;
    font-weight: 500 !important;
}

/* Number inputs */
div[data-baseweb="input"] {
    background: #151a2a !important;
    border: 1px solid #252d45 !important;
    border-radius: 12px !important;
}

div[data-baseweb="input"]:focus-within {
    border-color: #647cff !important;
    box-shadow: 0 0 0 1px #647cff !important;
}

/* Button */
.stButton > button {
    width: 100%;
    height: 58px;
    border-radius: 16px;
    border: none;
    background: linear-gradient(90deg, #5865f2, #7b61ff);
    color: white;
    font-size: 17px;
    font-weight: 700;
    letter-spacing: 0.2px;
    box-shadow: 0 10px 30px rgba(88,101,242,0.30);
    transition: all 0.25s ease;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 15px 40px rgba(88,101,242,0.45);
}

/* Result cards */
.result-positive {
    padding: 28px;
    border-radius: 20px;
    background: linear-gradient(
        135deg,
        rgba(180, 45, 75, 0.18),
        rgba(90, 25, 45, 0.30)
    );
    border: 1px solid rgba(255, 90, 120, 0.25);
    text-align: center;
}

.result-negative {
    padding: 28px;
    border-radius: 20px;
    background: linear-gradient(
        135deg,
        rgba(20, 160, 115, 0.16),
        rgba(15, 75, 65, 0.28)
    );
    border: 1px solid rgba(50, 210, 160, 0.25);
    text-align: center;
}

.result-icon {
    font-size: 50px;
}

.result-title {
    font-size: 27px;
    font-weight: 800;
    margin-top: 10px;
}

.result-text {
    color: #aeb8cc;
    margin-top: 8px;
    font-size: 14px;
}

/* Info cards */
.info-card {
    background: rgba(20, 26, 43, 0.75);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 18px;
    padding: 20px;
    text-align: center;
}

.info-number {
    font-size: 25px;
    font-weight: 800;
    color: #8ea7ff;
}

.info-label {
    font-size: 12px;
    color: #8994ad;
    margin-top: 4px;
}

/* Footer */
.footer {
    text-align: center;
    color: #66718a;
    font-size: 12px;
    margin-top: 35px;
    padding-top: 20px;
    border-top: 1px solid rgba(255,255,255,0.06);
}

.disclaimer {
    padding: 15px 18px;
    border-radius: 14px;
    background: rgba(255, 190, 70, 0.07);
    border: 1px solid rgba(255,190,70,0.15);
    color: #aaa898;
    font-size: 12px;
    margin-top: 20px;
}

</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# HERO
# ---------------------------------------------------------
st.markdown("""
<div class="hero">
    <div class="hero-content">
        <div class="brain">🧠</div>
        <div>
            <div class="hero-title">Parkinson's AI Detection</div>
            <div class="hero-subtitle">
                Machine Learning Based Voice Analysis System
            </div>
            <div class="badge">
                ✦ SVM • Voice Biomarkers • AI Assisted Screening
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# PATIENT INFORMATION
# ---------------------------------------------------------
st.markdown("""
<div class="section-card">
    <div class="section-title">👤 Patient Information</div>
    <div class="section-description">
        Enter basic patient information before performing the analysis.
    </div>
</div>
""", unsafe_allow_html=True)

patient_name = st.text_input(
    "Patient Name",
    placeholder="Enter patient's full name"
)


# ---------------------------------------------------------
# MODEL PARAMETERS
# ---------------------------------------------------------
st.markdown("""
<div class="section-card">
    <div class="section-title">🎙️ Voice Biomarker Analysis</div>
    <div class="section-description">
        Enter the 22 voice-analysis parameters required by the trained machine learning model.
    </div>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# DEFAULT VALUES
# ---------------------------------------------------------
defaults = [
    154.228641,
    197.104918,
    116.324631,
    0.006220,
    0.000044,
    0.003306,
    0.003446,
    0.009920,
    0.029709,
    0.282251,
    0.015664,
    0.017878,
    0.024081,
    0.046993,
    0.024847,
    21.885974,
    0.498536,
    0.718099,
    -5.684397,
    0.226510,
    2.381732,
    0.206552
]

features = [
    "MDVP:Fo(Hz)",
    "MDVP:Fhi(Hz)",
    "MDVP:Flo(Hz)",
    "MDVP:Jitter(%)",
    "MDVP:Jitter(Abs)",
    "MDVP:RAP",
    "MDVP:PPQ",
    "Jitter:DDP",
    "MDVP:Shimmer",
    "MDVP:Shimmer(dB)",
    "Shimmer:APQ3",
    "Shimmer:APQ5",
    "MDVP:APQ",
    "Shimmer:DDA",
    "NHR",
    "HNR",
    "RPDE",
    "DFA",
    "spread1",
    "spread2",
    "D2",
    "PPE"
]


# ---------------------------------------------------------
# INPUT GROUPS
# ---------------------------------------------------------

st.markdown("#### 🎵 Fundamental Frequency")

c1, c2, c3 = st.columns(3)

with c1:
    fo = st.number_input(features[0], value=defaults[0], format="%.6f")

with c2:
    fhi = st.number_input(features[1], value=defaults[1], format="%.6f")

with c3:
    flo = st.number_input(features[2], value=defaults[2], format="%.6f")


st.markdown("#### 📈 Jitter & Frequency Variation")

c1, c2, c3 = st.columns(3)

with c1:
    jitter = st.number_input(features[3], value=defaults[3], format="%.6f")

with c2:
    jitter_abs = st.number_input(features[4], value=defaults[4], format="%.6f")

with c3:
    rap = st.number_input(features[5], value=defaults[5], format="%.6f")

c1, c2 = st.columns(2)

with c1:
    ppq = st.number_input(features[6], value=defaults[6], format="%.6f")

with c2:
    ddp = st.number_input(features[7], value=defaults[7], format="%.6f")


st.markdown("#### 🔊 Shimmer & Amplitude Variation")

c1, c2, c3 = st.columns(3)

with c1:
    shimmer = st.number_input(features[8], value=defaults[8], format="%.6f")

with c2:
    shimmer_db = st.number_input(features[9], value=defaults[9], format="%.6f")

with c3:
    apq3 = st.number_input(features[10], value=defaults[10], format="%.6f")

c1, c2, c3 = st.columns(3)

with c1:
    apq5 = st.number_input(features[11], value=defaults[11], format="%.6f")

with c2:
    apq = st.number_input(features[12], value=defaults[12], format="%.6f")

with c3:
    dda = st.number_input(features[13], value=defaults[13], format="%.6f")


st.markdown("#### 🧬 Noise & Harmonic Features")

c1, c2 = st.columns(2)

with c1:
    nhr = st.number_input(features[14], value=defaults[14], format="%.6f")

with c2:
    hnr = st.number_input(features[15], value=defaults[15], format="%.6f")


st.markdown("#### 🧠 Nonlinear Voice Characteristics")

c1, c2, c3 = st.columns(3)

with c1:
    rpde = st.number_input(features[16], value=defaults[16], format="%.6f")

with c2:
    dfa = st.number_input(features[17], value=defaults[17], format="%.6f")

with c3:
    spread1 = st.number_input(features[18], value=defaults[18], format="%.6f")

c1, c2, c3 = st.columns(3)

with c1:
    spread2 = st.number_input(features[19], value=defaults[19], format="%.6f")

with c2:
    d2 = st.number_input(features[20], value=defaults[20], format="%.6f")

with c3:
    ppe = st.number_input(features[21], value=defaults[21], format="%.6f")


# ---------------------------------------------------------
# PREDICTION BUTTON
# ---------------------------------------------------------
st.markdown("<br>", unsafe_allow_html=True)

_, button_col, _ = st.columns([1, 2, 1])

with button_col:
    predict = st.button("🔍  Analyze Voice & Detect", use_container_width=True)


# ---------------------------------------------------------
# PREDICTION
# ---------------------------------------------------------
if predict:

    input_data = (
        fo, fhi, flo,
        jitter, jitter_abs, rap, ppq, ddp,
        shimmer, shimmer_db, apq3, apq5, apq, dda,
        nhr, hnr,
        rpde, dfa, spread1, spread2, d2, ppe
    )

    input_array = np.asarray(input_data).reshape(1, -1)

    scaled_input = scaler.transform(input_array)

    prediction = model.predict(scaled_input)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("## 📋 Analysis Result")

    if prediction[0] == 1:

        st.markdown("""
        <div class="result-positive">
            <div class="result-icon">⚠️</div>
            <div class="result-title">Parkinson's Positive</div>
            <div class="result-text">
                The machine learning model detected voice characteristics
                associated with Parkinson's disease.
            </div>
        </div>
        """, unsafe_allow_html=True)

    else:

        st.markdown("""
        <div class="result-negative">
            <div class="result-icon">✓</div>
            <div class="result-title">No Parkinson's Detected</div>
            <div class="result-text">
                The machine learning model did not detect voice characteristics
                associated with Parkinson's disease.
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Patient result
    if patient_name.strip():
        st.markdown(
            f"""
            <div style="
                text-align:center;
                margin-top:15px;
                color:#8994ad;
                font-size:13px;">
                Analysis completed for <b style="color:#d9def0;">
                {patient_name}</b>
            </div>
            """,
            unsafe_allow_html=True
        )


# ---------------------------------------------------------
# INFORMATION CARDS
# ---------------------------------------------------------
st.markdown("<br><br>", unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown("""
    <div class="info-card">
        <div class="info-number">22</div>
        <div class="info-label">Voice Features</div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown("""
    <div class="info-card">
        <div class="info-number">SVM</div>
        <div class="info-label">ML Algorithm</div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown("""
    <div class="info-card">
        <div class="info-number">AI</div>
        <div class="info-label">Analysis System</div>
    </div>
    """, unsafe_allow_html=True)

with c4:
    st.markdown("""
    <div class="info-card">
        <div class="info-number">Voice</div>
        <div class="info-label">Analysis Type</div>
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------
# DISCLAIMER
# ---------------------------------------------------------
st.markdown("""
<div class="disclaimer">
    ⚕️ <b>Important:</b> This application is an academic machine-learning
    project intended for preliminary screening and educational purposes.
    It is not a medical diagnosis and should not replace evaluation by
    a qualified healthcare professional.
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------
st.markdown("""
<div class="footer">
    Parkinson's AI Detection System &nbsp;•&nbsp;
    Machine Learning Project &nbsp;•&nbsp;
    SVM Voice Analysis
</div>
""", unsafe_allow_html=True)