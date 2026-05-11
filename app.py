# Enhanced Flight Delay Predictor UI (Modern Airline Dashboard)


import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------

st.set_page_config(
    page_title="Flight Delay Predictor",
    page_icon="✈️",
    layout="wide"
)

# ---------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------

model = joblib.load("flight_delay_model.pkl")

# ---------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------

st.markdown("""
<style>

/* MAIN BACKGROUND */

.stApp {
    background-image: url("https://wallpaperbat.com/img/499020-hd-aviation-wallpaper-top-free-hd-aviation-background.png");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}

/* REMOVE STREAMLIT DEFAULT */

[data-testid="stHeader"] {
    background: rgba(0,0,0,0);
}

[data-testid="stToolbar"] {
    right: 2rem;
}

/* MAIN CONTAINER */

.main-container {
    background: rgba(5, 15, 35, 0.68);
    padding: 2rem;
    border-radius: 24px;
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255,255,255,0.08);
    box-shadow: 0 0 30px rgba(0,0,0,0.35);
}

/* TITLE */

.title {
    text-align: center;
    font-size: 58px;
    font-weight: 800;
    color: #5EEBFF;
    text-shadow: 0 0 18px rgba(94,235,255,0.5);
}

.subtitle {
    text-align: center;
    color: #EAF6FF;
    font-size: 20px;
    margin-bottom: 25px;
}

.section-title {
    color: #FFD166;
    font-size: 28px;
    font-weight: 700;
    margin-bottom: 20px;
}

/* INPUT LABELS */

label {
    color: #F8F9FA !important;
    font-weight: 600 !important;
    font-size: 16px !important;
}

/* SELECT BOX */

div[data-baseweb="select"] > div {
    background: rgba(8, 18, 35, 0.82) !important;
    border: 1px solid rgba(94,235,255,0.35) !important;
    border-radius: 14px !important;
    min-height: 52px;
    color: white !important;
    backdrop-filter: blur(10px);
    box-shadow: 0 0 12px rgba(94,235,255,0.12);
    transition: 0.3s ease;
}

/* SELECT HOVER */

div[data-baseweb="select"] > div:hover {
    border: 1px solid #5EEBFF !important;
    box-shadow: 0 0 18px rgba(94,235,255,0.35);
}

/* SELECT TEXT */

div[data-baseweb="select"] span {
    color: #EAF6FF !important;
    font-size: 16px;
}

/* DROPDOWN MENU */

ul {
    background: rgba(10,20,40,0.96) !important;
    border-radius: 14px !important;
    border: 1px solid rgba(94,235,255,0.18);
}

li {
    color: white !important;
}

li:hover {
    background: rgba(0,180,216,0.25) !important;
    color: #5EEBFF !important;
}

/* TEXT INPUT */

.stTextInput input {
    background: rgba(8, 18, 35, 0.82) !important;
    color: white !important;
    border-radius: 14px !important;
    border: 1px solid rgba(94,235,255,0.35) !important;
    height: 52px;
}

/* NUMBER INPUT */

.stNumberInput input {
    background: rgba(8, 18, 35, 0.82) !important;
    color: white !important;
    border-radius: 14px !important;
    border: 1px solid rgba(94,235,255,0.35) !important;
}

/* SLIDER */

.stSlider > div > div {
    color: #5EEBFF !important;
}

/* BUTTON */

.stButton > button {
    width: 100%;
    height: 60px;
    border-radius: 16px;
    border: none;
    background: linear-gradient(135deg, #00B4D8, #0077B6);
    color: white;
    font-size: 20px;
    font-weight: 700;
    transition: all 0.3s ease;
    box-shadow: 0 0 18px rgba(0,180,216,0.35);
}

.stButton > button:hover {
    transform: translateY(-3px) scale(1.02);
    background: linear-gradient(135deg, #48CAE4, #0096C7);
    box-shadow: 0 0 24px rgba(72,202,228,0.55);
}

/* RESULT BOX */

.result-box {
    background: rgba(10,20,40,0.82);
    padding: 25px;
    border-radius: 20px;
    border: 1px solid rgba(255,255,255,0.08);
    box-shadow: 0 0 18px rgba(0,0,0,0.25);
}

.result-delay {
    color: #FF6B6B;
    font-size: 34px;
    font-weight: 800;
}

.result-ontime {
    color: #52FFA8;
    font-size: 34px;
    font-weight: 800;
}

/* METRIC CARD */

.metric-card {
    background: rgba(8,18,35,0.85);
    padding: 25px;
    border-radius: 20px;
    text-align: center;
    border: 1px solid rgba(94,235,255,0.15);
    box-shadow: 0 0 18px rgba(0,0,0,0.25);
}

.metric-title {
    color: #EAF6FF;
    font-size: 22px;
    font-weight: 600;
}

.metric-value {
    color: #5EEBFF;
    font-size: 42px;
    font-weight: 800;
}

/* SIDEBAR */

section[data-testid="stSidebar"] {
    background: rgba(5,15,35,0.92);
    border-right: 1px solid rgba(255,255,255,0.08);
}

section[data-testid="stSidebar"] * {
    color: white !important;
}

</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# MAIN WRAPPER
# ---------------------------------------------------

st.markdown("<div class='main-container'>", unsafe_allow_html=True)

# ---------------------------------------------------
# HEADER
# ---------------------------------------------------

st.markdown("""
<div class='title'>✈️ Flight Delay Predictor</div>
<div class='subtitle'>Predict whether a flight will be delayed using Machine Learning</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------

st.sidebar.title("🛫 Flight Analytics")

st.sidebar.markdown("---")

st.sidebar.info("""
### Technologies Used

✅ Logistic Regression  
✅ Streamlit Frontend  
✅ Probability Calibration  
✅ ROC-AUC Evaluation  
✅ Interactive Dashboard
""")

st.sidebar.success("🎯 Model Accuracy: 85%")
st.sidebar.success("📈 ROC-AUC Score: 0.89")

# ---------------------------------------------------
# INPUT SECTION
# ---------------------------------------------------

st.markdown("<div class='section-title'>🛬 Enter Flight Details</div>", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:

    month = st.selectbox(
        "📅 Month",
        list(range(1, 13))
    )

    day_of_week = st.selectbox(
        "📆 Day Of Week",
        [1,2,3,4,5,6,7]
    )

    carrier = st.selectbox(
        "✈️ Airline Carrier",
        ['AA', 'DL', 'UA', 'WN', 'B6']
    )

    dep_time = st.slider(
        "⏰ Departure Hour",
        0,
        23,
        12
    )

with col2:

    origin = st.text_input(
        "🛫 Origin Airport",
        "JFK"
    )

    dest = st.text_input(
        "🛬 Destination Airport",
        "LAX"
    )

    distance = st.number_input(
        "📍 Distance (Miles)",
        min_value=50,
        max_value=10000,
        value=2475
    )

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------
# BUTTON
# ---------------------------------------------------

if st.button("Predict Flight Status"):

    input_df = pd.DataFrame({
        'Month': [month],
        'DayOfWeek': [day_of_week],
        'Carrier': [carrier],
        'Origin': [origin],
        'Dest': [dest],
        'Distance': [distance],
        'DepTimeHour': [dep_time]
    })

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.markdown("---")

    result_col1, result_col2 = st.columns([2,1])

    with result_col1:

        st.markdown("<div class='result-box'>", unsafe_allow_html=True)

        if prediction == 1:
            st.markdown(
                "<div class='result-delay'>⚠️ Flight Likely Delayed</div>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                "<div class='result-ontime'>✅ Flight Likely On Time</div>",
                unsafe_allow_html=True
            )

        st.markdown("### 📊 Delay Probability")

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            title={'text': "Delay Risk %"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#00B4D8"},
                'steps': [
                    {'range': [0, 40], 'color': "#52FFA8"},
                    {'range': [40, 70], 'color': "#FFD166"},
                    {'range': [70, 100], 'color': "#FF6B6B"}
                ]
            }
        ))

        fig.update_layout(
            height=350,
            paper_bgcolor='rgba(0,0,0,0)',
            font={'color': 'white', 'size': 16}
        )

        st.plotly_chart(fig, use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)

    with result_col2:

        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-title'>📈 Model Accuracy</div>
            <div class='metric-value'>85%</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-title'>🎯 ROC-AUC Score</div>
            <div class='metric-value'>0.89</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    with st.expander("🔍 View Flight Input Data"):
        st.dataframe(input_df, use_container_width=True)

st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------------------
# FOOTER
# ---------------------------------------------------






