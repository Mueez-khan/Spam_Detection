import streamlit as st
import pandas as pd
import pickle as pkl

from click import style

st.set_page_config(
    page_title="Spam Shield",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Styling the app

st.markdown("""
    <style>
    /* Global Background */
    .stApp {
        background-color: #F8FAFC;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #EDF2F7;
        border-right: 1px solid #E2E8F0;
    }

    /* Title Header Card */
    .header-box {
        background-color: #3182CE;
        color: white;
        padding: 20px;
        border-radius: 8px;
        text-align: center;
        font-size: 32px;
        font-weight: bold;
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 25px;
    }

    /* Main Input Area */
    .stTextArea textarea {
        border-radius: 12px;
        border: 2px solid #E2E8F0;
        padding: 15px;
    }
    
    # div.stButton > button {
    #     background-color: black;
    #     color: white !important;
    #     border-radius: 8px;
    #     font-weight: bold;
    #     border: none;
    #     width: 100%;
    # }
    
    div.stButton > button:hover {
        background-color: #2f3542 !important;
        color: white !important;
        border: none;
    }

    /* Result Banner (Red) */
    .spam-banner {
        background-color: #E53E3E;
        color: white;
        padding: 10px;
        border-radius: 5px;
        text-align: center;
        font-weight: bold;
        margin-top: 20px;
    }
    
    /* Result Banner (Green) */
    .not_spam-banner {
        background-color: Green;
        color: white;
        padding: 10px;
        border-radius: 5px;
        text-align: center;
        font-weight: bold;
        margin-top: 20px;
    }
    

    /* Metric Cards */
    .metric-card {
        background-color: #1A365D;
        color: white;
        padding: 20px;
        border-radius: 8px;
        text-align: center;
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
    }
    .metric-label { font-size: 14px; opacity: 0.8; margin-bottom: 5px;}
    .metric-value { font-size: 24px; font-weight: bold;}

    /* Reasoning Box */
    .reasoning-box {
        background-color: #FEEBC1;
        color: red;
        padding: 15px;
        border-radius: 8px;
        border-left: 5px solid #DD6B20;
        margin-top: 20px;
    }
    </style>
    """, unsafe_allow_html=True)


# Prediction and model accessing code


model = pkl.load(open('spam_message.pkl', 'rb'))
vectorizer = pkl.load(open('vectoriaer.pkl', 'rb'))

st.markdown('<div class="header-box">🛡️ Spam Detection Shield</div>', unsafe_allow_html=True)
st.write("Paste your message below to analyze it for spam or phishing attempts.")

txt =  st.text_area("st.text_area", placeholder='e.g., "Congratulations! You\'ve won a $1000 gift card..."', label_visibility="collapsed")
col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    btn = st.button("🔍 Analyze Message", use_container_width=True )
if btn:

    # st.write(txt)
    vector = vectorizer.transform([txt])
    prediction = model.predict(vector)
    prediction_probability = model.predict_proba(vector)[0]

    if prediction == 0:
        st.markdown('<div class="not_spam-banner">## Not Spam Detected!</div>', unsafe_allow_html=True)
        m_col1, m_col2 = st.columns(2)
        with m_col1:
            st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-label">Classification</div>
                            <div class="metric-value">NOT SPAM</div>
                        </div>
                    """, unsafe_allow_html=True)
        with m_col2:
            st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-label">Confidence Score</div>
                            <div class="metric-value">{int(prediction_probability[0] * 100)}%</div>
                        </div>
                    """, unsafe_allow_html=True)



    else:
        st.markdown('<div class="spam-banner">## Potential Spam Detected!</div>', unsafe_allow_html=True)
        m_col3, m_col4 = st.columns(2)
        with m_col3:
            st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-label">Classification</div>
                            <div class="metric-value">SPAM</div>
                        </div>
                    """, unsafe_allow_html=True)
        with m_col4:
            st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-label">Confidence Score</div>
                            <div class="metric-value">{int(prediction_probability[1] * 100)}%</div>
                        </div>
                    """, unsafe_allow_html=True)

        st.markdown("""
                    <div class="reasoning-box">
                        <b>*Reasoning:</b> This message contains high-risk keywords associated with phishing.
                    </div>
                """, unsafe_allow_html=True)


