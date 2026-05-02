"""
streamlit_app.py — Entry point & navigation
============================================
Run with:
    streamlit run streamlit_app.py
"""

import streamlit as st

st.set_page_config(
    page_title="ASL Sign Language Detector",
    layout="wide",
)

pages = st.navigation(
    [
        st.Page("pages/1_Home.py",             title="Home"),
        st.Page("pages/2_Photo_Detection.py",  title="Photo Detection"),
        st.Page("pages/3_Realtime_Webcam.py",  title="Realtime Webcam"),
        st.Page("pages/4_Model_Report.py",     title="Model Report"),
    ],
    position="top",
)

pages.run()