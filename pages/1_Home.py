"""
pages/1_Home.py — Landing page
"""

import streamlit as st
from shared import inject_styles, RUNTIME_IMPORT_ERROR

inject_styles()

# ── Dependency check ──────────────────────────────────────────────────────────
if RUNTIME_IMPORT_ERROR is not None:
    missing = getattr(RUNTIME_IMPORT_ERROR, "name", None) or str(RUNTIME_IMPORT_ERROR)
    st.error(f"Missing dependency: `{missing}`")
    st.code("pip install -r requirements.txt", language="bash")
    st.warning("Use Python 3.9–3.11 for best TensorFlow + MediaPipe compatibility.")
    st.stop()

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <section class="hero">
        <h1>ASL Sign Language Detector</h1>
        <p>
            Real-time American Sign Language recognition powered by MediaPipe
            hand landmarks and a lightweight neural network.
        </p>
    </section>
    """,
    unsafe_allow_html=True,
)

# ── Feature cards ─────────────────────────────────────────────────────────────
col_photo, col_live, col_report = st.columns(3, gap="medium")

with col_photo:
    st.markdown(
        """
        <div class="choice-panel">
            <h3>Photo Detection</h3>
            <p>
                Take a camera snapshot or upload a hand-sign image.
                Get an instant letter prediction with confidence score.
            </p>  
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("Detect from Photo") :
        st.switch_page("pages/2_Photo_Detection.py")


with col_live:
    st.markdown(
        """
        <div class="choice-panel">
            <h3>Realtime Webcam</h3>
            <p>
                Opens a live OpenCV window for continuous recognition.
                Press <code>Q</code> to quit, <code>R</code> to reset smoothing.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("Detect Using Webcam") :
        st.switch_page("pages/3_Realtime_Webcam.py")

with col_report:
    st.markdown(
        """
        <div class="choice-panel">
            <h3>Model Report</h3>
            <p>
                View test accuracy, per-letter F1 scores, confusion matrix,
                training curves, and dataset statistics.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("View Model Report") :
        st.switch_page("pages/4_Model_Report.py")

st.markdown(
    '<div class="status-strip">'
    'Detects A–Z except <strong>J</strong> and <strong>Z</strong> '
    '(motion-based signs). Predictions use majority-vote smoothing over '
    'the last 10 frames with a configurable confidence threshold.'
    '</div>',
    unsafe_allow_html=True,
)
