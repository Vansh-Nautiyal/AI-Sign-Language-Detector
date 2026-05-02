"""
pages/3_Realtime_Webcam.py — Live OpenCV webcam window
"""

import streamlit as st

from shared import (
    RUNTIME_IMPORT_ERROR,
    inject_styles,
    load_detector,
    load_predictor,
    run_realtime_webcam,
)

inject_styles()

if RUNTIME_IMPORT_ERROR is not None:
    missing = getattr(RUNTIME_IMPORT_ERROR, "name", None) or str(RUNTIME_IMPORT_ERROR)
    st.error(f"Missing dependency: `{missing}`")
    st.stop()

# ── Page header ───────────────────────────────────────────────────────────────
st.title("🎥 Realtime Webcam")
st.caption("Opens a live OpenCV desktop window for continuous hand sign recognition.")

# ── Sidebar controls ──────────────────────────────────────────────────────────
with st.sidebar:
    st.subheader("Controls")
    confidence       = st.slider("Confidence threshold", 0.10, 0.95, 0.60, 0.05)
    smoothing_window = st.slider("Smoothing window",     1,    20,   10,   1)
    mirror_image     = st.toggle("Mirror image", value=True)

    if st.button("Reset smoothing", use_container_width=True):
        load_predictor(confidence, smoothing_window).reset()
        st.success("Smoothing buffer reset.")

# ── Load model ────────────────────────────────────────────────────────────────
try:
    predictor = load_predictor(confidence, smoothing_window)
    load_detector()
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()
except Exception as exc:
    st.error(f"Could not load model: {exc}")
    st.stop()

# ── Two-column layout ─────────────────────────────────────────────────────────
col_left, col_right = st.columns([1, 1.2], gap="large")

with col_left:
    launch = st.button("▶  Open realtime app", type="primary", use_container_width=True)

    # J / Z warning
    st.markdown(
        """
        <div style="
            margin-top: 1rem;
            padding: .75rem 1rem;
            background: rgba(255,193,7,.10);
            border-left: 3px solid #ffc107;
            border-radius: 4px;
            font-size: .875rem;
            color: #b8860b;
        ">
            ⚠️ Signs for letter <strong>'J'</strong> and <strong>'Z'</strong>
            cannot be detected as they require motion.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("")
    st.markdown(
        """
        **Window keyboard shortcuts:**
        | Key | Action |
        |-----|--------|
        | `Q` | Quit the webcam window |
        | `R` | Reset the smoothing buffer |
        """
    )

with col_right:
    img_path = "images/Hand_Signs.png"
    if img_path:
        st.subheader("ASL Hand Signs")
        st.image(
            img_path,
            caption="Reference chart — all detectable letters",
            use_container_width=True,
        )
    else:
        st.info(
            "Reference image not found.  \n"
            "Place `Hand_Signs.png` in `data/images/` to show it here."
        )

# ── Run session ───────────────────────────────────────────────────────────────
if not launch:
    st.stop()

status = st.empty()
status.info("Opening webcam window — return here after pressing Q in the OpenCV window.")

try:
    session = run_realtime_webcam(predictor, mirror_image=mirror_image)
except Exception as exc:
    st.error(f"Realtime webcam failed: {exc}")
    st.stop()

status.success("Session ended.")

if session.last_prediction is None:
    st.info("No frames were processed before the session ended.")
    st.stop()

# ── Post-session results ──────────────────────────────────────────────────────
st.divider()
st.subheader("Session summary")

if not session.guessed_confidences.empty:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Best confidence per letter**")
        st.dataframe(session.guessed_confidences,
                     use_container_width=True, hide_index=True)
    with c2:
        st.markdown("**Confidence chart**")
        st.bar_chart(
            session.guessed_confidences.set_index("Letter")[["Confidence"]],
            height=300,
        )

st.subheader("Prediction log")
if session.prediction_log.empty:
    st.info("No predictions were recorded.")
else:
    st.dataframe(session.prediction_log,
                 use_container_width=True, hide_index=True)