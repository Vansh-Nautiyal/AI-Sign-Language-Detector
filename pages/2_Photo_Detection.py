"""
pages/2_Photo_Detection.py — Static image / snapshot prediction
"""

import cv2
import streamlit as st

from shared import (
    RUNTIME_IMPORT_ERROR,
    decode_image,
    inject_styles,
    load_detector,
    load_predictor,
    process_frame,
    render_metrics,
)

inject_styles()

if RUNTIME_IMPORT_ERROR is not None:
    missing = getattr(RUNTIME_IMPORT_ERROR, "name", None) or str(RUNTIME_IMPORT_ERROR)
    st.error(f"Missing dependency: `{missing}`")
    st.stop()

# ── Page header ───────────────────────────────────────────────────────────────
st.title("📷 Photo Detection")
st.caption("Take a camera snapshot or upload a hand-sign image for instant prediction.")

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

# ── Image source ──────────────────────────────────────────────────────────────
photo_source = st.radio(
    "Photo source", ["Camera snapshot", "Upload image"], horizontal=True
)

left, right = st.columns([1.4, 1])

with left:
    if photo_source == "Camera snapshot":
        image_file = st.camera_input("Take a clear hand-sign snapshot")
    else:
        image_file = st.file_uploader(
            "Choose a hand-sign image", type=["jpg", "jpeg", "png", "bmp"]
        )

    if image_file is None:
        st.info("Add an image above to get a prediction.")
        st.stop()

    try:
        frame  = decode_image(image_file.getvalue())
        result = process_frame(frame, predictor, mirror_image=mirror_image)
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
        st.stop()

    st.image(
        cv2.cvtColor(result.frame_bgr, cv2.COLOR_BGR2RGB),
        caption="Annotated result",
        use_container_width=True,
    )

with right:
    render_metrics(result)
    st.divider()
    st.subheader("Detectable Letters")
    st.caption("A–Z except J and Z (motion signs)")
    st.markdown(
        "**A B C D E F G H I K L M**  \n"
        "**N O P Q R S T U V W X Y**"
    )