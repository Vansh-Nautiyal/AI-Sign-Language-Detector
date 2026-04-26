"""
Streamlit frontend for the ASL Sign Language Detector.

Run with:
    streamlit run streamlit_app.py
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
import streamlit as st


BASE_DIR = os.path.dirname(__file__)
SRC_DIR = os.path.join(BASE_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

try:
    import cv2  # noqa: E402
    from app import (  # noqa: E402
        COLOR_GREEN,
        COLOR_RED,
        COLOR_WHITE,
        COLOR_YELLOW,
        draw_hand_landmarks,
        draw_label_box,
        draw_rounded_rect,
        get_hand_bbox,
        make_detector,
    )
    from predict import ASLPredictor  # noqa: E402
except ModuleNotFoundError as exc:
    cv2 = None
    ASLPredictor = None
    RUNTIME_IMPORT_ERROR = exc
else:
    RUNTIME_IMPORT_ERROR = None


@dataclass
class PredictionResult:
    frame_bgr: np.ndarray
    status: str
    raw_letter: str | None = None
    smoothed_letter: str | None = None
    confidence: float = 0.0
    probabilities: pd.DataFrame | None = None


st.set_page_config(
    page_title="ASL Sign Language Detector",
    layout="wide",
)


@st.cache_resource(show_spinner=False)
def load_predictor(confidence: float, smoothing_window: int) -> ASLPredictor:
    return ASLPredictor(confidence=confidence, window=smoothing_window)


@st.cache_resource(show_spinner=False)
def load_detector():
    return make_detector()


def decode_image(file_bytes: bytes) -> np.ndarray:
    data = np.frombuffer(file_bytes, np.uint8)
    frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Could not read the selected image.")
    return frame


def top_probabilities(class_names: list[str], raw_probs: np.ndarray, limit: int = 5) -> pd.DataFrame:
    rows = sorted(
        zip(class_names, raw_probs),
        key=lambda item: float(item[1]),
        reverse=True,
    )[:limit]
    return pd.DataFrame(
        {"Letter": [row[0] for row in rows], "Confidence": [float(row[1]) for row in rows]}
    )


def process_frame(
    frame_bgr: np.ndarray,
    predictor: ASLPredictor,
    mirror_image: bool,
) -> PredictionResult:
    frame = cv2.flip(frame_bgr, 1) if mirror_image else frame_bgr.copy()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    hands_list = load_detector().process(rgb)
    rgb.flags.writeable = True

    if not hands_list:
        predictor.reset()
        draw_label_box(
            frame,
            "No hand detected",
            (16, 42),
            font_scale=0.8,
            fg=COLOR_RED,
            bg=(28, 31, 36),
        )
        return PredictionResult(frame_bgr=frame, status="No hand detected")

    landmarks = hands_list[0]
    draw_hand_landmarks(frame, landmarks)
    x1, y1, x2, y2 = get_hand_bbox(landmarks, frame.shape)
    draw_rounded_rect(frame, (x1, y1), (x2, y2), COLOR_GREEN, 2)

    letter, conf, smoothed, raw_probs = predictor.predict_with_probabilities(landmarks)
    probs = top_probabilities(predictor.class_names, raw_probs)

    if letter is None:
        raw_letter = probs.iloc[0]["Letter"] if not probs.empty else None
        predictor.reset()
        draw_label_box(
            frame,
            f"Low confidence: {raw_letter} ({conf:.0%})",
            (16, 42),
            font_scale=0.8,
            fg=COLOR_YELLOW,
            bg=(28, 31, 36),
        )
        return PredictionResult(
            frame_bgr=frame,
            status="Low confidence",
            raw_letter=raw_letter,
            confidence=conf,
            probabilities=probs,
        )

    color = COLOR_GREEN if conf >= 0.80 else COLOR_YELLOW

    cv2.putText(
        frame,
        smoothed or letter,
        (x1, max(y1 - 12, 52)),
        cv2.FONT_HERSHEY_SIMPLEX,
        3.0,
        color,
        4,
        cv2.LINE_AA,
    )
    draw_label_box(
        frame,
        f"Prediction: {smoothed or letter} ({conf:.0%})",
        (16, 42),
        font_scale=0.9,
        thickness=2,
        fg=COLOR_WHITE,
        bg=(28, 31, 36),
    )

    return PredictionResult(
        frame_bgr=frame,
        status="Prediction ready",
        raw_letter=letter,
        smoothed_letter=smoothed,
        confidence=conf,
        probabilities=probs,
    )


def render_metrics(result: PredictionResult) -> None:
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Prediction", result.smoothed_letter or result.raw_letter or "-")
    col_b.metric("Confidence", f"{result.confidence:.0%}" if result.confidence else "-")
    col_c.metric("Status", result.status)

    if result.probabilities is not None and not result.probabilities.empty:
        st.bar_chart(
            result.probabilities.set_index("Letter"),
            y="Confidence",
            height=220,
        )


def main() -> None:
    st.title("ASL Sign Language Detector")

    if RUNTIME_IMPORT_ERROR is not None:
        missing_module = RUNTIME_IMPORT_ERROR.name or str(RUNTIME_IMPORT_ERROR)
        st.error(f"Missing dependency: `{missing_module}`")
        st.code("pip install -r requirements.txt", language="bash")
        st.warning(
            "This app also needs TensorFlow and MediaPipe. Use Python 3.9 to 3.11 "
            "for the best compatibility with those packages."
        )
        return

    with st.sidebar:
        st.header("Controls")
        source = st.radio("Input source", ["Camera snapshot", "Upload image"])
        confidence = st.slider("Confidence threshold", 0.10, 0.95, 0.60, 0.05)
        smoothing_window = st.slider("Smoothing window", 1, 20, 10, 1)
        mirror_image = st.toggle("Mirror image", value=True)

        if st.button("Reset smoothing", use_container_width=True):
            load_predictor(confidence, smoothing_window).reset()
            st.success("Smoothing buffer reset.")

    try:
        predictor = load_predictor(confidence, smoothing_window)
        load_detector()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()
    except Exception as exc:
        st.error(f"Could not load detector or model: {exc}")
        st.stop()

    left, right = st.columns([1.4, 1])

    with left:
        if source == "Camera snapshot":
            image_file = st.camera_input("Take a clear hand-sign snapshot")
        else:
            image_file = st.file_uploader(
                "Choose a hand-sign image",
                type=["jpg", "jpeg", "png", "bmp"],
            )

        if image_file is None:
            st.info("Add an image to start prediction.")
            return

        try:
            frame = decode_image(image_file.getvalue())
            result = process_frame(frame, predictor, mirror_image=mirror_image)
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
            return

        st.image(
            cv2.cvtColor(result.frame_bgr, cv2.COLOR_BGR2RGB),
            caption="Annotated result",
            use_container_width=True,
        )

    with right:
        render_metrics(result)
        st.subheader("Available Letters")
        st.write(" ".join(predictor.class_names))


if __name__ == "__main__":
    main()
