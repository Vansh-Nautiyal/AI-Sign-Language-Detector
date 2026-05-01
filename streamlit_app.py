"""
Streamlit frontend for the ASL Sign Language Detector.

"""

from __future__ import annotations

import os
import sys
import time
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
        COLOR_TEAL,
        COLOR_WHITE,
        COLOR_YELLOW,
        FPSCounter,
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


@dataclass
class RealtimeSessionResult:
    last_prediction: PredictionResult | None
    guessed_confidences: pd.DataFrame
    prediction_log: pd.DataFrame


st.set_page_config(
    page_title="ASL Sign Language Detector",
    layout="wide",
)


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 1120px;
        }
        .hero {
            padding: 2.25rem 0 1.5rem;
        }
        .hero h1 {
            font-size: clamp(2.1rem, 5vw, 4rem);
            line-height: 1.02;
            margin-bottom: 1rem;
        }
        .hero p {
            color: #52616b;
            font-size: 1.15rem;
            max-width: 760px;
        }
        .choice-panel {
            border: 1px solid rgba(125, 140, 155, .35);
            border-radius: 8px;
            padding: 1.25rem;
            min-height: 150px;
            background: var(--secondary-background-color);
        }
        .choice-panel h3 {
            margin-top: 0;
        }
        .choice-panel p {
            color: #52616b;
        }
        .status-strip {
            border-top: 1px solid #e7edf2;
            border-bottom: 1px solid #e7edf2;
            padding: .8rem 0;
            color: #52616b;
        }
        </style>
        """,
        unsafe_allow_html=True,
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


def guessed_confidences_table(confidences: dict[str, float]) -> pd.DataFrame:
    if not confidences:
        return pd.DataFrame(columns=["Letter", "Confidence"])

    return pd.DataFrame(
        {
            "Letter": list(confidences.keys()),
            "Confidence": [float(value) for value in confidences.values()],
        }
    ).sort_values("Letter")


def draw_confidence_history(frame: np.ndarray, confidences: dict[str, float]) -> None:
    if not confidences:
        return

    rows = sorted(confidences.items())
    chart_width = 300
    row_height = 24
    padding = 12
    chart_height = padding * 2 + row_height * len(rows)
    x1 = max(10, frame.shape[1] - chart_width - 18)
    y1 = max(70, frame.shape[0] - chart_height - 22)
    x2 = frame.shape[1] - 18
    y2 = y1 + chart_height

    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (25, 25, 25), cv2.FILLED)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)
    cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_WHITE, 1)

    label_x = x1 + padding
    bar_x = x1 + 54
    bar_width = chart_width - 112
    for index, (letter, conf) in enumerate(rows):
        y = y1 + padding + (index * row_height) + 17
        cv2.putText(frame, letter, (label_x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_WHITE, 1)
        cv2.rectangle(frame, (bar_x, y - 12), (bar_x + bar_width, y - 3), (75, 75, 75), cv2.FILLED)
        cv2.rectangle(
            frame,
            (bar_x, y - 12),
            (bar_x + int(bar_width * conf), y - 3),
            COLOR_GREEN if conf >= 0.80 else COLOR_YELLOW,
            cv2.FILLED,
        )
        cv2.putText(
            frame,
            f"{conf:.0%}",
            (bar_x + bar_width + 8, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            COLOR_WHITE,
            1,
        )


def prediction_log_table(rows: list[dict[str, object]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["Time", "Letter", "Confidence", "Status"])


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


def run_realtime_webcam(
    predictor: ASLPredictor,
    mirror_image: bool,
) -> RealtimeSessionResult:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    detector = load_detector()
    fps_counter = FPSCounter()
    last_result: PredictionResult | None = None
    guessed_confidences: dict[str, float] = {}
    prediction_rows: list[dict[str, object]] = []
    last_logged_label: str | None = None
    last_logged_time = 0.0
    window_name = "ASL Sign Language Reader"
    window_focused = False

    try:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1) if mirror_image else frame
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            hands_list = detector.process(rgb)
            rgb.flags.writeable = True
            fps_counter.tick()

            if hands_list:
                landmarks = hands_list[0]
                draw_hand_landmarks(frame, landmarks)
                x1, y1, x2, y2 = get_hand_bbox(landmarks, frame.shape)
                draw_rounded_rect(frame, (x1, y1), (x2, y2), COLOR_GREEN, 2)

                letter, conf, smoothed, raw_probs = predictor.predict_with_probabilities(landmarks)
                probs = top_probabilities(predictor.class_names, raw_probs)

                if letter is None:
                    raw_letter = probs.iloc[0]["Letter"] if not probs.empty else None
                    predictor.reset()
                    last_result = PredictionResult(
                        frame_bgr=frame.copy(),
                        status="Low confidence",
                        raw_letter=raw_letter,
                        confidence=conf,
                        probabilities=probs,
                    )
                else:
                    label = smoothed or letter
                    guessed_confidences[label] = max(guessed_confidences.get(label, 0.0), conf)
                    now = time.time()
                    if label != last_logged_label or now - last_logged_time >= 1.0:
                        prediction_rows.append(
                            {
                                "Time": time.strftime("%H:%M:%S"),
                                "Letter": label,
                                "Confidence": f"{conf:.0%}",
                                "Status": "Prediction ready",
                            }
                        )
                        last_logged_label = label
                        last_logged_time = now
                    color = COLOR_GREEN if conf >= 0.80 else COLOR_YELLOW
                    cv2.putText(
                        frame,
                        label,
                        (x1, max(y1 - 10, 40)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        3.0,
                        color,
                        4,
                        cv2.LINE_AA,
                    )
                    last_result = PredictionResult(
                        frame_bgr=frame.copy(),
                        status="Prediction ready",
                        raw_letter=letter,
                        smoothed_letter=smoothed,
                        confidence=conf,
                        probabilities=probs,
                    )
            else:
                predictor.reset()
                last_result = PredictionResult(frame_bgr=frame.copy(), status="No hand detected")

            draw_label_box(
                frame,
                f"FPS: {fps_counter.fps:.1f}",
                (frame.shape[1] - 130, 40),
                font_scale=0.7,
                fg=COLOR_TEAL,
                bg=(30, 30, 30),
            )
            draw_label_box(
                frame,
                "Q: Quit   R: Reset",
                (10, frame.shape[0] - 15),
                font_scale=0.55,
                fg=COLOR_WHITE,
                bg=(30, 30, 30),
            )
            draw_confidence_history(frame, guessed_confidences)

            cv2.imshow(window_name, frame)
            if not window_focused:
                cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
                cv2.waitKey(1)
                cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 0)
                window_focused = True

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("r"):
                predictor.reset()
    finally:
        cap.release()
        cv2.destroyAllWindows()

    return RealtimeSessionResult(
        last_prediction=last_result,
        guessed_confidences=guessed_confidences_table(guessed_confidences),
        prediction_log=prediction_log_table(prediction_rows),
    )


def render_metrics(result: PredictionResult, chart_title: str = "Top Confidence Scores") -> None:
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Prediction", result.smoothed_letter or result.raw_letter or "-")
    col_b.metric("Confidence", f"{result.confidence:.0%}" if result.confidence else "-")
    col_c.metric("Status", result.status)

    if result.probabilities is not None and not result.probabilities.empty:
        st.subheader(chart_title)
        st.bar_chart(
            result.probabilities.set_index("Letter"),
            y="Confidence",
            height=220,
        )


def render_landing() -> None:
    st.markdown(
        """
        <section class="hero">
            <h1>ASL Sign Language Detector</h1>
            <p>
                Choose a detection mode to start. Use a single photo for a quick
                prediction with the confidence graph, or open the realtime webcam
                reader for continuous recognition.
            </p>
        </section>
        """,
        unsafe_allow_html=True,
    )

    col_photo, col_live = st.columns(2, gap="large")
    with col_photo:
        st.markdown(
            """
            <div class="choice-panel">
                <h3>Photo detection</h3>
                <p>Capture a camera snapshot or upload an existing hand-sign image.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Start photo detection", type="primary", width='stretch'):
            st.session_state.mode = "photo"
            st.rerun()

    with col_live:
        st.markdown(
            """
            <div class="choice-panel">
                <h3>Realtime webcam</h3>
                <p>Open a live OpenCV window. Press q to quit or r to reset smoothing.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Start realtime webcam", width='stretch'):
            st.session_state.mode = "realtime"
            st.rerun()

    st.markdown(
        '<div class="status-strip">Predictions include smoothing, confidence percentage, and top-letter probability graph where available.</div>',
        unsafe_allow_html=True,
    )


def render_controls() -> tuple[float, int, bool]:
    with st.container(border=True):
        st.subheader("Controls")
        nav_col, _ = st.columns([1, 3])
        with nav_col:
            if st.button("Back to Home (Landing)", width='stretch'):
                st.session_state.mode = None
                st.rerun()

        col_conf, col_smooth, col_mirror, col_reset = st.columns([1.35, 1.35, .85, .95])
        with col_conf:
            confidence = st.slider("Confidence threshold", 0.10, 0.95, 0.60, 0.05)
        with col_smooth:
            smoothing_window = st.slider("Smoothing window", 1, 20, 10, 1)
        with col_mirror:
            mirror_image = st.toggle("Mirror image", value=True)
        with col_reset:
            if st.button("Reset smoothing", width='stretch'):
                load_predictor(confidence, smoothing_window).reset()
                st.success("Smoothing buffer reset.")

    return confidence, smoothing_window, mirror_image


def render_photo_mode(predictor: ASLPredictor, mirror_image: bool) -> None:
    st.title("Photo Detection")
    st.caption("Use the built-in camera snapshot or upload a precaptured image.")

    photo_source = st.radio(
        "Photo source",
        ["Camera snapshot", "Upload image"],
        horizontal=True,
    )

    left, right = st.columns([1.4, 1])

    with left:
        if photo_source == "Camera snapshot":
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
            width='stretch',
        )

    with right:
        render_metrics(result)
        st.subheader("Available Letters : A-Z (except J and Z, as they require motion signs)")


def render_realtime_mode(predictor: ASLPredictor, mirror_image: bool) -> None:
    st.title("Realtime Webcam")
    st.caption("This opens a desktop OpenCV window. Press q to quit, or r to reset smoothing.")

    col_launch, col_ref = st.columns([1, 1.5], gap="large")
    
    with col_launch:
        launch = st.button("Open realtime app", type="primary", use_container_width=False)
        if not launch:
            st.info("Click the button above to start the live webcam reader.")
            return
    
    with col_ref:
        st.subheader("ASL Hand Signs")
        if os.path.isfile(os.path.join(BASE_DIR, "hand_signs.png")):
            st.image(os.path.join(BASE_DIR, "hand_signs.png"), width=300)
        st.caption("⚠️ Signs for 'J' and 'Z' cannot be classified as they require motion")

    status = st.empty()
    status.info("Opening webcam window. Return here after pressing q in the OpenCV window.")

    try:
        session_result = run_realtime_webcam(predictor, mirror_image=mirror_image)
    except Exception as exc:
        st.error(f"Realtime webcam failed: {exc}")
        return

    status.success("Realtime session ended.")

    if session_result.last_prediction is None:
        st.info("No frames were processed before the session ended.")
        return

    st.subheader("Prediction Log")
    if session_result.prediction_log.empty:
        st.info("No prediction log entries were recorded.")
    else:
        st.dataframe(session_result.prediction_log, width='stretch', hide_index=True)


def main() -> None:
    inject_styles()

    if RUNTIME_IMPORT_ERROR is not None:
        missing_module = RUNTIME_IMPORT_ERROR.name or str(RUNTIME_IMPORT_ERROR)
        st.error(f"Missing dependency: `{missing_module}`")
        st.code("pip install -r requirements.txt", language="bash")
        st.warning(
            "This app also needs TensorFlow and MediaPipe. Use Python 3.9 to 3.11 "
            "for the best compatibility with those packages."
        )
        return

    if "mode" not in st.session_state:
        st.session_state.mode = None

    if st.session_state.mode is None:
        render_landing()
        return

    confidence, smoothing_window, mirror_image = render_controls()

    try:
        predictor = load_predictor(confidence, smoothing_window)
        load_detector()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()
    except Exception as exc:
        st.error(f"Could not load detector or model: {exc}")
        st.stop()

    if st.session_state.mode == "photo":
        render_photo_mode(predictor, mirror_image)
    elif st.session_state.mode == "realtime":
        render_realtime_mode(predictor, mirror_image)
    else:
        st.session_state.mode = None
        st.rerun()


if __name__ == "__main__":
    main()
