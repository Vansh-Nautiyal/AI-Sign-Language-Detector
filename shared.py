"""
shared.py — Shared utilities for all pages
============================================
All imports, dataclasses, cached resources, and helper functions
used across all pages. Lives in the PROJECT ROOT (not in pages/)
so Streamlit never treats it as a page and no circular imports occur.

Every page imports what it needs:
    from shared import inject_styles, load_predictor, ...
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import streamlit as st

# ── Path setup ────────────────────────────────────────────────────────────────
# shared.py lives in the project root → __file__ parent IS the project root
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR  = os.path.join(BASE_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# ── Runtime imports (need cv2, mediapipe, tensorflow) ─────────────────────────
try:
    import cv2
    from app import (
        COLOR_GREEN, COLOR_RED, COLOR_TEAL, COLOR_WHITE, COLOR_YELLOW,
        FPSCounter, draw_hand_landmarks, draw_label_box,
        draw_rounded_rect, get_hand_bbox, make_detector,
    )
    from predict import ASLPredictor
    RUNTIME_IMPORT_ERROR = None
except ModuleNotFoundError as exc:
    cv2 = None
    ASLPredictor = None
    RUNTIME_IMPORT_ERROR = exc

# ── Report module ──────────────────────────────────────────────────────────────
try:
    from model_report import generate_report, ModelReport
    REPORT_AVAILABLE = True
except ModuleNotFoundError:
    generate_report  = None
    ModelReport      = None
    REPORT_AVAILABLE = False


# ── Dataclasses ───────────────────────────────────────────────────────────────
@dataclass
class PredictionResult:
    frame_bgr:       np.ndarray
    status:          str
    raw_letter:      str | None       = None
    smoothed_letter: str | None       = None
    confidence:      float            = 0.0
    probabilities:   pd.DataFrame | None = None


@dataclass
class RealtimeSessionResult:
    last_prediction:     PredictionResult | None
    guessed_confidences: pd.DataFrame
    prediction_log:      pd.DataFrame


# ── Cached resources ──────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_predictor(confidence: float, smoothing_window: int) -> "ASLPredictor":
    return ASLPredictor(confidence=confidence, window=smoothing_window)


@st.cache_resource(show_spinner=False)
def load_detector():
    return make_detector()


@st.cache_data(show_spinner=False)
def cached_report() -> "ModelReport":
    return generate_report()


# ── CSS injection ─────────────────────────────────────────────────────────────
def inject_styles() -> None:
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 1120px;
        }
        .hero { padding: 2.25rem 0 1.5rem; }
        .hero h1 {
            font-size: clamp(2.1rem, 5vw, 4rem);
            line-height: 1.02;
            margin-bottom: 1rem;
        }
        .hero p { color: #52616b; font-size: 1.15rem; max-width: 760px; }
        .choice-panel {
            border: 1px solid rgba(125,140,155,.35);
            border-radius: 8px;
            padding: 1.25rem;
            min-height: 150px;
            background: var(--secondary-background-color);
        }
        .choice-panel h3 { margin-top: 0; }
        .choice-panel p  { color: #52616b; }
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


# ── Image / frame helpers ─────────────────────────────────────────────────────
def decode_image(file_bytes: bytes) -> np.ndarray:
    data  = np.frombuffer(file_bytes, np.uint8)
    frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Could not read the selected image.")
    return frame


def top_probabilities(class_names: list[str], raw_probs: np.ndarray,
                      limit: int = 5) -> pd.DataFrame:
    rows = sorted(zip(class_names, raw_probs),
                  key=lambda item: float(item[1]), reverse=True)[:limit]
    return pd.DataFrame({
        "Letter":     [r[0] for r in rows],
        "Confidence": [float(r[1]) for r in rows],
    })


def guessed_confidences_table(confidences: dict[str, float]) -> pd.DataFrame:
    if not confidences:
        return pd.DataFrame(columns=["Letter", "Confidence"])
    return pd.DataFrame({
        "Letter":     list(confidences.keys()),
        "Confidence": [float(v) for v in confidences.values()],
    }).sort_values("Letter")


def prediction_log_table(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["Time", "Letter", "Confidence", "Status"])


def render_metrics(result: PredictionResult,
                   chart_title: str = "Top Confidence Scores") -> None:
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Prediction",  result.smoothed_letter or result.raw_letter or "-")
    col_b.metric("Confidence",  f"{result.confidence:.0%}" if result.confidence else "-")
    col_c.metric("Status",      result.status)
    if result.probabilities is not None and not result.probabilities.empty:
        st.subheader(chart_title)
        st.bar_chart(result.probabilities.set_index("Letter"),
                     y="Confidence", height=220)


def draw_confidence_history(frame: np.ndarray,
                             confidences: dict[str, float]) -> None:
    if not confidences:
        return
    rows         = sorted(confidences.items())
    chart_width  = 300
    row_height   = 24
    padding      = 12
    chart_height = padding * 2 + row_height * len(rows)
    x1 = max(10, frame.shape[1] - chart_width - 18)
    y1 = max(70, frame.shape[0] - chart_height - 22)
    x2 = frame.shape[1] - 18
    y2 = y1 + chart_height

    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (25, 25, 25), cv2.FILLED)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)
    cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_WHITE, 1)

    label_x   = x1 + padding
    bar_x     = x1 + 54
    bar_width = chart_width - 112
    for i, (letter, conf) in enumerate(rows):
        y = y1 + padding + (i * row_height) + 17
        cv2.putText(frame, letter, (label_x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_WHITE, 1)
        cv2.rectangle(frame, (bar_x, y-12), (bar_x+bar_width, y-3),
                      (75, 75, 75), cv2.FILLED)
        cv2.rectangle(frame, (bar_x, y-12),
                      (bar_x + int(bar_width * conf), y-3),
                      COLOR_GREEN if conf >= 0.80 else COLOR_YELLOW, cv2.FILLED)
        cv2.putText(frame, f"{conf:.0%}",
                    (bar_x + bar_width + 8, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_WHITE, 1)



def process_frame(frame_bgr: np.ndarray, predictor: "ASLPredictor",
                  mirror_image: bool) -> PredictionResult:
    frame = cv2.flip(frame_bgr, 1) if mirror_image else frame_bgr.copy()
    rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    hands_list = load_detector().process(rgb)
    rgb.flags.writeable = True

    if not hands_list:
        predictor.reset()
        draw_label_box(frame, "No hand detected", (16, 42),
                       font_scale=0.8, fg=COLOR_RED, bg=(28, 31, 36))
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
        draw_label_box(frame, f"Low confidence: {raw_letter} ({conf:.0%})",
                       (16, 42), font_scale=0.8, fg=COLOR_YELLOW, bg=(28, 31, 36))
        return PredictionResult(frame_bgr=frame, status="Low confidence",
                                raw_letter=raw_letter, confidence=conf,
                                probabilities=probs)

    color = COLOR_GREEN if conf >= 0.80 else COLOR_YELLOW
    cv2.putText(frame, smoothed or letter, (x1, max(y1-12, 52)),
                cv2.FONT_HERSHEY_SIMPLEX, 3.0, color, 4, cv2.LINE_AA)
    draw_label_box(frame, f"Prediction: {smoothed or letter} ({conf:.0%})",
                   (16, 42), font_scale=0.9, thickness=2,
                   fg=COLOR_WHITE, bg=(28, 31, 36))
    return PredictionResult(frame_bgr=frame, status="Prediction ready",
                            raw_letter=letter, smoothed_letter=smoothed,
                            confidence=conf, probabilities=probs)


def run_realtime_webcam(predictor: "ASLPredictor",
                        mirror_image: bool) -> RealtimeSessionResult:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)

    detector          = load_detector()
    fps_counter       = FPSCounter()
    last_result       = None
    guessed_confidences: dict[str, float] = {}
    prediction_rows:  list[dict]          = []
    last_logged_label = None
    last_logged_time  = 0.0
    window_name       = "ASL Sign Language Reader"
    window_focused    = False

    try:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1) if mirror_image else frame
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            hands_list = detector.process(rgb)
            rgb.flags.writeable = True
            fps_counter.tick()

            if hands_list:
                landmarks = hands_list[0]
                draw_hand_landmarks(frame, landmarks)
                x1, y1, x2, y2 = get_hand_bbox(landmarks, frame.shape)
                draw_rounded_rect(frame, (x1, y1), (x2, y2), COLOR_GREEN, 2)

                letter, conf, smoothed, raw_probs = \
                    predictor.predict_with_probabilities(landmarks)
                probs = top_probabilities(predictor.class_names, raw_probs)

                if letter is None:
                    raw_letter = probs.iloc[0]["Letter"] if not probs.empty else None
                    predictor.reset()
                    draw_label_box(frame, f"Low confidence: {raw_letter} ({conf:.0%})",
                                   (10, 40), font_scale=0.8,
                                   fg=COLOR_YELLOW, bg=(30, 30, 30))
                    last_result = PredictionResult(
                        frame_bgr=frame.copy(), status="Low confidence",
                        raw_letter=raw_letter, confidence=conf, probabilities=probs)
                else:
                    label = smoothed or letter
                    guessed_confidences[label] = max(
                        guessed_confidences.get(label, 0.0), conf)
                    now = time.time()
                    if label != last_logged_label or now - last_logged_time >= 1.0:
                        prediction_rows.append({
                            "Time":       time.strftime("%H:%M:%S"),
                            "Letter":     label,
                            "Confidence": f"{conf:.0%}",
                            "Status":     "Prediction ready",
                        })
                        last_logged_label = label
                        last_logged_time  = now

                    color = COLOR_GREEN if conf >= 0.80 else COLOR_YELLOW
                    cv2.putText(frame, label, (x1, max(y1-10, 40)),
                                cv2.FONT_HERSHEY_SIMPLEX, 3.0, color, 4, cv2.LINE_AA)
                    draw_label_box(frame, f"Prediction: {label} ({conf:.0%})",
                                   (10, 40), font_scale=1.0, thickness=2,
                                   fg=color, bg=(30, 30, 30))
                    last_result = PredictionResult(
                        frame_bgr=frame.copy(), status="Prediction ready",
                        raw_letter=letter, smoothed_letter=smoothed,
                        confidence=conf, probabilities=probs)
            else:
                predictor.reset()
                draw_label_box(frame, "No hand detected", (10, 40),
                               font_scale=0.8, fg=COLOR_RED, bg=(30, 30, 30))
                last_result = PredictionResult(
                    frame_bgr=frame.copy(), status="No hand detected")

            draw_label_box(frame, f"FPS: {fps_counter.fps:.1f}",
                           (frame.shape[1]-130, 40), font_scale=0.7,
                           fg=COLOR_TEAL, bg=(30, 30, 30))
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