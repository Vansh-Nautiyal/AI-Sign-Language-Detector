"""
app.py — Real-Time ASL Sign Language Reader
===========================================
Main entry-point. Captures webcam, runs MediaPipe hand tracking,
feeds landmarks to the MLP predictor, and overlays results on-screen.

Compatible with both:
  - mediapipe >= 0.10  (new Tasks API — uses hand_landmarker.task model)
  - mediapipe  < 0.10  (legacy mp.solutions API)

Usage:
  python app.py

Controls:
  q  : quit
  r  : reset smoothing buffer
"""

import os
import sys
import time
import logging
import collections
import cv2
import numpy as np

# Add src/ to path so we can import predict.py
SRC_DIR = os.path.join(os.path.dirname(__file__), "src")
sys.path.insert(0, SRC_DIR)

from predict import ASLPredictor

# ── Logging setup ─────────────────────────────────────────────────────────────
LOG_DIR  = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_PATH = os.path.join(LOG_DIR, "predictions.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("ASL")

# ── MediaPipe — detect which API is available ─────────────────────────────────
# NOTE: mp.solutions is NOT accessed at module level — it does not exist in
# mediapipe >= 0.10. All solutions access is deferred inside _LegacyDetector.
import mediapipe as mp

_USE_NEW_API = False
_MODEL_PATH  = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")

try:
    from mediapipe.tasks.python.vision import (
        HandLandmarker,
        HandLandmarkerOptions,
        RunningMode,
    )
    from mediapipe.tasks.python import BaseOptions
    _USE_NEW_API = True
    log.info("mediapipe >= 0.10 detected — using Tasks API.")
except ImportError:
    log.info("mediapipe < 0.10 detected — using legacy solutions API.")

# ── UI colours (BGR) ──────────────────────────────────────────────────────────
COLOR_GREEN  = (0,   220,  50)
COLOR_YELLOW = (0,   200, 255)
COLOR_RED    = (0,    50, 220)
COLOR_WHITE  = (255, 255, 255)
COLOR_BLACK  = (0,     0,   0)
COLOR_TEAL   = (180, 180,   0)


# ── Drawing helpers ───────────────────────────────────────────────────────────
def draw_rounded_rect(img, pt1, pt2, color, thickness=2, radius=10):
    """Draw a rectangle with rounded corners."""
    x1, y1 = pt1
    x2, y2 = pt2
    cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, thickness)
    cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, thickness)
    cv2.ellipse(img, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90,  color, thickness)
    cv2.ellipse(img, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90,  color, thickness)
    cv2.ellipse(img, (x1 + radius, y2 - radius), (radius, radius),  90, 0, 90,  color, thickness)
    cv2.ellipse(img, (x2 - radius, y2 - radius), (radius, radius),   0, 0, 90,  color, thickness)


def draw_label_box(frame, text, pos, font_scale=0.8, thickness=2,
                   fg=COLOR_WHITE, bg=COLOR_BLACK, padding=6):
    """Draw text with a filled background rectangle."""
    (tw, th), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
    )
    x, y = pos
    cv2.rectangle(
        frame,
        (x - padding, y - th - padding),
        (x + tw + padding, y + baseline + padding),
        bg, cv2.FILLED,
    )
    cv2.putText(frame, text, (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, fg, thickness)


def get_hand_bbox(landmarks, frame_shape, margin=20):
    """Return (x1, y1, x2, y2) pixel bounding box around the hand."""
    h, w = frame_shape[:2]
    xs = [lm.x * w for lm in landmarks]
    ys = [lm.y * h for lm in landmarks]
    x1 = max(0,     int(min(xs)) - margin)
    y1 = max(0,     int(min(ys)) - margin)
    x2 = min(w - 1, int(max(xs)) + margin)
    y2 = min(h - 1, int(max(ys)) + margin)
    return x1, y1, x2, y2


def draw_hand_landmarks(frame, landmarks):
    """
    Draw hand skeleton using pure OpenCV.
    No mp.solutions or mediapipe.framework imports needed — works on all versions.
    """
    h, w = frame.shape[:2]

    # All 21 MediaPipe hand connection pairs
    CONNECTIONS = [
        (0,1),(1,2),(2,3),(3,4),           # thumb
        (0,5),(5,6),(6,7),(7,8),           # index
        (0,9),(9,10),(10,11),(11,12),      # middle
        (0,13),(13,14),(14,15),(15,16),    # ring
        (0,17),(17,18),(18,19),(19,20),    # pinky
        (5,9),(9,13),(13,17),              # palm cross-bar
    ]

    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

    for a, b in CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (0, 200, 0), 2, cv2.LINE_AA)

    for i, (x, y) in enumerate(pts):
        r = 5 if i in (4, 8, 12, 16, 20) else 3   # fingertips slightly larger
        cv2.circle(frame, (x, y), r, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(frame, (x, y), r, (0, 150, 0),      1, cv2.LINE_AA)


# ── FPS tracker ───────────────────────────────────────────────────────────────
class FPSCounter:
    def __init__(self, window=30):
        self._times = collections.deque(maxlen=window)

    def tick(self):
        self._times.append(time.perf_counter())

    @property
    def fps(self):
        if len(self._times) < 2:
            return 0.0
        return (len(self._times) - 1) / (self._times[-1] - self._times[0])


# ── Version-safe detector wrappers ────────────────────────────────────────────
class _NewAPIDetector:
    """mediapipe >= 0.10 HandLandmarker (VIDEO mode)."""

    def __init__(self):
        if not os.path.isfile(_MODEL_PATH):
            log.info("Downloading hand_landmarker.task (~23 MB) …")
            import urllib.request
            urllib.request.urlretrieve(
                "https://storage.googleapis.com/mediapipe-models/"
                "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
                _MODEL_PATH,
            )
            log.info("Download complete.")

        opts = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=_MODEL_PATH),
            running_mode=RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.7,
            min_hand_presence_confidence=0.7,
            min_tracking_confidence=0.7,
        )
        self._detector = HandLandmarker.create_from_options(opts)
        self._ts_ms    = 0

    def process(self, rgb_frame):
        """Returns list[list[NormalizedLandmark]] — one sub-list per hand."""
        self._ts_ms += 33
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = self._detector.detect_for_video(mp_img, self._ts_ms)
        return result.hand_landmarks

    def close(self):
        self._detector.close()

    def __enter__(self):  return self
    def __exit__(self, *_): self.close()


class _LegacyDetector:
    """
    mediapipe < 0.10 mp.solutions.hands.Hands.
    mp.solutions is accessed here — inside __init__ — never at module level,
    so importing this file on mediapipe >= 0.10 is safe.
    """

    def __init__(self):
        # Lazy import: only reached when _USE_NEW_API is False
        hands_module = mp.solutions.hands
        self._hands  = hands_module.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
        )

    def process(self, rgb_frame):
        """Returns list[list[NormalizedLandmark]] for API parity with new detector."""
        result = self._hands.process(rgb_frame)
        if not result.multi_hand_landmarks:
            return []
        return [list(h.landmark) for h in result.multi_hand_landmarks]

    def close(self):
        self._hands.close()

    def __enter__(self):  return self
    def __exit__(self, *_): self.close()


def make_detector():
    """Return the right detector based on the installed mediapipe version."""
    return _NewAPIDetector() if _USE_NEW_API else _LegacyDetector()


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    log.info("Starting ASL Reader …")

    try:
        predictor = ASLPredictor()
        log.info("Model loaded successfully.")
    except FileNotFoundError as exc:
        log.error(str(exc))
        sys.exit(1)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        log.error("Cannot open webcam.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)

    fps_counter   = FPSCounter()
    last_log_time = 0
    LOG_INTERVAL  = 2.0

    with make_detector() as detector:
        while True:
            ret, frame = cap.read()
            if not ret:
                log.warning("Empty frame — skipping.")
                continue

            frame = cv2.flip(frame, 1)
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            hands_list = detector.process(rgb)
            rgb.flags.writeable = True

            fps_counter.tick()

            # ── Hand detected ─────────────────────────────────────────────
            if hands_list:
                landmarks = hands_list[0]

                draw_hand_landmarks(frame, landmarks)

                x1, y1, x2, y2 = get_hand_bbox(landmarks, frame.shape)
                draw_rounded_rect(frame, (x1, y1), (x2, y2), COLOR_GREEN, 2)

                letter, conf, smoothed = predictor.predict(landmarks)

                if smoothed:
                    conf_pct  = int(conf * 100)
                    disp_text = f"Prediction: {smoothed}  ({conf_pct}%)"
                    color     = COLOR_GREEN if conf >= 0.80 else COLOR_YELLOW

                    cv2.putText(frame, smoothed,
                                (x1, max(y1 - 10, 40)),
                                cv2.FONT_HERSHEY_SIMPLEX, 3.0, color, 4)
                    draw_label_box(frame, disp_text, (10, 40),
                                   font_scale=1.0, thickness=2,
                                   fg=color, bg=(30, 30, 30))

                    now = time.time()
                    if now - last_log_time > LOG_INTERVAL:
                        log.info(f"Letter={smoothed}  conf={conf:.2f}")
                        last_log_time = now
                else:
                    draw_label_box(frame, "Low confidence …", (10, 40),
                                   font_scale=0.8, fg=COLOR_YELLOW, bg=(30, 30, 30))
            else:
                predictor.reset()
                draw_label_box(frame, "No hand detected", (10, 40),
                               font_scale=0.8, fg=COLOR_RED, bg=(30, 30, 30))

            # ── HUD ───────────────────────────────────────────────────────
            draw_label_box(frame, f"FPS: {fps_counter.fps:.1f}",
                           (frame.shape[1] - 130, 40),
                           font_scale=0.7, fg=COLOR_TEAL, bg=(30, 30, 30))
            draw_label_box(frame, "Q: Quit   R: Reset",
                           (10, frame.shape[0] - 15),
                           font_scale=0.55, fg=COLOR_WHITE, bg=(30, 30, 30))

            cv2.imshow("ASL Sign Language Reader", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                log.info("User quit.")
                break
            if key == ord("r"):
                predictor.reset()
                log.info("Smoothing buffer reset.")

    cap.release()
    cv2.destroyAllWindows()
    log.info("Session ended.")


if __name__ == "__main__":
    main()