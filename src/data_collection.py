"""
collect_data.py — ASL Hand Landmark Data Collector
====================================================
Uses OpenCV + MediaPipe to capture webcam frames and extract
21 hand landmarks (x, y, z) normalized to the wrist position.

Compatible with both mediapipe >= 0.10 (Tasks API) and < 0.10 (solutions API).
NOTE: mp.solutions is never accessed at module level — only inside
_LegacyDetector.__init__, so this file is safe to import on all versions.

Controls:
  A–Z       : label and save the current frame's landmarks
  ESC       : quit
  Window X  : quit (click the window close button)

Note: Q is now a collectible letter, not a quit key.
"""

import os
import time

import cv2
import mediapipe as mp

from utils import append_sample_to_csv

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
CSV_PATH = os.path.join(DATA_DIR, "dataset.csv")

# ── MediaPipe — detect which API is available ─────────────────────────────────
_USE_NEW_API = False
_MODEL_PATH  = os.path.join(os.path.dirname(__file__), "..", "hand_landmarker.task")

try:
    from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode
    from mediapipe.tasks.python import BaseOptions
    _USE_NEW_API = True
except ImportError:
    pass   # will use _LegacyDetector below


# ── Version-safe detector wrappers ────────────────────────────────────────────
class _NewAPIDetector:
    """mediapipe >= 0.10 HandLandmarker (VIDEO mode)."""

    def __init__(self):
        if not os.path.isfile(_MODEL_PATH):
            import urllib.request
            print("Downloading hand_landmarker.task (~23 MB) …")
            urllib.request.urlretrieve(
                "https://storage.googleapis.com/mediapipe-models/"
                "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
                _MODEL_PATH,
            )
            print("Download complete.")

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
        self._ts_ms += 33
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = self._detector.detect_for_video(mp_img, self._ts_ms)
        return result.hand_landmarks   # list[list[NormalizedLandmark]] or []

    def close(self):
        self._detector.close()

    def __enter__(self):  return self
    def __exit__(self, *_): self.close()


class _LegacyDetector:
    """
    mediapipe < 0.10 mp.solutions.hands.Hands.
    mp.solutions is accessed here — inside __init__ — never at module level.
    """

    def __init__(self):
        # Lazy: mp.solutions only touched when _USE_NEW_API is False
        hands_module = mp.solutions.hands
        self._hands  = hands_module.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
        )

    def process(self, rgb_frame):
        result = self._hands.process(rgb_frame)
        if not result.multi_hand_landmarks:
            return []
        return [list(h.landmark) for h in result.multi_hand_landmarks]

    def close(self):
        self._hands.close()

    def __enter__(self):  return self
    def __exit__(self, *_): self.close()


def make_detector():
    return _NewAPIDetector() if _USE_NEW_API else _LegacyDetector()


def draw_landmarks(frame, landmarks):
    """Pure OpenCV hand skeleton — no mp.solutions needed."""
    h, w = frame.shape[:2]
    CONNECTIONS = [
        (0,1),(1,2),(2,3),(3,4),
        (0,5),(5,6),(6,7),(7,8),
        (0,9),(9,10),(10,11),(11,12),
        (0,13),(13,14),(14,15),(15,16),
        (0,17),(17,18),(18,19),(19,20),
        (5,9),(9,13),(13,17),
    ]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for a, b in CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (0, 200, 0), 2, cv2.LINE_AA)
    for i, (x, y) in enumerate(pts):
        r = 5 if i in (4, 8, 12, 16, 20) else 3
        cv2.circle(frame, (x, y), r, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(frame, (x, y), r, (0, 150, 0),      1, cv2.LINE_AA)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam. Check your camera index.")

    status_msg   = "Press A-Z to collect | ESC to quit"
    status_color = (0, 255, 0)
    last_saved   = None
    save_time    = 0

    with make_detector() as detector:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            hands_list = detector.process(rgb)
            rgb.flags.writeable = True

            if hands_list:
                draw_landmarks(frame, hands_list[0])

            cv2.putText(frame, status_msg, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

            if last_saved and (time.time() - save_time) < 1.0:
                cv2.putText(frame, f"Saved: '{last_saved}'",
                            (10, 65), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0, 200, 255), 2)

            cv2.imshow("ASL Data Collector", frame)

            key = cv2.waitKey(1) & 0xFF

            # ── Quit: ESC key OR clicking the window X button ─────────────
            window_closed = cv2.getWindowProperty(
                "ASL Data Collector", cv2.WND_PROP_VISIBLE
            ) < 1

            if key == 27 or window_closed:   # 27 = ESC
                print("Quitting data collection.")
                break

            # ── Collect: any letter key A-Z (including Q) ─────────────────
            if ord("a") <= key <= ord("z") or ord("A") <= key <= ord("Z"):
                label = chr(key).upper()
                if hands_list:
                    append_sample_to_csv(CSV_PATH, label, hands_list[0])
                    last_saved   = label
                    save_time    = time.time()
                    status_msg   = "Collecting... (A-Z to save | ESC to quit)"
                    status_color = (0, 255, 0)
                    print(f"Saved sample for '{label}'")
                else:
                    status_msg   = "No hand detected - try again"
                    status_color = (0, 0, 255)

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nDataset saved to: {CSV_PATH}")


if __name__ == "__main__":
    main()