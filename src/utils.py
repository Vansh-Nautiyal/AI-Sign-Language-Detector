"""
utils.py — Shared Utilities for the ASL Sign Language Reader
=============================================================
All reusable, stateless logic lives here so that collect_data.py,
train_model.py, predict.py, and app.py stay thin and DRY.

Sections
--------
1. Constants & CSV schema
2. Landmark normalization
3. Feature extraction
4. Majority voting
5. Label encoding / decoding
"""

import collections
import csv
import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# ─────────────────────────────────────────────────────────────────────────────
# 1. Constants & CSV schema
# ─────────────────────────────────────────────────────────────────────────────

NUM_LANDMARKS: int = 21          # MediaPipe Hands produces 21 landmarks
NUM_AXES: int      = 3           # x, y, z per landmark
NUM_FEATURES: int  = NUM_LANDMARKS * NUM_AXES   # 63 total features

# Column names: label, x0,y0,z0, x1,y1,z1, …, x20,y20,z20
CSV_HEADER: list[str] = ["label"] + [
    f"{axis}{i}"
    for i in range(NUM_LANDMARKS)
    for axis in ("x", "y", "z")
]


# ─────────────────────────────────────────────────────────────────────────────
# 2. Landmark normalization
# ─────────────────────────────────────────────────────────────────────────────

def normalize_landmarks(landmarks) -> list[float]:
    """
    Make hand position translation-invariant by subtracting the wrist
    (landmark 0) from every landmark.

    Parameters
    ----------
    landmarks : sequence of MediaPipe NormalizedLandmark objects
        Must contain exactly NUM_LANDMARKS (21) items, each with
        `.x`, `.y`, `.z` float attributes in [0, 1].

    Returns
    -------
    list[float]
        Flat list of NUM_FEATURES (63) floats:
        [x0,y0,z0, x1,y1,z1, …, x20,y20,z20]
        where each coordinate is relative to the wrist.

    Example
    -------
    >>> features = normalize_landmarks(hand_result.landmark)
    >>> len(features)
    63
    """
    wrist = landmarks[0]
    flat: list[float] = []
    for lm in landmarks:
        flat.extend([
            lm.x - wrist.x,
            lm.y - wrist.y,
            lm.z - wrist.z,
        ])
    return flat


# ─────────────────────────────────────────────────────────────────────────────
# 3. Feature extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_features(landmarks) -> np.ndarray:
    """
    Normalize landmarks and return a model-ready (1, 63) float32 array.

    This is the single source of truth for the input format shared by
    both training (collect_data.py) and inference (predict.py / app.py).

    Parameters
    ----------
    landmarks : sequence of MediaPipe NormalizedLandmark objects

    Returns
    -------
    np.ndarray, shape (1, 63), dtype float32
        Ready to pass directly to ``model.predict()``.

    Example
    -------
    >>> X = extract_features(hand_result.landmark)
    >>> X.shape
    (1, 63)
    """
    flat = normalize_landmarks(landmarks)
    return np.array(flat, dtype=np.float32).reshape(1, -1)


def landmarks_to_row(label: str, landmarks) -> list:
    """
    Build one CSV row: [label, x0, y0, z0, …, x20, y20, z20].

    Parameters
    ----------
    label : str
        Single uppercase letter, e.g. ``"A"``.
    landmarks : sequence of MediaPipe NormalizedLandmark objects

    Returns
    -------
    list
        Length NUM_FEATURES + 1 (64 items).
    """
    return [label] + normalize_landmarks(landmarks)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Majority voting
# ─────────────────────────────────────────────────────────────────────────────

class MajorityVoter:
    """
    Temporal smoother based on majority voting over a sliding window.

    Keeps the last ``window`` predictions in a deque and returns the
    most-common label, reducing single-frame noise.

    Parameters
    ----------
    window : int
        Number of recent predictions to consider (default: 10).

    Example
    -------
    >>> voter = MajorityVoter(window=10)
    >>> voter.update("A")
    >>> voter.update("A")
    >>> voter.update("B")
    >>> voter.vote()
    'A'
    """

    def __init__(self, window: int = 10) -> None:
        self._window  = window
        self._history: collections.deque[str] = collections.deque(maxlen=window)

    # ── Public API ────────────────────────────────────────────────────────────

    def update(self, label: str) -> None:
        """Push a new prediction into the buffer."""
        self._history.append(label)

    def vote(self) -> str | None:
        """
        Return the most common label in the current window.

        Returns ``None`` when the buffer is empty.
        """
        if not self._history:
            return None
        counter = collections.Counter(self._history)
        return counter.most_common(1)[0][0]

    def reset(self) -> None:
        """Clear all stored predictions (e.g. when the hand leaves frame)."""
        self._history.clear()

    # ── Convenience ───────────────────────────────────────────────────────────

    def update_and_vote(self, label: str) -> str:
        """Push ``label`` then immediately return the current majority vote."""
        self.update(label)
        return self.vote()

    def __len__(self) -> int:
        return len(self._history)

    def __repr__(self) -> str:
        return (
            f"MajorityVoter(window={self._window}, "
            f"filled={len(self._history)}, vote={self.vote()!r})"
        )


def majority_vote(predictions: list[str]) -> str | None:
    """
    Stateless helper: return the most common label in ``predictions``.

    Parameters
    ----------
    predictions : list[str]
        Flat list of label strings.

    Returns
    -------
    str | None
        Most common label, or ``None`` if the list is empty.

    Example
    -------
    >>> majority_vote(["A", "A", "B", "A", "C"])
    'A'
    """
    if not predictions:
        return None
    return collections.Counter(predictions).most_common(1)[0][0]


# ─────────────────────────────────────────────────────────────────────────────
# 5. Label encoding / decoding
# ─────────────────────────────────────────────────────────────────────────────

class ASLLabelEncoder:
    """
    Thin wrapper around ``sklearn.LabelEncoder`` with persistence helpers
    for saving/loading the class list alongside the Keras model.

    The class list is stored as a plain-text file (one label per line) so
    it stays human-readable and independent of sklearn's version.

    Example
    -------
    >>> enc = ASLLabelEncoder()
    >>> y_int = enc.fit_transform(df["label"].values)
    >>> enc.save("model/class_names.txt")

    >>> enc2 = ASLLabelEncoder.load("model/class_names.txt")
    >>> enc2.decode(0)
    'A'
    """

    def __init__(self) -> None:
        self._le          = LabelEncoder()
        self.class_names: list[str] = []

    # ── Fit / transform ───────────────────────────────────────────────────────

    def fit_transform(self, labels: np.ndarray) -> np.ndarray:
        """
        Fit the encoder on ``labels`` and return integer-encoded array.

        Parameters
        ----------
        labels : array-like of str
            Raw string labels, e.g. ``["A", "B", "A", …]``.

        Returns
        -------
        np.ndarray of int
        """
        encoded          = self._le.fit_transform(labels)
        self.class_names = list(self._le.classes_)
        return encoded

    def encode(self, label: str) -> int:
        """Convert a single label string to its integer index."""
        return int(self._le.transform([label])[0])

    def decode(self, index: int) -> str:
        """Convert an integer index back to its label string."""
        return self.class_names[index]

    def decode_batch(self, indices: np.ndarray) -> list[str]:
        """Convert an array of integer indices to label strings."""
        return [self.class_names[i] for i in indices]

    @property
    def num_classes(self) -> int:
        """Number of unique labels seen during ``fit_transform``."""
        return len(self.class_names)

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """
        Write class names to a text file, one per line.

        Parameters
        ----------
        path : str
            Destination file path, e.g. ``"model/class_names.txt"``.
        """
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as f:
            f.write("\n".join(self.class_names))

    @classmethod
    def load(cls, path: str) -> "ASLLabelEncoder":
        """
        Reconstruct an encoder from a previously saved class-names file.

        Parameters
        ----------
        path : str
            Path to the text file produced by ``save()``.

        Returns
        -------
        ASLLabelEncoder
            Ready for ``encode`` / ``decode`` calls (no re-fitting needed).

        Raises
        ------
        FileNotFoundError
            If ``path`` does not exist.
        """
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Class-names file not found: {path}\n"
                "Run src/train_model.py first."
            )
        instance              = cls()
        instance.class_names  = open(path).read().strip().splitlines()
        # Reconstruct the sklearn encoder so encode() works too
        instance._le.classes_ = np.array(instance.class_names)
        return instance

    def __repr__(self) -> str:
        return (
            f"ASLLabelEncoder(num_classes={self.num_classes}, "
            f"classes={self.class_names})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# CSV I/O helpers (used by collect_data.py)
# ─────────────────────────────────────────────────────────────────────────────

def append_sample_to_csv(csv_path: str, label: str, landmarks) -> None:
    """
    Append one labelled sample to the dataset CSV.

    Creates the file with a header row on first call; subsequent calls
    append without rewriting the header.

    Parameters
    ----------
    csv_path : str
        Path to the CSV file (e.g. ``"data/dataset.csv"``).
    label : str
        Single uppercase letter, e.g. ``"B"``.
    landmarks : sequence of MediaPipe NormalizedLandmark objects
    """
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(CSV_HEADER)
        writer.writerow(landmarks_to_row(label, landmarks))


def load_dataset_from_csv(csv_path: str) -> tuple[np.ndarray, np.ndarray, "ASLLabelEncoder"]:
    """
    Load the landmark CSV and return (X, y, encoder).

    Parameters
    ----------
    csv_path : str
        Path to dataset CSV produced by collect_data.py.

    Returns
    -------
    X : np.ndarray, shape (N, 63), float32
        Feature matrix.
    y : np.ndarray, shape (N,), int
        Integer-encoded labels.
    encoder : ASLLabelEncoder
        Fitted encoder; call ``encoder.decode(i)`` to get letter back.

    Raises
    ------
    SystemExit
        If the file does not exist, with a helpful message.
    """
    import sys
    if not os.path.isfile(csv_path):
        sys.exit(
            f"\n[ERROR] Dataset not found: {csv_path}\n"
            "Run  src/collect_data.py  first.\n"
        )
    df = pd.read_csv(csv_path)
    if "label" not in df.columns and len(df.columns) == len(CSV_HEADER):
        df = pd.read_csv(csv_path, header=None, names=CSV_HEADER)

    # Some dataset operations can accidentally append a CSV header row into
    # the middle of the file. Drop those rows before numeric conversion.
    df = df[df["label"].astype(str).str.lower() != "label"].copy()
    df = df.dropna(subset=["label"])

    encoder = ASLLabelEncoder()
    y       = encoder.fit_transform(df["label"].values)
    feature_df = df.drop(columns=["label"]).apply(pd.to_numeric, errors="coerce")
    bad_rows = feature_df.isna().any(axis=1)
    if bad_rows.any():
        df = df.loc[~bad_rows].copy()
        feature_df = feature_df.loc[~bad_rows].copy()
        y = encoder.fit_transform(df["label"].values)
    X       = feature_df.values.astype(np.float32)
    return X, y, encoder


# ─────────────────────────────────────────────────────────────────────────────
# 6. Dataset management utilities
# ─────────────────────────────────────────────────────────────────────────────

def get_dataset_stats(csv_path: str) -> dict:
    """
    Return a summary of how many samples exist per letter in the CSV.

    Parameters
    ----------
    csv_path : str
        Path to the dataset CSV.

    Returns
    -------
    dict
        {letter: count} sorted alphabetically.
        Empty dict if file does not exist.
    """
    if not os.path.isfile(csv_path):
        return {}
    df     = pd.read_csv(csv_path)
    # Filter out accidentally appended header rows
    df = df[df["label"].astype(str).str.lower() != "label"].copy()
    counts = df["label"].value_counts().to_dict()
    return dict(sorted(counts.items()))


def delete_letter_samples(csv_path: str, letters: list, backup: bool = True) -> dict:
    """
    Remove all samples for one or more letters from the dataset CSV.

    Parameters
    ----------
    csv_path : str
        Path to the dataset CSV.
    letters : list of str
        Letters to delete, e.g. ["A", "B"]. Case-insensitive.
    backup : bool
        If True (default), saves a .bak file before modifying so you
        can recover if needed.

    Returns
    -------
    dict
        {"removed": {letter: n}, "remaining": {letter: n},
         "total_before": int, "total_after": int}
    """
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    letters_upper = [l.strip().upper() for l in letters]
    df            = pd.read_csv(csv_path)
    # Filter out accidentally appended header rows
    df = df[df["label"].astype(str).str.lower() != "label"].copy()
    original_len  = len(df)

    removed = {
        letter: int((df["label"] == letter).sum())
        for letter in letters_upper
        if (df["label"] == letter).any()
    }

    df_clean = df[~df["label"].isin(letters_upper)].copy()

    if backup:
        df.to_csv(csv_path + ".bak", index=False)

    df_clean.to_csv(csv_path, index=False)
    remaining = dict(sorted(df_clean["label"].value_counts().to_dict().items()))

    return {
        "removed":      removed,
        "remaining":    remaining,
        "total_before": original_len,
        "total_after":  len(df_clean),
    }


def keep_only_letters(csv_path: str, letters: list, backup: bool = True) -> dict:
    """
    Remove ALL letters from the CSV EXCEPT the ones specified.

    Parameters
    ----------
    csv_path : str
        Path to dataset CSV.
    letters : list of str
        Letters to KEEP.
    backup : bool
        Save a .bak file first (default: True).

    Returns
    -------
    dict
        Same format as delete_letter_samples.
    """
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    df          = pd.read_csv(csv_path)
    # Filter out accidentally appended header rows
    df = df[df["label"].astype(str).str.lower() != "label"].copy()
    keep_upper  = [l.strip().upper() for l in letters]
    to_delete   = [l for l in df["label"].unique() if l not in keep_upper]

    return delete_letter_samples(csv_path, to_delete, backup=backup)
