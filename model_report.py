"""
model_report.py — Model Performance Report Generator
======================================================
Generates a complete performance report for the trained ASL MLP model.
Called by streamlit_app.py when the user opens the Model Report tab.

Produces:
  - Overall accuracy and loss on the 20% test split
  - Per-letter precision, recall, F1-score, support
  - Confusion matrix as a DataFrame
  - Training history image path (if model/training_history.png exists)
  - Dataset sample counts per letter

All functions return plain DataFrames / dicts so Streamlit
can render them freely without any coupling to the UI.

Usage (CLI):
    python model_report.py
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
SRC_DIR  = BASE_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

from utils import load_dataset_from_csv   # noqa: E402

MODEL_PATH  = str(BASE_DIR / "model" / "asl_model.h5")
LABEL_PATH  = str(BASE_DIR / "model" / "class_names.txt")
CSV_PATH    = str(BASE_DIR / "data"  / "dataset.csv")
HISTORY_IMG = str(BASE_DIR / "model" / "training_history.png")

TEST_SIZE   = 0.20
RANDOM_SEED = 42


@dataclass
class ModelReport:
    test_accuracy:    float         = 0.0
    test_loss:        float         = 0.0
    total_params:     int           = 0
    num_classes:      int           = 0
    train_samples:    int           = 0
    test_samples:     int           = 0
    per_letter_df:    pd.DataFrame  = field(default_factory=pd.DataFrame)
    confusion_df:     pd.DataFrame  = field(default_factory=pd.DataFrame)
    dataset_df:       pd.DataFrame  = field(default_factory=pd.DataFrame)
    history_img_path: str | None    = None
    error:            str           = ""


def generate_report() -> ModelReport:
    report = ModelReport()

    missing = []
    if not os.path.isfile(MODEL_PATH):
        missing.append(f"Model not found: {MODEL_PATH}")
    if not os.path.isfile(LABEL_PATH):
        missing.append(f"Class names missing: {LABEL_PATH}")
    if not os.path.isfile(CSV_PATH):
        missing.append(f"Dataset not found: {CSV_PATH}")
    if missing:
        report.error = (
            "Cannot generate report. Run these first:\n\n"
            + "\n".join(f"  • {m}" for m in missing)
            + "\n\n  → python import_dataset.py\n  → python src/train_model.py"
        )
        return report

    try:
        import tensorflow as tf
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import classification_report, confusion_matrix

        X, y, encoder = load_dataset_from_csv(CSV_PATH)
        class_names   = encoder.class_names
        num_classes   = len(class_names)
        y_cat         = tf.keras.utils.to_categorical(y, num_classes)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y_cat,
            test_size=TEST_SIZE,
            random_state=RANDOM_SEED,
            stratify=y,
        )

        report.train_samples = len(X_train)
        report.test_samples  = len(X_test)
        report.num_classes   = num_classes

        model     = tf.keras.models.load_model(MODEL_PATH)
        loss, acc = model.evaluate(X_test, y_test, verbose=0)

        report.test_accuracy = float(acc)
        report.test_loss     = float(loss)
        report.total_params  = int(model.count_params())

        y_pred  = np.argmax(model.predict(X_test, verbose=0), axis=1)
        y_true  = np.argmax(y_test, axis=1)
        present = sorted(set(y_true))
        names   = [class_names[i] for i in present]

        report_dict = classification_report(
            y_true, y_pred,
            labels=present,
            target_names=names,
            output_dict=True,
            zero_division=0,
        )

        rows = []
        for name in names:
            m = report_dict[name]
            rows.append({
                "Letter":    name,
                "Precision": round(float(m["precision"]), 3),
                "Recall":    round(float(m["recall"]),    3),
                "F1 score":  round(float(m["f1-score"]),  3),
                "Support":   int(m["support"]),
            })
        report.per_letter_df = pd.DataFrame(rows)

        cm = confusion_matrix(y_true, y_pred, labels=present)
        report.confusion_df  = pd.DataFrame(cm, index=names, columns=names)

        full_df   = pd.read_csv(CSV_PATH)
        # Filter out accidentally appended header rows (where label="label")
        full_df = full_df[full_df["label"].astype(str).str.lower() != "label"].copy()
        label_cts = full_df["label"].value_counts().sort_index()
        report.dataset_df = pd.DataFrame({
            "Letter":  label_cts.index.tolist(),
            "Samples": label_cts.values.tolist(),
        })

        if os.path.isfile(HISTORY_IMG):
            report.history_img_path = HISTORY_IMG

    except Exception as exc:
        report.error = f"Report generation failed:\n\n{exc}"

    return report


if __name__ == "__main__":
    r = generate_report()
    if r.error:
        print(f"\nERROR:\n{r.error}\n")
        sys.exit(1)
    print(f"\n{'='*52}")
    print(f"  Test accuracy  :  {r.test_accuracy*100:.2f}%")
    print(f"  Test loss      :  {r.test_loss:.4f}")
    print(f"  Total params   :  {r.total_params:,}")
    print(f"  Train samples  :  {r.train_samples:,}")
    print(f"  Test  samples  :  {r.test_samples:,}")
    print(f"  Classes        :  {r.num_classes}")
    print(f"{'='*52}\n")
    print(r.per_letter_df.to_string(index=False))