"""
train_model.py — ASL MLP Classifier Training
=============================================
Loads landmark CSV, trains a Dense Neural Network (MLP),
and saves the model to model/asl_model.h5.

Usage:
  python src/train_model.py                   # train on data/dataset.csv
  python src/train_model.py --mix             # merge dataset.csv + webcam.csv then train
  python src/train_model.py --webcam-only     # train only on webcam-collected samples

The --mix flag is the recommended workflow after collecting your own
webcam samples with collect_data.py. It combines the base synthetic
dataset with your real hand data for best real-world accuracy.

Webcam data (collect_data.py output) is auto-saved to data/dataset.csv.
If you want to keep synthetic and real data separate, save webcam data
to a different CSV and point --webcam to it:
  python src/train_model.py --mix --webcam data/my_hand.csv
"""

import argparse
import os
import sys
import tempfile
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                  # headless — no display needed
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

from utils import CSV_HEADER, load_dataset_from_csv, ASLLabelEncoder   # ← from utils.py

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(__file__))
CSV_PATH        = os.path.join(BASE_DIR, "data", "dataset.csv")
WEBCAM_CSV_PATH = os.path.join(BASE_DIR, "data", "webcam.csv")
MODEL_DIR  = os.path.join(BASE_DIR, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "asl_model.h5")
PLOT_PATH  = os.path.join(MODEL_DIR, "training_history.png")
os.makedirs(MODEL_DIR, exist_ok=True)

# ── Hyper-parameters ──────────────────────────────────────────────────────────
TEST_SIZE   = 0.20
RANDOM_SEED = 42
EPOCHS      = 100    
BATCH_SIZE  = 32


# ── Data loading ──────────────────────────────────────────────────────────────
def _read_landmark_csv(path: str, label: str) -> pd.DataFrame:
    """Read a landmark CSV and remove accidental repeated header rows."""
    df = pd.read_csv(path)
    if "label" not in df.columns and len(df.columns) == len(CSV_HEADER):
        df = pd.read_csv(path, header=None, names=CSV_HEADER)

    before = len(df)
    df = df[df["label"].astype(str).str.lower() != "label"].copy()
    df = df.dropna(subset=["label"])

    feature_cols = [col for col in CSV_HEADER if col != "label"]
    missing = [col for col in feature_cols if col not in df.columns]
    if missing:
        raise ValueError(
            f"{label} is missing required columns: {missing[:5]}"
            f"{'...' if len(missing) > 5 else ''}"
        )

    features = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    bad_rows = features.isna().any(axis=1)
    if bad_rows.any():
        df = df.loc[~bad_rows].copy()
        features = features.loc[~bad_rows].copy()

    removed = before - len(df)
    if removed:
        print(f"{label} cleanup: skipped {removed:,} invalid/header rows")

    df[feature_cols] = features
    return df[CSV_HEADER]


def load_dataset(path: str, webcam_path: str | None = None):
    """
    Load dataset CSV. If webcam_path is given and exists, merge both
    CSVs before training so the model learns from both synthetic and
    real webcam samples.

    Webcam rows are duplicated 3x to give real data higher weight
    over synthetic data — this is the key to closing the gap between
    test accuracy and real-world accuracy.
    """
    df_base = _read_landmark_csv(path, "Base dataset")
    print(f"Base dataset   : {len(df_base):,} samples")

    if webcam_path and os.path.isfile(webcam_path):
        df_webcam = _read_landmark_csv(webcam_path, "Webcam dataset")
        print(f"Webcam dataset : {len(df_webcam):,} samples  (weighted x3)")
        # Repeat webcam rows 3× so real hand data dominates
        df_webcam_weighted = pd.concat([df_webcam] * 3, ignore_index=True)
        df_combined = pd.concat([df_base, df_webcam_weighted], ignore_index=True)
        print(f"Combined total : {len(df_combined):,} samples")
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix="_combined_dataset.csv", delete=False, newline=""
        )
        tmp.close()
        df_combined.to_csv(tmp.name, index=False)
        path = tmp.name
    elif webcam_path:
        print(f"⚠  Webcam CSV not found: {webcam_path} — using base dataset only")

    X, y, encoder = load_dataset_from_csv(path)
    class_names   = encoder.class_names
    print(f"\nFinal: {len(X):,} samples  |  {encoder.num_classes} classes: {class_names}")
    print(f"Feature shape: {X.shape}")
    return X, y, class_names, encoder


# ── Model definition ──────────────────────────────────────────────────────────
def build_model(input_dim: int, num_classes: int) -> tf.keras.Model:
    """
    Improved MLP — extra hidden layer + L2 regularisation to reduce
    confusion between similar signs (U/V, A/E/S, M/N etc.).
    Still lightweight enough for real-time CPU inference (~2 ms).
    """
    reg = tf.keras.regularizers.l2(1e-4)

    model = models.Sequential(
        [
            layers.Input(shape=(input_dim,)),

            # Layer 1 — broad feature extraction
            layers.Dense(256, activation="relu", kernel_regularizer=reg),
            layers.BatchNormalization(),
            layers.Dropout(0.4),

            # Layer 2 — pattern combination
            layers.Dense(128, activation="relu", kernel_regularizer=reg),
            layers.BatchNormalization(),
            layers.Dropout(0.3),

            # Layer 3 — fine discrimination between similar signs
            layers.Dense(64, activation="relu", kernel_regularizer=reg),
            layers.BatchNormalization(),
            layers.Dropout(0.2),

            layers.Dense(num_classes, activation="softmax"),
        ],
        name="ASL_MLP_v2",
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ── Training ──────────────────────────────────────────────────────────────────
def train(X, y, class_names):
    num_classes = len(class_names)

    # One-hot encode targets
    y_cat = tf.keras.utils.to_categorical(y, num_classes)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_cat, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )
    print(f"\nTrain samples : {len(X_train)}")
    print(f"Test  samples : {len(X_test)}\n")

    model = build_model(X.shape[1], num_classes)
    model.summary()

    # Callbacks
    early_stop = callbacks.EarlyStopping(
        monitor="val_accuracy", patience=15, restore_best_weights=True
    )
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=5, min_lr=1e-5, verbose=1
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stop, reduce_lr],
        verbose=1,
    )

    return model, history, X_test, y_test


# ── Evaluation ────────────────────────────────────────────────────────────────
def evaluate(model, history, X_test, y_test, class_names):
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n{'='*50}")
    print(f"  Test Accuracy : {acc*100:.2f}%")
    print(f"  Test Loss     : {loss:.4f}")
    print(f"{'='*50}\n")

    # Per-class report
    y_pred  = np.argmax(model.predict(X_test, verbose=0), axis=1)
    y_true  = np.argmax(y_test, axis=1)
    present = sorted(set(y_true))
    names   = [class_names[i] for i in present]
    print(classification_report(y_true, y_pred, labels=present, target_names=names))

    # Plot training curves
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, metric, title in zip(
        axes,
        [("accuracy", "val_accuracy"), ("loss", "val_loss")],
        ["Accuracy", "Loss"],
    ):
        ax.plot(history.history[metric[0]], label="Train")
        ax.plot(history.history[metric[1]], label="Val")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.legend()
    plt.tight_layout()
    plt.savefig(PLOT_PATH)
    print(f"Training curves saved to: {PLOT_PATH}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Train the ASL MLP classifier")
    parser.add_argument(
        "--dataset",
        default=CSV_PATH,
        help=f"Base dataset CSV path (default: {CSV_PATH})",
    )
    parser.add_argument(
        "--webcam",
        default=WEBCAM_CSV_PATH,
        help=f"Webcam dataset CSV path (default: {WEBCAM_CSV_PATH})",
    )
    parser.add_argument(
        "--mix",
        action="store_true",
        help="Train on dataset.csv plus webcam.csv weighted 3x",
    )
    parser.add_argument(
        "--webcam-only",
        action="store_true",
        help="Train only on webcam-collected samples",
    )
    args = parser.parse_args()

    print("\n=== ASL MLP Training ===\n")
    if args.webcam_only:
        X, y, class_names, encoder = load_dataset(args.webcam)
    else:
        webcam_path = args.webcam if args.mix else None
        X, y, class_names, encoder = load_dataset(args.dataset, webcam_path)

    model, history, X_test, y_test = train(X, y, class_names)
    evaluate(model, history, X_test, y_test, class_names)

    # Save model
    model.save(MODEL_PATH)
    print(f"\nModel saved to: {MODEL_PATH}")

    # Save class names via utils encoder
    label_path = os.path.join(MODEL_DIR, "class_names.txt")
    encoder.save(label_path)
    print(f"Class names  saved to: {label_path}\n")


if __name__ == "__main__":
    main()
