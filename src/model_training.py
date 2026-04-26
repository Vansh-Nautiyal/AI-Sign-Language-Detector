"""
train_model.py — ASL MLP Classifier Training
=============================================
Loads landmark CSV, trains a lightweight Dense Neural Network,
and saves the model to model/asl_model.h5.

Usage:
  python src/train_model.py
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                  # headless — no display needed
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

from utils import load_dataset_from_csv, ASLLabelEncoder   # ← from utils.py

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(__file__))
CSV_PATH   = os.path.join(BASE_DIR, "data",  "dataset.csv")
MODEL_DIR  = os.path.join(BASE_DIR, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "asl_model.h5")
PLOT_PATH  = os.path.join(MODEL_DIR, "training_history.png")
os.makedirs(MODEL_DIR, exist_ok=True)

# ── Hyper-parameters ──────────────────────────────────────────────────────────
TEST_SIZE   = 0.20
RANDOM_SEED = 42
EPOCHS      = 50
BATCH_SIZE  = 32


# ── Data loading ──────────────────────────────────────────────────────────────
def load_dataset(path: str):
    """Delegates to utils.load_dataset_from_csv for DRY data loading."""
    X, y, encoder = load_dataset_from_csv(path)
    class_names   = encoder.class_names
    print(f"Loaded {len(X):,} samples  |  {encoder.num_classes} classes: {class_names}")
    print(f"Feature shape: {X.shape}")
    return X, y, class_names, encoder


# ── Model definition ──────────────────────────────────────────────────────────
def build_model(input_dim: int, num_classes: int) -> tf.keras.Model:
    """Lightweight MLP suitable for real-time inference."""
    model = models.Sequential(
        [
            layers.Input(shape=(input_dim,)),
            layers.Dense(128, activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(64, activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(num_classes, activation="softmax"),
        ],
        name="ASL_MLP",
    )
    model.compile(
        optimizer="adam",
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
        monitor="val_accuracy", patience=10, restore_best_weights=True
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
    print("\n=== ASL MLP Training ===\n")
    X, y, class_names, encoder = load_dataset(CSV_PATH)
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
