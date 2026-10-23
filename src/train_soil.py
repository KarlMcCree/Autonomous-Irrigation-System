
import os
import json
import argparse
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger
)

from . import config


# ═══════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════
def load_and_prepare(csv_path, window_size, horizon,
                     val_split, test_split, seed):

    print(f" Loading: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"   Rows: {len(df)}, Columns: {list(df.columns)}")

    # Auto-detect columns
    cols = {c.lower(): c for c in df.columns}
    soil_col = next((cols[k] for k in ["soil_moisture", "soilmoisture",
                                        "moisture", "soil_moisture_%"]
                     if k in cols), None)
    temp_col = next((cols[k] for k in ["temperature", "temp",
                                        "air_temperature", "temp_c"]
                     if k in cols), None)
    hum_col  = next((cols[k] for k in ["humidity", "humid", "rh"]
                     if k in cols), None)

    if soil_col is None:
        raise ValueError(f"No soil column in {list(df.columns)}")

    # Extract features
    soil = df[soil_col].values.astype("float32")
    soil = soil / 100.0 if soil.max() > 1.5 else soil  # normalize to [0,1]

    temp = df[temp_col].values.astype("float32") if temp_col else np.full_like(soil, 25.0)
    humid = df[hum_col].values.astype("float32") if hum_col else np.full_like(soil, 50.0)

    features = np.stack([soil, temp, humid], axis=1)  # (N, 3)
    print(f"   Features: {features.shape}")

    # Build windows
    X, y = [], []
    for i in range(len(features) - window_size - horizon):
        X.append(features[i : i + window_size])
        y.append(soil[i + window_size + horizon])
    X = np.array(X, dtype="float32")
    y = np.array(y, dtype="float32")
    print(f"   Windows: X={X.shape}, y={y.shape}")

    # Split (chronological, not random — time series!)
    n = len(X)
    n_test = int(n * test_split)
    n_val = int(n * val_split)
    n_train = n - n_val - n_test

    return {
        "X_train": X[:n_train],           "y_train": y[:n_train],
        "X_val":   X[n_train:n_train+n_val], "y_val": y[n_train:n_train+n_val],
        "X_test":  X[n_train+n_val:],     "y_test": y[n_train+n_val:],
    }


# ═══════════════════════════════════════════════════════════════
# MODEL
# ═══════════════════════════════════════════════════════════════
def build_model(window_size, n_features, lstm_units, dense_units, dropout):

    model = models.Sequential([
        layers.Input(shape=(window_size, n_features)),
        layers.LSTM(lstm_units, return_sequences=False),
        layers.Dropout(dropout),
        layers.Dense(dense_units, activation="relu"),
        layers.Dropout(dropout),
        layers.Dense(1, activation="sigmoid"),  # soil moisture in [0, 1]
    ], name="SoilPredictor")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(config.LEARNING_RATE),
        loss="mse",
        metrics=["mae"],
    )
    return model


# ═══════════════════════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════════════════════
def train(args):
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    run_name = args.run_name or datetime.now().strftime("soil_%Y%m%d_%H%M%S")
    log_dir = os.path.join(config.LOGS_DIR, run_name)
    os.makedirs(log_dir, exist_ok=True)

    print("=" * 60)
    print(f"🌱 TRAINING SOIL PREDICTOR — {run_name}")
    print("=" * 60)
    for k, v in vars(args).items():
        print(f"   {k:20s}: {v}")
    print("=" * 60)

    with open(os.path.join(log_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2, default=str)

    if args.dry_run:
        print(" Dry-run. Exiting.")
        return

    # Data
    splits = load_and_prepare(
        args.data, args.window_size, args.horizon,
        config.VAL_SPLIT, config.TEST_SPLIT, args.seed
    )
    X_tr, y_tr = splits["X_train"], splits["y_train"]
    X_va, y_va = splits["X_val"],   splits["y_val"]
    X_te, y_te = splits["X_test"],  splits["y_test"]

    # Model
    print(f"\n Building LSTM (units={args.lstm_units})...")
    model = build_model(
        args.window_size, config.N_FEATURES,
        args.lstm_units, config.DENSE_UNITS, config.DROPOUT,
    )
    model.summary()

    # Callbacks
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=args.patience,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1),
        ModelCheckpoint(config.SOIL_MODEL_KERAS, monitor="val_loss",
                        save_best_only=True, verbose=1),
        CSVLogger(os.path.join(log_dir, "history.csv")),
    ]

    # Train
    history = model.fit(
        X_tr, y_tr,
        validation_data=(X_va, y_va),
        epochs=args.epochs,
        batch_size=config.BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
    )

    # Evaluate
    print("\n" + "=" * 60)
    print("TEST EVALUATION")
    print("=" * 60)
    test_loss, test_mae = model.evaluate(X_te, y_te, verbose=0)
    print(f"   Test MSE:  {test_loss:.6f}")
    print(f"   Test MAE:  {test_mae:.6f}  ({test_mae*100:.2f}% moisture)")

    # Save metadata alongside model
    meta = {
        "model_name": "soil_predictor",
        "window_size": args.window_size,
        "horizon": args.horizon,
        "n_features": config.N_FEATURES,
        "feature_names": ["soil_moisture", "temperature", "humidity"],
        "uses_preprocess_input": False,       # our LSTM handles raw values
        "expected_input_range": [0, 100],     # raw sensor scale
        "test_mae": float(test_mae),
        "test_mse": float(test_loss),
        "trained_at": datetime.now().isoformat(),
    }
    with open(config.SOIL_MODEL_META, "w") as f:
        json.dump(meta, f, indent=2)
    print(f" Metadata: {config.SOIL_MODEL_META}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(history.history["loss"], label="Train")
    axes[0].plot(history.history["val_loss"], label="Val")
    axes[0].set_title("MSE Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(history.history["mae"], label="Train")
    axes[1].plot(history.history["val_mae"], label="Val")
    axes[1].set_title("MAE")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(config.DOCS_DIR, "soil_training_history.png"), dpi=120)
    print(f" Plot: docs/soil_training_history.png")

    # Prediction quality plot
    y_pred = model.predict(X_te, verbose=0).flatten()
    plt.figure(figsize=(14, 4))
    plt.plot(y_te[:300], label="Actual", alpha=0.8)
    plt.plot(y_pred[:300], label="Predicted", alpha=0.8)
    plt.title("Test Set — Actual vs Predicted Soil Moisture")
    plt.xlabel("Sample")
    plt.ylabel("Soil Moisture [0,1]")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(config.DOCS_DIR, "soil_prediction_quality.png"), dpi=120)
    print(f" Plot: docs/soil_prediction_quality.png")

    print(f"\n Model saved: {config.SOIL_MODEL_KERAS}")


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("--data", default=config.SIMULATION_CSV)
    p.add_argument("--window-size", type=int, default=config.WINDOW_SIZE)
    p.add_argument("--horizon", type=int, default=config.PREDICT_HORIZON)
    p.add_argument("--lstm-units", type=int, default=config.LSTM_UNITS)
    p.add_argument("--epochs", type=int, default=config.EPOCHS)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    p.add_argument("--run-name", default=None)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())