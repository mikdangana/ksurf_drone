#!/usr/bin/env python3

import argparse
import os

import numpy as np
import pandas as pd
import tensorflow as tf

from tcn import TCN, tcn_full_summary

def create_dataset(series, window_size):
    """
    Create (X, y) samples for a univariate time series.
    Each X is 'window_size' points, the y is the next point.
    """
    X, y = [], []
    for i in range(len(series) - window_size):
        X.append(series[i : i + window_size])
        y.append(series[i + window_size])
    return np.array(X), np.array(y)

def build_tcn_model(window_size):
    """
    Build a simple TCN-based regression model using keras-tcn.
    """
    inputs = tf.keras.Input(shape=(window_size, 1))  # univariate => 1 channel
    x = TCN(
        nb_filters=32,          # Number of convolutional filters
        kernel_size=3,          # Size of the convolutional kernel
        nb_stacks=1,            # Number of stacks of residual blocks
        dilations=[1, 2, 4, 8], # Dilation rates
        padding='causal',
        use_skip_connections=True,
        dropout_rate=0.0,       # adjust dropout if needed
        return_sequences=False  # Return last output (for regression)
    )(inputs)
    outputs = tf.keras.layers.Dense(1)(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='mse',
        metrics=['mse']
    )
    return model

def test(csv_file, column_name, model_path,
         window_size=20, test_ratio=0.2, epochs=10, runs=5, verbose=1):
    """
    1) Load data from csv_file's specified column_name.
    2) Create a sliding-window dataset for TCN.
    3) Train or load the TCN model from disk.
    4) Evaluate MSE on test set, print the MSE.
    """

    # --- 1) Load CSV and extract the column
    df = pd.read_csv(csv_file)
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in CSV.")

    # Convert to a numpy array (float)
    series = df[column_name].values.astype(np.float32)

    # --- 2) Prepare (X, y) with a sliding window
    X_all, y_all = create_dataset(series, window_size)
    # Reshape X to (samples, window_size, 1) for univariate TCN
    X_all = np.expand_dims(X_all, axis=-1)  # shape: (samples, window_size, 1)

    # Train/test split
    n = len(X_all)
    split_index = int(n * (1 - test_ratio))
    X_train, X_test = X_all[:split_index], X_all[split_index:]
    y_train, y_test = y_all[:split_index], y_all[split_index:]

    # --- 3) Check if model file exists; if not, train and save.
    if os.path.exists(model_path):
        print(f"Loading existing TCN model from '{model_path}'...")
        # When loading a model that contains custom layers (like TCN),
        # specify custom_objects to let Keras know how to reconstruct the layer.
        model = tf.keras.models.load_model(model_path, custom_objects={'TCN': TCN})
    else:
        print("Model not found. Building and training a new TCN model...")
        model = build_tcn_model(window_size)

        # Train
        model.fit(
            X_train, y_train,
            validation_split=0.1,
            epochs=epochs,
            batch_size=32,
            verbose=verbose
        )

        # Save
        print(f"Saving model to '{model_path}'...")
        model.save(model_path)

    # --- 4) Evaluate on test set
    loss, mse = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test MSE = {mse:.6f}")

    # 2) Get predictions
    y_pred = model.predict(X_test)

    # 3) Compute residuals = prediction - ground truth
    residuals = y_pred.flatten() - y_test.flatten()

    # 4) Standard deviation of the residuals
    residual_std = np.std(residuals, ddof=1)  # ddof=1 for sample std
    print("Standard Deviation of residuals:", residual_std)
    return mse


def main(csv_file, column_name, model_path,
         window_size=20, test_ratio=0.2, epochs=10, runs=5, verbose=1):
    mses = [test(csv_file, column_name, model_path, window_size, test_ratio, epochs, runs, verbose) for run in range(runs)]
    mses = np.array(mses)
    print(f"MSE mean = {np.mean(mses)}, std = {np.std(mses)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Time Series Prediction with TCN (save/load model).")
    parser.add_argument("csv_file", type=str, help="Path to the CSV file.")
    parser.add_argument("column_name", type=str, help="Column name in the CSV for the time series.")
    parser.add_argument("--model_path", type=str, default="my_tcn_model.h5",
                        help="Filepath to load/save the trained model.")
    parser.add_argument("--window_size", type=int, default=20, help="Sliding window size.")
    parser.add_argument("--test_ratio", type=float, default=0.2, help="Fraction of data used for testing.")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs.")
    parser.add_argument("--runs", type=int, default=1, help="Runs.")
    parser.add_argument("--verbose", type=int, default=1, help="Verbosity level (0 or 1).")
    args = parser.parse_args()

    main(
        csv_file=args.csv_file,
        column_name=args.column_name,
        model_path=args.model_path,
        window_size=args.window_size,
        test_ratio=args.test_ratio,
        epochs=args.epochs,
        runs=args.runs,
        verbose=args.verbose
    )

