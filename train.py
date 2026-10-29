"""
CNN Training and Evaluation Script for Handwritten Digit Recognition
Author: Rohini S (CMR University)

Trains a deep Convolutional Neural Network (CNN) on the MNIST dataset
incorporating data augmentation, batch normalization, dropout regularization,
early stopping, and adaptive learning rate decay.
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


def build_model():
    """
    Constructs the CNN architecture with integrated data augmentation.
    Total parameters: ~242,442
    """
    data_augmentation = tf.keras.Sequential([
        layers.RandomRotation(0.08),
        layers.RandomZoom(0.10),
        layers.RandomTranslation(0.10, 0.10)
    ], name="data_augmentation")

    model = models.Sequential([
        layers.Input(shape=(28, 28, 1)),
        data_augmentation,

        # Block 1: 32 Filters
        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # Block 2: 64 Filters
        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # Block 3: 128 Filters
        layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),

        # Classification Head
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.40),
        layers.Dense(10, activation="softmax")
    ], name="mnist_cnn")

    return model


def main():
    parser = argparse.ArgumentParser(description="Train CNN on MNIST dataset")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs (default: 10)")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size (default: 64)")
    parser.add_argument("--lr", type=float, default=0.001, help="Initial learning rate (default: 0.001)")
    parser.add_argument("--output_model", type=str, default="models/digit_recognition_cnn.keras",
                        help="Destination path for trained model")
    args = parser.parse_args()

    print("=" * 60)
    print("HANDWRITTEN DIGIT RECOGNITION - MODEL TRAINING")
    print("=" * 60)
    print(f"TensorFlow Version: {tf.__version__}")
    print(f"Target Output Path: {args.output_model}")

    # 1. Load Dataset
    print("\n[1/5] Loading MNIST dataset...")
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    print(f"-> Train samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")

    # 2. Preprocess & Reshape
    print("\n[2/5] Normalizing pixel intensities to [0, 1] & adding channel dimension...")
    X_train = (X_train.astype("float32") / 255.0).reshape(-1, 28, 28, 1)
    X_test = (X_test.astype("float32") / 255.0).reshape(-1, 28, 28, 1)

    # 3. Build & Compile Model
    print("\n[3/5] Building CNN Architecture...")
    model = build_model()
    model.summary()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-5,
            verbose=1
        )
    ]

    # 4. Train Model
    print(f"\n[4/5] Training for up to {args.epochs} epochs with batch size {args.batch_size}...")
    history = model.fit(
        X_train,
        y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_split=0.1,
        callbacks=callbacks,
        verbose=1
    )

    # 5. Evaluate on Unseen Test Set
    print("\n[5/5] Evaluating on unseen test set (10,000 images)...")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"-> Test Accuracy: {test_acc * 100:.2f}%")
    print(f"-> Test Loss:     {test_loss:.4f}")

    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, digits=4))

    # Save Confusion Matrix Plot
    os.makedirs(os.path.dirname(args.output_model) or ".", exist_ok=True)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title("MNIST Confusion Matrix - Rohini S")
    plt.xlabel("Predicted Digit")
    plt.ylabel("Ground Truth Digit")
    plt.tight_layout()
    cm_path = "confusion_matrix.png"
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"-> Saved confusion matrix plot to {cm_path}")

    # Save Model Weights
    model.save(args.output_model)
    print(f"-> Model successfully saved to {args.output_model} ✅")
    print("=" * 60)


if __name__ == "__main__":
    main()
