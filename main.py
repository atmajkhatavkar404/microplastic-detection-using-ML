# ==========================================
# MICROPLASTIC DETECTION - RESEARCH GRADE
# MobileNetV2 + Fine Tuning
# TensorFlow 2.16+ / Keras 3
# Local Linux / Parrot OS
# ==========================================

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint
)
from tensorflow.keras.utils import load_img, img_to_array


# ================================
# CONFIGURATION
# ================================
DATASET_PATH = "/home/parrot/Downloads/project"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
VALID_LABELS = {'a', 'b', 'c', 'd', 'f'}
EPOCHS_STAGE_1 = 10
EPOCHS_STAGE_2 = 10
MODEL_SAVE_PATH = "microplastic_best_model.keras"
CLASS_MAP_PATH = "class_mapping.json"


# ================================
# CREATE DATAFRAME
# ================================
def create_df(folder):
    data = []

    if not os.path.exists(folder):
        print(f"❌ Folder not found: {folder}")
        return pd.DataFrame()

    for file_name in os.listdir(folder):
        if file_name.lower().endswith((".jpg", ".jpeg", ".png")):
            parts = file_name.split("--")
            if len(parts) < 2:
                continue

            label = parts[0].lower().strip()

            if label in VALID_LABELS:
                data.append({
                    "filename": os.path.join(folder, file_name),
                    "label": label
                })

    return pd.DataFrame(data)


# ================================
# BUILD MODEL
# ================================
def build_model(num_classes):
    base_model = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(224, 224, 3)
    )

    # Stage 1: freeze backbone
    base_model.trainable = False

    inputs = layers.Input(shape=(224, 224, 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model, base_model


# ================================
# DATA GENERATORS
# ================================
def build_generators():
    train_dir = os.path.join(DATASET_PATH, "train")
    valid_dir = os.path.join(DATASET_PATH, "valid")

    train_df = create_df(train_dir)
    valid_df = create_df(valid_dir)

    if train_df.empty or valid_df.empty:
        print("❌ Dataset empty or wrong structure")
        return None, None, None, None

    print(f"✅ Train Images: {len(train_df)}")
    print(f"✅ Valid Images: {len(valid_df)}")

    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=35,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.25,
        horizontal_flip=True,
        brightness_range=[0.7, 1.3],
        fill_mode="nearest"
    )

    valid_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input
    )

    train_gen = train_datagen.flow_from_dataframe(
        train_df,
        x_col="filename",
        y_col="label",
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=True
    )

    valid_gen = valid_datagen.flow_from_dataframe(
        valid_df,
        x_col="filename",
        y_col="label",
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False
    )

    with open(CLASS_MAP_PATH, "w") as f:
        json.dump(train_gen.class_indices, f)

    return train_gen, valid_gen, train_df, valid_df


# ================================
# TRAIN MODEL
# ================================
def build_and_train():
    train_gen, valid_gen, _, _ = build_generators()

    if train_gen is None:
        return None, None, None

    model, base_model = build_model(len(train_gen.class_indices))

    callbacks = [
        EarlyStopping(
            monitor="val_accuracy",
            patience=5,
            restore_best_weights=True
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            patience=2,
            factor=0.5
        ),
        ModelCheckpoint(
            MODEL_SAVE_PATH,
            monitor="val_accuracy",
            save_best_only=True
        )
    ]

    print("\n🚀 Stage 1 Training (Frozen Backbone)...\n")
    history1 = model.fit(
        train_gen,
        validation_data=valid_gen,
        epochs=EPOCHS_STAGE_1,
        callbacks=callbacks
    )

    print("\n🚀 Stage 2 Fine-Tuning...\n")

    base_model.trainable = True

    # Fine-tune only top layers
    for layer in base_model.layers[:-40]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    history2 = model.fit(
        train_gen,
        validation_data=valid_gen,
        epochs=EPOCHS_STAGE_2,
        callbacks=callbacks
    )

    return model, train_gen, (history1, history2)


# ================================
# PLOT HISTORY
# ================================
def plot_history(histories):
    history1, history2 = histories

    acc = history1.history["accuracy"] + history2.history["accuracy"]
    val_acc = history1.history["val_accuracy"] + history2.history["val_accuracy"]

    plt.figure(figsize=(10, 5))
    plt.plot(acc, label="Train Accuracy")
    plt.plot(val_acc, label="Validation Accuracy")
    plt.title("Microplastic Classification Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.show()


# ================================
# PREDICTION UI
# ================================
def run_prediction_ui(model, train_gen):
    idx_to_class = {
        v: k for k, v in train_gen.class_indices.items()
    }

    while True:
        image_path = input("\nEnter image path (or 'exit'): ").strip()

        if image_path.lower() == "exit":
            print("👋 Exiting...")
            break

        if not os.path.exists(image_path):
            print("❌ File not found")
            continue

        try:
            img = load_img(image_path, target_size=IMG_SIZE)
            arr = img_to_array(img)
            arr = preprocess_input(arr)
            arr = np.expand_dims(arr, axis=0)

            preds = model.predict(arr, verbose=0)[0]

            top3_idx = np.argsort(preds)[-3:][::-1]

            print(f"\n📷 File: {image_path}")
            print("🔍 Top Predictions:")

            for rank, idx in enumerate(top3_idx, 1):
                label = idx_to_class[idx]
                conf = preds[idx]
                print(f"{rank}. {label} → {conf:.2%}")

        except Exception as e:
            print(f"❌ Prediction failed: {e}")


# ================================
# MAIN
# ================================
if __name__ == "__main__":
    model, train_gen, histories = build_and_train()

    if model is not None:
        plot_history(histories)
        run_prediction_ui(model, train_gen)
