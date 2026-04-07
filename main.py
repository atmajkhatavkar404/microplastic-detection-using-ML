# ==========================================
# MICROPLASTIC DETECTION - SMART TRAIN/LOAD + BEAUTIFUL GUI
# MobileNetV2 + Fine Tuning
# TensorFlow 2.16+ / Keras 3
# Local Linux / Parrot OS
# Enhanced Tkinter UI
# ==========================================

import os
import json
import threading
import pandas as pd
import numpy as np
import tensorflow as tf

from tkinter import Tk, Label, Button, Text, filedialog, END, Frame
from PIL import Image, ImageTk

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras import layers
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint,
)
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.models import load_model

# ================================
# CONFIGURATION
# ================================
DATASET_PATH = "/home/parrot/Downloads/project"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
VALID_LABELS = {"a", "b", "c", "d", "f"}
EPOCHS_STAGE_1 = 10
EPOCHS_STAGE_2 = 10
MODEL_SAVE_PATH = "microplastic_best_model.keras"
CLASS_MAP_PATH = "class_mapping.json"

# UI COLORS
BG_COLOR = "#F8F6FF"
CARD_COLOR = "#FFFFFF"
PRIMARY = "#7C3AED"
SECONDARY = "#EC4899"
TEXT_COLOR = "#1F2937"
ACCENT = "#DDD6FE"

model = None
idx_to_class = None
current_image_path = None
preview_photo = None


def create_df(folder):
    data = []
    if not os.path.exists(folder):
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
                    "label": label,
                })
    return pd.DataFrame(data)


def build_generators():
    train_dir = os.path.join(DATASET_PATH, "train")
    valid_dir = os.path.join(DATASET_PATH, "valid")

    train_df = create_df(train_dir)
    valid_df = create_df(valid_dir)

    if train_df.empty or valid_df.empty:
        return None, None

    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=35,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.25,
        horizontal_flip=True,
        brightness_range=[0.7, 1.3],
        fill_mode="nearest",
    )

    valid_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    train_gen = train_datagen.flow_from_dataframe(
        train_df, x_col="filename", y_col="label",
        target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode="categorical", shuffle=True,
    )

    valid_gen = valid_datagen.flow_from_dataframe(
        valid_df, x_col="filename", y_col="label",
        target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode="categorical", shuffle=False,
    )

    with open(CLASS_MAP_PATH, "w") as f:
        json.dump(train_gen.class_indices, f)

    return train_gen, valid_gen


def build_model(num_classes):
    base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False

    inputs = layers.Input(shape=(224, 224, 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    built_model = tf.keras.Model(inputs, outputs)
    built_model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return built_model, base_model


def train_new_model():
    train_gen, valid_gen = build_generators()
    if train_gen is None:
        return None, None

    built_model, base_model = build_model(len(train_gen.class_indices))

    callbacks = [
        EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", patience=2, factor=0.5),
        ModelCheckpoint(MODEL_SAVE_PATH, monitor="val_accuracy", save_best_only=True),
    ]

    built_model.fit(train_gen, validation_data=valid_gen, epochs=EPOCHS_STAGE_1, callbacks=callbacks, verbose=1)

    base_model.trainable = True
    for layer in base_model.layers[:-40]:
        layer.trainable = False

    built_model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    built_model.fit(train_gen, validation_data=valid_gen, epochs=EPOCHS_STAGE_2, callbacks=callbacks, verbose=1)
    idx_map = {v: k for k, v in train_gen.class_indices.items()}
    return built_model, idx_map


def load_saved_model_and_mapping():
    if os.path.exists(MODEL_SAVE_PATH) and os.path.exists(CLASS_MAP_PATH):
        loaded_model = load_model(MODEL_SAVE_PATH)
        with open(CLASS_MAP_PATH, "r") as f:
            class_indices = json.load(f)
        idx_map = {int(v): k for k, v in class_indices.items()}
        return loaded_model, idx_map
    return None, None


def append_output(text):
    output_box.insert(END, text + "\n")
    output_box.see(END)


def animate_status(message, dots=0):
    if dots < 4:
        status_label.config(text=message + "." * dots)
        root.after(400, lambda: animate_status(message, dots + 1))
    else:
        status_label.config(text=message)


def initialize_model():
    global model, idx_to_class
    animate_status("Initializing model")
    append_output("Checking for saved model...")
    model, idx_to_class = load_saved_model_and_mapping()

    if model is not None:
        append_output("Loaded pretrained model successfully.")
        status_label.config(text="Model Ready")
        return

    append_output("No saved model found. Training started...")
    model, idx_to_class = train_new_model()

    if model is not None:
        append_output("Training complete and model saved.")
        status_label.config(text="Training Complete")
    else:
        append_output("Training failed. Check dataset structure.")
        status_label.config(text="Training Failed")


def start_model_init_thread():
    threading.Thread(target=initialize_model, daemon=True).start()


def upload_image():
    global current_image_path, preview_photo
    file_path = filedialog.askopenfilename(
        title="Select Microplastic Image",
        filetypes=[("Image Files", "*.jpg *.jpeg *.png")],
    )
    if not file_path:
        return

    current_image_path = file_path
    pil_img = Image.open(file_path).convert("RGB")
    pil_img.thumbnail((320, 320))
    preview_photo = ImageTk.PhotoImage(pil_img)
    image_label.config(image=preview_photo, bg=CARD_COLOR)
    append_output(f"Selected image: {file_path}")
    status_label.config(text="Image Uploaded")


def predict_uploaded_image():
    if model is None:
        append_output("Model not ready yet.")
        return
    if current_image_path is None:
        append_output("Please upload an image first.")
        return

    status_label.config(text="Predicting...")

    try:
        img = load_img(current_image_path, target_size=IMG_SIZE)
        arr = img_to_array(img)
        arr = preprocess_input(arr)
        arr = np.expand_dims(arr, axis=0)

        preds = model.predict(arr, verbose=0)[0]
        top3_idx = np.argsort(preds)[-3:][::-1]

        append_output("\n✨ Top Predictions:")
        for rank, idx in enumerate(top3_idx, 1):
            label = idx_to_class[idx]
            conf = preds[idx]
            append_output(f"{rank}. {label} → {conf:.2%}")

        status_label.config(text="Prediction Complete")

    except Exception as e:
        append_output(f"Prediction failed: {e}")
        status_label.config(text="Prediction Failed")


# ================================
# BEAUTIFUL GUI
# ================================
root = Tk()
root.title("✨ Microplastic Detection AI")
root.geometry("980x760")
root.configure(bg=BG_COLOR)

header = Label(
    root,
    text="✨ Microplastic Detection AI",
    font=("Arial", 24, "bold"),
    bg=BG_COLOR,
    fg=PRIMARY,
)
header.pack(pady=15)

status_label = Label(
    root,
    text="Ready",
    font=("Arial", 12, "bold"),
    bg=BG_COLOR,
    fg=SECONDARY,
)
status_label.pack(pady=5)

btn_frame = Frame(root, bg=BG_COLOR)
btn_frame.pack(pady=15)

for i, (text, cmd) in enumerate([
    ("Initialize Model", start_model_init_thread),
    ("Upload Image", upload_image),
    ("Predict", predict_uploaded_image),
]):
    Button(
        btn_frame,
        text=text,
        command=cmd,
        width=18,
        font=("Arial", 12, "bold"),
        bg=PRIMARY,
        fg="white",
        activebackground=SECONDARY,
        relief="flat",
        bd=0,
        padx=10,
        pady=8,
        cursor="hand2",
    ).grid(row=0, column=i, padx=12)

image_frame = Frame(root, bg=CARD_COLOR, bd=0)
image_frame.pack(pady=20)

image_label = Label(image_frame, bg=CARD_COLOR)
image_label.pack(padx=20, pady=20)

output_box = Text(
    root,
    height=14,
    width=100,
    font=("Consolas", 11),
    bg=CARD_COLOR,
    fg=TEXT_COLOR,
    relief="flat",
    padx=15,
    pady=15,
)
output_box.pack(pady=20)

# ================================
# ADVANCED UI ANIMATIONS
# ================================
def pulse_header(step=0):
    colors = [PRIMARY, SECONDARY, "#8B5CF6", "#A855F7"]
    header.config(fg=colors[step % len(colors)])
    root.after(500, lambda: pulse_header(step + 1))


def animate_image_border(step=0):
    shades = ["#E9D5FF", "#DDD6FE", "#F5D0FE", "#EDE9FE"]
    image_frame.config(bg=shades[step % len(shades)])
    image_label.config(bg=shades[step % len(shades)])
    root.after(700, lambda: animate_image_border(step + 1))


def blink_status(step=0):
    status_colors = [SECONDARY, PRIMARY]
    status_label.config(fg=status_colors[step % 2])
    root.after(600, lambda: blink_status(step + 1))


def animate_output_glow(step=0):
    glow = [CARD_COLOR, "#F3E8FF", "#FAE8FF"]
    output_box.config(bg=glow[step % len(glow)])
    root.after(900, lambda: animate_output_glow(step + 1))


# Start all animations
pulse_header()
animate_image_border()
blink_status()
animate_output_glow()

root.mainloop()
