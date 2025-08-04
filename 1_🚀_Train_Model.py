import streamlit as st
import tensorflow as tf
import numpy as np
import os
import time
import pandas as pd
import pathlib
import logging
from typing import List, Tuple, Dict, Optional
from tensorflow.keras.layers import Input, Dense, Dropout, GlobalAveragePooling2D, RandomFlip, RandomRotation, RandomZoom, RandomContrast
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.callbacks import Callback

tf.get_logger().setLevel(logging.ERROR)

st.set_page_config(page_title="Train Model", page_icon="🚀", layout="wide")
st.write("📢 Training page loaded.")

st.title("🚀 Train Cancer Cell Classifier Model")

MODEL_PATH = 'cancer_model.h5'
CLASS_NAMES_PATH = 'class_names.txt'
IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS = 224, 224, 3

if 'training_logs' not in st.session_state: st.session_state.training_logs = []
if 'training_complete' not in st.session_state: st.session_state.training_complete = False
if 'training_history' not in st.session_state: st.session_state.training_history = None
if 'training_time' not in st.session_state: st.session_state.training_time = None
if 'dataset_path_ss' not in st.session_state: st.session_state.dataset_path_ss = ""

tab_data, tab_config, tab_train, tab_results = st.tabs([
    "💾 Data Setup", "⚙️ Configuration", "🚀 Train & Monitor", "📊 Results"
])

with tab_data:
    st.header("1. Data Setup")
    st.write("Provide the path to your dataset folder (must contain `cancer/` and `normal/` subfolders).")
    dataset_path = st.text_input("Dataset Folder Path", value=st.session_state.dataset_path_ss or "cancer_dataset_prepared")
    st.session_state.dataset_path_ss = dataset_path

with tab_config:
    st.header("2. Training Configuration")
    col1, col2, col3 = st.columns(3)
    with col1:
        epochs = st.number_input("Epochs", min_value=1, max_value=100, value=10, step=1)
    with col2:
        batch_size = st.selectbox("Batch Size", options=[8, 16, 32], index=1)
    with col3:
        learning_rate = st.number_input("Learning Rate", min_value=1e-6, max_value=1e-2, value=1e-3, step=1e-4, format="%.5f")

    use_augmentation = st.toggle("Enable Data Augmentation", value=True)
    validation_split = 0.2
    st.write(f"Using validation split of {int(validation_split*100)}%")

with tab_train:
    st.header("3. Start Training & Monitor Progress")
    start_training = st.button("Start Training")

    if start_training:
        st.session_state.training_logs = []
        st.session_state.training_complete = False
        progress_bar = st.progress(0, text="Starting...")

        data_dir = pathlib.Path(dataset_path)
        if not data_dir.exists():
            st.error(f"Dataset path not found: {data_dir}")
            st.stop()

        # Load data from folders
        class_names = sorted([f.name for f in data_dir.iterdir() if f.is_dir()])
        class_to_index = {name: i for i, name in enumerate(class_names)}

        image_paths = []
        labels = []
        for class_name in class_names:
            class_dir = data_dir / class_name
            for img_path in class_dir.glob("*"):
                if img_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                    image_paths.append(str(img_path))
                    labels.append(class_to_index[class_name])

        if len(image_paths) == 0:
            st.error("No images found in dataset folders.")
            st.stop()

        st.success(f"Found {len(image_paths)} images across {len(class_names)} classes.")
        with open(CLASS_NAMES_PATH, 'w') as f:
            for name in class_names:
                f.write(f"{name}\n")

        # Split data
        full_ds = tf.data.Dataset.from_tensor_slices((image_paths, labels))

        def load_image(path, label):
            img = tf.io.read_file(path)
            img = tf.image.decode_jpeg(img, channels=3)
            img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH])
            return tf.cast(img, tf.float32), label

        full_ds = full_ds.shuffle(buffer_size=len(image_paths), seed=42)
        train_size = int((1 - validation_split) * len(image_paths))
        train_ds = full_ds.take(train_size)
        val_ds = full_ds.skip(train_size)

        AUTOTUNE = tf.data.AUTOTUNE
        train_ds = train_ds.map(load_image, num_parallel_calls=AUTOTUNE)
        val_ds = val_ds.map(load_image, num_parallel_calls=AUTOTUNE)

        if use_augmentation:
            aug = Sequential([
                RandomFlip("horizontal"),
                RandomRotation(0.1),
                RandomZoom(0.1),
                RandomContrast(0.1)
            ])
            def apply_aug(img, label):
                img = aug(tf.expand_dims(img, 0), training=True)
                return tf.squeeze(img, 0), label
            train_ds = train_ds.map(apply_aug, num_parallel_calls=AUTOTUNE)

        train_ds = train_ds.batch(batch_size).prefetch(AUTOTUNE)
        val_ds = val_ds.batch(batch_size).prefetch(AUTOTUNE)

        progress_bar.progress(40, text="Data prepared.")

        # Build model
        inputs = Input(shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
        x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
        base_model = MobileNetV2(input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), include_top=False, weights='imagenet')
        base_model.trainable = False
        x = base_model(x, training=False)
        x = GlobalAveragePooling2D()(x)
        x = Dropout(0.2)(x)
        outputs = Dense(len(class_names), activation="softmax")(x)
        model = Model(inputs, outputs)

        model.compile(optimizer=Adam(learning_rate=learning_rate), loss=SparseCategoricalCrossentropy(), metrics=["accuracy"])
        progress_bar.progress(60, text="Model built.")

        # Callback for live logs
        class StreamlitLogger(Callback):
            def on_epoch_end(self, epoch, logs=None):
                log_msg = f"Epoch {epoch+1}/{epochs} - " + " - ".join([f"{k}: {v:.4f}" for k, v in logs.items()])
                st.session_state.training_logs.append(log_msg)
                st.text_area("Training Logs", value="\n".join(st.session_state.training_logs), height=250)

        st.write("📈 Training model...")
        start_time = time.time()
        history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=[StreamlitLogger()], verbose=0)
        end_time = time.time()

        model.save(MODEL_PATH)
        st.session_state.training_complete = True
        st.session_state.training_history = history.history
        st.session_state.training_time = end_time - start_time

        progress_bar.progress(100, text="Training complete!")
        st.success(f"✅ Model trained and saved as `{MODEL_PATH}` in {end_time - start_time:.2f} seconds.")
        st.balloons()

with tab_results:
    st.header("4. Training Results")
    if st.session_state.training_complete:
        st.success("Training finished!")
        st.write(f"Total training time: {st.session_state.training_time:.2f} seconds")
        hist = st.session_state.training_history
        df = pd.DataFrame(hist)

        if "accuracy" in df and "val_accuracy" in df:
            st.line_chart(df[["accuracy", "val_accuracy"]])
        if "loss" in df and "val_loss" in df:
            st.line_chart(df[["loss", "val_loss"]])

        st.dataframe(df)
        st.page_link("pages/2_🔬_Classify_Cell.py", label="Go to Classification Page", icon="🔬")
    else:
        st.info("Train the model to see results.")
