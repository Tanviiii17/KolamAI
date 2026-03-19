\

import os, random, shutil
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers, callbacks
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.applications.efficientnet import preprocess_input
from sklearn.metrics import classification_report, confusion_matrix
from pathlib import Path

# ── CONFIG ──
IMG_SIZE    = 300        # EfficientNetB3 optimal size
BATCH_SIZE  = 32
SEED        = 42
MODEL_PATH  = "kolam_model.keras"
DATASET_DIR = Path("/content/drive/MyDrive/kolam-dataset")
CLASSES     = ["Dot_Grid", "Flower_Motif", "Geometric",
               "Sikku_Pattern", "Spiral_Design", "Star_Pattern"]

tf.random.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

gpus = tf.config.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
print(f"✅ GPU: {len(gpus)} device(s)" if gpus else "⚠️ CPU only")

# ── STRATIFIED SPLIT ──
SPLIT_DIR = Path("/content/kolam_split")
TRAIN_DIR = SPLIT_DIR / "train"
VAL_DIR   = SPLIT_DIR / "val"

if SPLIT_DIR.exists():
    shutil.rmtree(SPLIT_DIR)

print("\n📂 Creating stratified split...")
for cls in CLASSES:
    images = list((DATASET_DIR / cls).glob("*.*"))
    random.shuffle(images)
    val_n = int(len(images) * 0.2)
    splits = {"val": images[:val_n], "train": images[val_n:]}
    for split, imgs in splits.items():
        dest = SPLIT_DIR / split / cls
        dest.mkdir(parents=True, exist_ok=True)
        for img in imgs:
            shutil.copy(img, dest / img.name)
    print(f"  {cls}: {len(splits['train'])} train | {val_n} val")

# ── LOAD WITH tf.keras ──
# IMPORTANT: no rescaling here — preprocess_input handles it
train_ds = keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    labels="inferred", label_mode="categorical",
    class_names=CLASSES,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=True, seed=SEED,
)
val_ds = keras.utils.image_dataset_from_directory(
    VAL_DIR,
    labels="inferred", label_mode="categorical",
    class_names=CLASSES,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=False,
)

print(f"\n✅ Train batches: {len(train_ds)}")
print(f"✅ Val   batches: {len(val_ds)}")

# ── PREPROCESSING ──
# EfficientNet needs its OWN preprocess_input (NOT /255)
# Light augmentation only — aggressive aug was killing accuracy

data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.15),       # ← reduced from 0.25
    layers.RandomZoom(0.1),            # ← reduced from 0.2
], name="augmentation")

def prepare_train(images, labels):
    images = data_augmentation(images, training=True)
    images = preprocess_input(images)  # ← correct EfficientNet preprocessing
    return images, labels

def prepare_val(images, labels):
    images = preprocess_input(images)  # ← correct EfficientNet preprocessing
    return images, labels

train_ds = train_ds.map(prepare_train,
    num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
val_ds   = val_ds.map(prepare_val,
    num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)

# ── CLASS WEIGHTS ──
counts = {cls: len(list((TRAIN_DIR / cls).glob("*"))) for cls in CLASSES}
total  = sum(counts.values())
class_weight_dict = {
    i: (total / (len(CLASSES) * counts[cls]))
    for i, cls in enumerate(CLASSES)
}
print("\n⚖️  Class weights:")
for i, cls in enumerate(CLASSES):
    print(f"   [{i}] {cls}: {class_weight_dict[i]:.3f}  ({counts[cls]} imgs)")

# ── MODEL ──
def build_model():
    base = EfficientNetB3(
        include_top=False,
        weights="imagenet",
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )
    base.trainable = False

    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation="relu",
                     kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(len(CLASSES), activation="softmax")(x)
    return keras.Model(inputs, outputs), base

model, base_model = build_model()
model.summary()

def make_callbacks(phase):
    return [
        callbacks.ModelCheckpoint(
            MODEL_PATH, monitor="val_accuracy",
            save_best_only=True, verbose=1),
        callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=8 if phase == 1 else 10,
            restore_best_weights=True, verbose=1),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5,
            patience=3, min_lr=1e-8, verbose=1),
    ]

# ── PHASE 1: Frozen base — train head ──
print("\n" + "="*60)
print("🔥 PHASE 1 — Head training (base frozen)")
print("="*60)

model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy",
             keras.metrics.TopKCategoricalAccuracy(k=2, name="top2_acc")],
)

h1 = model.fit(
    train_ds, validation_data=val_ds,
    epochs=25,
    class_weight=class_weight_dict,
    callbacks=make_callbacks(1),
    verbose=1,
)

best_p1_acc = max(h1.history["val_accuracy"])
print(f"\n✅ Phase 1 best val_accuracy: {best_p1_acc*100:.2f}%")

# ── PHASE 2: Unfreeze top layers ──
print("\n" + "="*60)
print("🔥 PHASE 2 — Fine-tuning (top 30 layers unfrozen)")
print("="*60)

# Unfreeze only top 30 layers — more conservative
base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

trainable = sum(1 for l in base_model.layers if l.trainable)
print(f"   Unfrozen base layers: {trainable}")

model.compile(
    optimizer=keras.optimizers.Adam(2e-5),   # low LR for fine-tuning
    loss="categorical_crossentropy",
    metrics=["accuracy",
             keras.metrics.TopKCategoricalAccuracy(k=2, name="top2_acc")],
)

h2 = model.fit(
    train_ds, validation_data=val_ds,
    epochs=40,
    class_weight=class_weight_dict,
    callbacks=make_callbacks(2),
    verbose=1,
)

# ── EVALUATION ──
print("\n" + "="*60)
print("📊 FINAL EVALUATION")
print("="*60)

best = keras.models.load_model(MODEL_PATH)
res  = best.evaluate(val_ds, verbose=1)
print(f"\n✅ Val Accuracy : {res[1]*100:.2f}%")
print(f"✅ Top-2 Acc    : {res[2]*100:.2f}%")
print(f"✅ Val Loss     : {res[0]:.4f}")

y_true, y_pred = [], []
for imgs, lbls in val_ds:
    preds = best.predict(imgs, verbose=0)
    y_pred.extend(np.argmax(preds, axis=1))
    y_true.extend(np.argmax(lbls.numpy(), axis=1))

print("\n📋 Classification Report:")
print(classification_report(y_true, y_pred,
      target_names=CLASSES, zero_division=0))

# ── PLOTS ──
p1 = len(h1.history["accuracy"])
acc  = h1.history["accuracy"]     + h2.history["accuracy"]
vacc = h1.history["val_accuracy"] + h2.history["val_accuracy"]
loss = h1.history["loss"]         + h2.history["loss"]
vloss= h1.history["val_loss"]     + h2.history["val_loss"]

fig, ax = plt.subplots(1, 2, figsize=(14, 5))
ax[0].plot(acc,  label="Train", color="#e8a020")
ax[0].plot(vacc, label="Val",   color="#d4559a")
ax[0].axvline(p1, color="gray", linestyle="--", label="Fine-tune start")
ax[0].set_title("Accuracy"); ax[0].legend()

ax[1].plot(loss,  label="Train", color="#e8a020")
ax[1].plot(vloss, label="Val",   color="#d4559a")
ax[1].axvline(p1, color="gray", linestyle="--", label="Fine-tune start")
ax[1].set_title("Loss"); ax[1].legend()

plt.tight_layout()
plt.savefig("training_curves.png", dpi=150)
plt.show()

cm = confusion_matrix(y_true, y_pred)
fig2, axis = plt.subplots(figsize=(9, 7))
sns.heatmap(cm, annot=True, fmt="d", cmap="YlOrRd",
            xticklabels=CLASSES, yticklabels=CLASSES, ax=axis)
axis.set_xlabel("Predicted")
axis.set_ylabel("True")
axis.set_title("Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.show()

print(f"\n🎉 Done! Best model → {MODEL_PATH}")