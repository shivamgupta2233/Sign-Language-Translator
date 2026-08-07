from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model

# =====================
# Configuration
# =====================

DATASET_PATH = "dataset/kaggle"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# =====================
# Dataset
# =====================

datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,

    rotation_range=20,
    zoom_range=0.2,
    shear_range=0.2
)

train_generator = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    subset="training",
    class_mode="categorical"
)

validation_generator = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    subset="validation",
    class_mode="categorical"
)

print("\nClasses Found:")
print(train_generator.class_indices)

# =====================
# MobileNetV2 Model
# =====================

base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

# Freeze pretrained layers
base_model.trainable = False

# Custom classifier
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)

predictions = Dense(
    train_generator.num_classes,
    activation="softmax"
)(x)

model = Model(
    inputs=base_model.input,
    outputs=predictions
)

# Compile
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# Model Summary
model.summary()

# =====================
# Create folders
# =====================

os.makedirs("model", exist_ok=True)
os.makedirs("graphs", exist_ok=True)

# =====================
# Callbacks
# =====================

callbacks = [
    EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True
    ),

    ModelCheckpoint(
        filepath="model/sign_model.keras",
        monitor="val_accuracy",
        save_best_only=True
    )
]

# =====================
# Train Model
# =====================

history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=1,
    callbacks=callbacks
)

# =====================
# Accuracy Graph
# =====================

plt.figure(figsize=(8,5))

plt.plot(history.history["accuracy"], label="Train")
plt.plot(history.history["val_accuracy"], label="Validation")

plt.title("Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.savefig("graphs/accuracy.png")

plt.close()

# =====================
# Loss Graph
# =====================

plt.figure(figsize=(8,5))

plt.plot(history.history["loss"], label="Train")
plt.plot(history.history["val_loss"], label="Validation")

plt.title("Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.savefig("graphs/loss.png")

plt.close()

# =====================
# Save Labels
# =====================

with open("model/labels.txt", "w") as f:
    for label in train_generator.class_indices:
        f.write(label + "\n")

print("\n====================================")
print("✅ Training Completed Successfully")
print("✅ Model Saved -> model/sign_model.keras")
print("✅ Labels Saved -> model/labels.txt")
print("====================================")
