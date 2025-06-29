import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import RandomFlip, RandomRotation, RandomZoom, RandomContrast, RandomBrightness
from tensorflow.keras.applications import EfficientNetV2B2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing import image
import os

# Load the dataset
dataset_path = "trashnet_dataset"

train_ds_raw = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=(128, 128),
    batch_size=32
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=(128, 128),
    batch_size=32
)

# Get class names BEFORE preprocessing
class_names = train_ds_raw.class_names
print("Classes:", class_names)

# Compute class weights
all_labels = []
for _, labels in train_ds_raw.unbatch():
    all_labels.append(labels.numpy())

class_weights_array = compute_class_weight(
    class_weight='balanced',
    classes=np.arange(len(class_names)),
    y=all_labels
)

class_weights = {i: weight for i, weight in enumerate(class_weights_array)}
print("Class Weights:", class_weights)

# Define standard augmentation
standard_aug = Sequential([
    RandomFlip("horizontal"),
    RandomRotation(0.1),
    RandomZoom(0.1)
])

# Define stronger augmentation for trash class
strong_aug = Sequential([
    RandomFlip("horizontal"),
    RandomRotation(0.2),
    RandomZoom(0.2),
    RandomContrast(0.2),
    RandomBrightness(0.2)
])

AUTOTUNE = tf.data.AUTOTUNE

# Separate trash samples (class index 5)
trash_ds = train_ds_raw.filter(lambda x, y: tf.reduce_all(tf.equal(y, 5)))
trash_ds = trash_ds.map(lambda x, y: (strong_aug(x), y)).repeat(3)

# Apply standard augmentation to the rest
non_trash_ds = train_ds_raw.filter(lambda x, y: tf.reduce_all(tf.not_equal(y, 5)))
non_trash_ds = non_trash_ds.map(lambda x, y: (standard_aug(x), y))

# Combine datasets
train_ds = non_trash_ds.concatenate(trash_ds)
train_ds = train_ds.shuffle(1000).cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

print("Data preprocessing and oversampling complete!")

# Load EfficientNetV2B2 without the top layer
base_model = EfficientNetV2B2(
    include_top=False, 
    input_shape=(128, 128, 3), 
    weights="imagenet"
)

# Unfreeze top layers for fine-tuning
base_model.trainable = True
for layer in base_model.layers[:200]:
    layer.trainable = False

# Define the final model
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dropout(0.3),
    Dense(6, activation="softmax")  
])

# Learning rate scheduler
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-4,
    decay_steps=1000,
    decay_rate=0.9
)

# Compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Early stopping callback
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model
epochs = 50

history = model.fit(
    train_ds,  
    validation_data=val_ds,  
    epochs=epochs,
    batch_size=32,
    class_weight=class_weights,
    callbacks=[early_stop]
)

# Save trained model
model.save("garbage_classifier_model_boosted_trash.keras")
print("Model saved successfully!")

# Plot training curves
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(len(acc))
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label="Training Accuracy")
plt.plot(epochs_range, val_acc, label="Validation Accuracy")
plt.legend(loc="lower right")
plt.title("Training vs Validation Accuracy")

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label="Training Loss")
plt.plot(epochs_range, val_loss, label="Validation Loss")
plt.legend(loc="upper right")
plt.title("Training vs Validation Loss")

plt.show()

# Evaluate model
model = tf.keras.models.load_model("garbage_classifier_model_boosted_trash.keras")
test_loss, test_acc = model.evaluate(val_ds)
print(f"Final Test Accuracy: {test_acc:.4f}")
print(f"Final Test Loss: {test_loss:.4f}")

# Confusion Matrix and Classification Report
y_true = []
y_pred = []

for images, labels in val_ds.unbatch():
    preds = model.predict(tf.expand_dims(images, axis=0), verbose=0)
    y_true.append(labels.numpy())
    y_pred.append(np.argmax(preds))

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

print("Classification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

# Real-world image prediction
def predict_image(img_path):
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = tf.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions)]
    confidence = np.max(predictions)
    
    print(f"Predicted Class: {predicted_class}")
    print(f"Confidence: {confidence:.2f}")

# Example usage:
# predict_image("test_images/trash_example.jpg")
