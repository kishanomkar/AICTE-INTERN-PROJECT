# 🗑️ Garbage Classification using EfficientNetV2B2

A deep learning project to classify waste into categories using the TrashNet dataset. This helps in automating the waste segregation process using image classification with TensorFlow and EfficientNet.

---

## 📅 WEEK 1: Data Handling & Initial Model

### 📂 Dataset Handling
- Loaded the **TrashNet** dataset.
- Split into **training** and **validation** sets.
- Applied **data augmentation** (flip, rotate, zoom) to improve generalization.

### 🧠 Model Architecture
- Used **EfficientNetV2B2** as a pre-trained backbone.
- Added custom classification layers:  
  `GlobalAveragePooling → Dropout → Dense (softmax)`
- Froze the base model to retain **ImageNet** features.

### 🏋️ Model Training
- Trained for **15 epochs**.
- Tracked training and validation **accuracy/loss**.
- Saved model as: `garbage_classifier_model.keras`

### 📈 Performance Visualization
- Plotted **accuracy** and **loss curves**.
- Evaluated on validation set to get final test accuracy and loss.

---

## 📅 WEEK 2: Evaluation & Error Analysis

### 📊 Evaluation
- Trained and saved as: `garbage_classifier_model_boosted_trash.keras`
- Final validation accuracy: **76.04%**
- ⚠️ **Trash class** had **0.00 precision/recall** (underperforming)

### 🧾 Reports Generated
- Confusion matrix
- Classification report
- Training/validation plots

### 🐛 Error Analysis
- Visualized **misclassified trash** images.
- Observed trash is often confused with **plastic**, **cardboard**, or **metal**.

---

## 📅 WEEK 3: Improvements & Testing

### ⚙️ Model Improvements
- Implemented **Focal Loss** to handle class imbalance (especially for trash).
- Augmented minority class samples to improve representation.
- Re-trained model with improved learning rate schedule.

### 🧪 Testing
- Re-evaluated using updated loss function.
- Improvement seen in trash class recall (details in classification report).
- Plotted **enhanced confusion matrix** and classification metrics.

### 💾 Deployment Prep
- Converted model to **TensorFlow Lite** (`.tflite`) for mobile/web deployment.
- Setup **Gradio UI** for testing predictions interactively.

---

## 📦 Files Generated
- `garbage_classifier_model_boosted_trash.keras`
- `garbage_classifier_focal.keras`
- `model_training_logs.png`
- `confusion_matrix_week3.png`
- `classification_report_week3.txt`
- `model.tflite`

---

---

Made with ❤️ by **Kishan Omkar**  
[LinkedIn ↗](https://www.linkedin.com/in/kishan-omkar-022226314)
