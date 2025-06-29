WEEK 1:-


1. Dataset Handling
* Loaded the TrashNet dataset.
* Split it into training and validation sets.
* Applied data augmentation (flip, rotate, zoom) to improve generalization.

2. Model Architecture
*Used EfficientNetV2B2 as a pre-trained backbone.
*Added custom classification layers (GlobalAveragePooling, Dropout, Dense).
*Froze the base model to retain learned features from ImageNet.

3. Model Training
*Trained the model for 15 epochs.
*Tracked training and validation accuracy/loss.
*Saved the trained model to disk (garbage_classifier_model.keras).

4. Performance Visualization
*Plotted training vs. validation accuracy and loss curves.
*Evaluated the model on validation data to get final test accuracy and loss.
