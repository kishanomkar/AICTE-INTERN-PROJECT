import gradio as gr
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from focal_loss import SparseCategoricalFocalLoss

# Load model
model = tf.keras.models.load_model(
    "garbage_classifier_focal.keras",
    custom_objects={"SparseCategoricalFocalLoss": SparseCategoricalFocalLoss}
)

class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# Prediction function
def classify_garbage(img):
    if img is None:
        return {"Error": 0.0}
    img = img.resize((128, 128))
    img_array = image.img_to_array(img)
    img_array = tf.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions)]
    confidence = float(np.max(predictions))
    return {predicted_class: confidence}

# UI
with gr.Blocks(title="Garbage Classifier") as demo:
    gr.Markdown("""
    # 🗑️ Garbage Classifier
    Upload an image or use your webcam to classify waste as:
    **cardboard**, **glass**, **metal**, **paper**, **plastic**, or **trash**.
    """)

    with gr.Row():
        upload_image = gr.Image(type="pil", label="Upload Image", sources=["upload"])
        webcam_image = gr.Image(type="pil", label="Use Webcam", sources=["webcam"])

    result = gr.Label(num_top_classes=3, label="Top Predictions")
    button = gr.Button("🔍 Classify")

    def process(uploaded, webcam):
        return classify_garbage(uploaded if uploaded else webcam)

    button.click(fn=process, inputs=[upload_image, webcam_image], outputs=result)

demo.launch()
