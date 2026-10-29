"""
Interactive Gradio Web Application for Handwritten Digit Recognition
Author: Rohini S (CMR University)

Provides an intuitive UI allowing users to either upload an image of a handwritten digit
(even with shadows or notebook lines) or sketch directly onto a canvas, showcasing
both real-time prediction and computer vision explainability ("What the AI Actually Sees").
"""

import os
import cv2
import numpy as np
import tensorflow as tf
import gradio as gr
from preprocess import preprocess_handwriting

# 1. Locate and Load the Model
MODEL_PATHS = [
    os.path.join("models", "digit_recognition_cnn.keras"),
    "digit_recognition_cnn.keras",
    "digit_recognition_model.keras",
    os.path.join(os.path.dirname(__file__), "models", "digit_recognition_cnn.keras")
]

model = None
for path in MODEL_PATHS:
    if os.path.exists(path):
        try:
            model = tf.keras.models.load_model(path)
            print(f"Loaded trained model from: {path}")
            break
        except Exception as e:
            print(f"Error loading {path}: {e}")

if model is None:
    print("Warning: Pretrained model not found. Please run train.py first.")


def predict_digit(image):
    """
    Inference endpoint:
    Preprocesses input image via OpenCV and outputs predicted digit,
    confidence score breakdown, and preprocessed 28x28 visualization.
    """
    if image is None:
        return "No input provided", "0.00%", None, {}

    # Handle Gradio Sketchpad dictionary or numpy array
    if isinstance(image, dict):
        # Gradio Image sketchpad format
        composite = image.get("composite") or image.get("image") or image.get("background")
        if composite is not None:
            image_array = np.array(composite)
        else:
            return "No drawing detected", "0.00%", None, {}
    else:
        image_array = np.array(image)

    # 1. Preprocess using our custom OpenCV pipeline
    processed_28 = preprocess_handwriting(image_array)

    if model is None:
        return "Model not loaded", "N/A", processed_28, {}

    # 2. Reshape for CNN input: (1, 28, 28, 1)
    tensor_in = processed_28.reshape(1, 28, 28, 1)

    # 3. Predict Probabilities
    probs = model.predict(tensor_in, verbose=0)[0]
    predicted_digit = int(np.argmax(probs))
    confidence_pct = float(probs[predicted_digit] * 100)

    # Dictionary of class probabilities for Gradio Label chart
    prob_dict = {f"Digit {i}": float(probs[i]) for i in range(10)}

    # Scaled visual display for the preprocessed image (zoom up for easy viewing)
    display_processed = cv2.resize((processed_28 * 255).astype(np.uint8), (140, 140), interpolation=cv2.INTER_NEAREST)

    return (
        f"{predicted_digit}",
        f"{confidence_pct:.2f}%",
        display_processed,
        prob_dict
    )


# 2. Build Modern Gradio Interface
custom_css = """
.container { max-width: 950px; margin: auto; }
.header-box { text-align: center; padding: 20px 10px; }
.output-digit { font-size: 40px !important; font-weight: bold; text-align: center; color: #2563eb; }
"""

with gr.Blocks(title="AI Handwritten Digit Recognition - Rohini S") as demo:
    gr.Markdown(
        """
        # 🔢 AI Handwritten Digit Recognition System
        ### Developed by **Rohini S** | B.Tech CSE (AI & ML), CMR University
        *An end-to-end Computer Vision & Deep Learning project featuring a **Deep CNN (98.94% Test Accuracy)** combined with an adaptive **OpenCV Preprocessing Pipeline** that cleans pencil strokes and eliminates lined-notebook rulings.*
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            input_mode = gr.Image(
                label="✍️ Draw or Upload Handwritten Digit",
                type="numpy",
                sources=["upload", "clipboard", "webcam"]
            )

            # Check if samples folder exists for example buttons
            sample_dir = os.path.join(os.path.dirname(__file__), "samples")
            sample_images = []
            if os.path.exists(sample_dir):
                for f in sorted(os.listdir(sample_dir)):
                    if f.lower().endswith((".jpg", ".jpeg", ".png")):
                        sample_images.append([os.path.join(sample_dir, f)])

            if sample_images:
                gr.Examples(
                    examples=sample_images,
                    inputs=input_mode,
                    label="📁 Test Samples (Click to Test Instantly)"
                )

            recognize_btn = gr.Button("🔍 Recognize Digit", variant="primary", size="lg")
            clear_btn = gr.ClearButton([input_mode])

        with gr.Column(scale=1):
            with gr.Group():
                pred_digit = gr.Textbox(label="🎯 Predicted Digit", elem_classes=["output-digit"])
                pred_conf = gr.Textbox(label="📊 Prediction Confidence")

            ai_view = gr.Image(
                label="👁️ What the AI Actually Sees (28x28 Preprocessed & Centered)",
                type="numpy"
            )

            prob_chart = gr.Label(num_top_classes=3, label="📈 Top Probability Distribution")

    recognize_btn.click(
        fn=predict_digit,
        inputs=[input_mode],
        outputs=[pred_digit, pred_conf, ai_view, prob_chart]
    )

    gr.Markdown(
        """
        ---
        **Key Features:**
        - **Model Architecture**: 3-Stage Conv2D + BatchNorm + MaxPool + Dense with Dropout (242K parameters).
        - **Real-World Robustness**: Automatically filters notebook horizontal lines, suppresses shadow gradients, and isolates pencil strokes.
        - **Academic Guidance**: Prof. Sirisha R, Dept. of CSE (AI&ML), CMR University, Bengaluru.
        """
    )

if __name__ == "__main__":
    demo.launch(share=False)
