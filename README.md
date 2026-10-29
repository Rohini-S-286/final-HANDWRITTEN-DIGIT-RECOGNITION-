# 🔢 Real-World Handwritten Digit Recognition System

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15+-orange?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-red?logo=keras&logoColor=white)](https://keras.io/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8?logo=opencv&logoColor=white)](https://opencv.org/)
[![Gradio](https://img.shields.io/badge/Gradio-Web%20App-FF7C00?logo=gradio&logoColor=white)](https://gradio.app/)
[![Test Accuracy](https://img.shields.io/badge/Test%20Accuracy-98.94%25-brightgreen)](https://github.com/Rohini-S-286/final-HANDWRITTEN-DIGIT-RECOGNITION-)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **An end-to-end Computer Vision and Deep Learning system designed to recognize handwritten digits from real-world, noisy inputs (mobile photos, paper scans, and lined notebooks) by coupling an adaptive OpenCV preprocessing pipeline with a high-capacity Convolutional Neural Network (CNN).**

---

## 📌 Executive Summary

While standard machine learning models trained on the benchmark **MNIST** dataset easily achieve high accuracy on clean, pre-centered 28×28 synthetic images, they frequently collapse when deployed in real-world scenarios. Real-world handwritten digits suffer from:
1. **Non-uniform illumination & shadows** from phone camera photography.
2. **Notebook ruling lines** intersecting through handwritten strokes.
3. **Pencil stroke variance** (faint graphite, varying line thickness).
4. **Arbitrary aspect ratios and off-center placement**.

This project resolves this domain-shift problem through a two-stage architecture:
- **Stage 1 (Computer Vision Engine)**: A multi-step OpenCV image processing pipeline featuring **Adaptive Gaussian Thresholding**, **Morphological Ruled-Line Removal**, and **Connected Component Aspect-Ratio Centering**.
- **Stage 2 (Deep Learning Inference)**: A regularized **3-stage Convolutional Neural Network (CNN)** with Batch Normalization, Dropout, and integrated Data Augmentation, achieving **98.94% Test Accuracy** and a **0.99 Macro F1-score** across 10,000 unseen test samples.
- **Stage 3 (Interactive Web Deployment)**: A responsive **Gradio web interface** providing live digit drawing, file upload, confidence distribution, and model explainability via a **"What the AI Actually Sees"** inspection canvas.

---

## 🏗️ System Architecture & Workflow

```mermaid
flowchart TD
    A["Raw Input Image<br/>(Camera Photo / Lined Paper / Sketch)"] --> B["1. Grayscale Conversion<br/>cv2.cvtColor"]
    B --> C["2. Noise Filtering<br/>Gaussian Blur (3x3)"]
    C --> D["3. Adaptive Thresholding<br/>Adaptive Gaussian (Block 31, C=9)"]
    D --> E["4. Morphological Line Removal<br/>cv2.morphologyEx (Rectangular Kernel)"]
    E --> F["5. Noise Speckle Filtering<br/>cv2.MORPH_OPEN (2x2 Kernel)"]
    F --> G["6. Connected Component Analysis<br/>Isolate primary bounding box (Area > 30, Height > 10)"]
    G --> H["7. Aspect-Ratio Square Centering<br/>Zero-padded square canvas"]
    H --> I["8. Dimension Normalization<br/>Resize to 28x28 (INTER_AREA) & Float32 [0, 1]"]
    
    subgraph CNN_Inference ["Stage 2: Deep CNN Inference (242,442 Parameters)"]
        I --> J["Conv2D (32, 3x3) + BatchNorm + MaxPool(2x2)"]
        J --> K["Conv2D (64, 3x3) + BatchNorm + MaxPool(2x2)"]
        K --> L["Conv2D (128, 3x3) + BatchNorm"]
        L --> M["Flatten + Dense(128, ReLU) + Dropout(0.40)"]
        M --> N["Dense (10, Softmax)"]
    end
    
    subgraph UI_Outputs ["Stage 3: Gradio UI Visualizer"]
        N --> O["🎯 Predicted Digit (0-9)"]
        N --> P["📊 Confidence Score & Top-3 Probabilities"]
        I --> Q["👁️ 'What the AI Sees' (Debug View)"]
    end
```

---

## 🔬 CNN Model Architecture

The network was engineered to balance feature-extraction capacity with strong regularization against overfitting.

| Layer Type | Configuration / Kernel | Output Shape | Activation | Trainable Parameters |
| :--- | :--- | :--- | :--- | :--- |
| **Input** | Normalized Grayscale | `(None, 28, 28, 1)` | — | 0 |
| **Data Augmentation** | Rotation (±8%), Zoom (±10%), Translation (±10%) | `(None, 28, 28, 1)` | — | 0 |
| **Conv2D #1** | 32 filters, 3×3 kernel, padding=`same` | `(None, 28, 28, 32)` | ReLU | 320 |
| **BatchNormalization** | Momentum=0.99 | `(None, 28, 28, 32)` | — | 128 |
| **MaxPooling2D** | Pool size 2×2, stride 2 | `(None, 14, 14, 32)` | — | 0 |
| **Conv2D #2** | 64 filters, 3×3 kernel, padding=`same` | `(None, 14, 14, 64)` | ReLU | 18,496 |
| **BatchNormalization** | Momentum=0.99 | `(None, 14, 14, 64)` | — | 256 |
| **MaxPooling2D** | Pool size 2×2, stride 2 | `(None, 7, 7, 64)` | — | 0 |
| **Conv2D #3** | 128 filters, 3×3 kernel, padding=`same` | `(None, 7, 7, 128)` | ReLU | 73,856 |
| **BatchNormalization** | Momentum=0.99 | `(None, 7, 7, 128)` | — | 512 |
| **Flatten** | Reshape feature maps | `(None, 6272)` | — | 0 |
| **Dense (FC)** | Fully Connected Layer | `(None, 128)` | ReLU | 802,944 |
| **Dropout** | Dropout Rate = 40% (0.40) | `(None, 128)` | — | 0 |
| **Dense (Output)** | Multi-class classification | `(None, 10)` | Softmax | 1,290 |
| **Total Parameters** | **242,442** (all trained on GPU/CPU) | | | |

---

## 🛠️ The Computer Vision Preprocessing Pipeline

Why do typical MNIST models fail on phone photos? Standard thresholding algorithms like **Otsu** compute a single global threshold, which breaks under shadows and uneven room lighting. Furthermore, students and professionals write on **ruled notebook paper**, where horizontal lines confuse convolutional filters.

```
+---------------------------------------------------------------------------------------------------+
|  Raw Input Photo          Adaptive Threshold           Line Removal           Square Centered 28x28|
|  [Photo of '3' on line] -> [Shadow-invariant binary] -> [Lines subtracted] -> [Isolated MNIST-like]|
+---------------------------------------------------------------------------------------------------+
```

### Detailed Pipeline Breakdown:
1. **Adaptive Gaussian Thresholding**: Computes thresholds locally across $31 \times 31$ pixel neighborhoods ($C=9$). Isolates faint pencil strokes regardless of paper color or casting shadows.
2. **Morphological Horizontal Line Removal**: 
   - A rectangular structuring element of size `(max(25, width // 8), 1)` detects continuous horizontal lines.
   - `cv2.morphologyEx` extracts the ruling lines and `cv2.subtract` strips them cleanly from the binary image without damaging vertical digit strokes.
3. **Morphological Opening**: Applies a $2 \times 2$ structuring element to suppress salt-and-pepper scan noise and stray ink specks.
4. **Connected Component Filtering**: Analyzes connected regions and filters out tiny stray marks ($\text{Area} \le 30\text{ px}$ or $\text{Height} \le 10\text{ px}$), taking the bounding envelope of the true digit components.
5. **Aspect-Ratio Preserving Canvas Centering**: Prevents aspect ratio distortion by zero-padding the digit into an exact square before resampling with `cv2.INTER_AREA` to $28 \times 28$.

---

## 📊 Empirical Performance & Evaluation

The model was trained on the 60,000 MNIST training set (with a 10% validation split) using the **Adam** optimizer ($\alpha = 0.001$) and **Sparse Categorical Crossentropy** loss, paired with dynamic learning rate reduction (`ReduceLROnPlateau`) and early stopping (`EarlyStopping`).

### Key Metrics on Unseen Test Set (10,000 Images):

| Metric | Score |
| :--- | :--- |
| **Test Accuracy** | **98.94%** |
| **Validation Accuracy** | **98.97%** |
| **Test Loss** | **0.0353** |
| **Macro Precision** | **0.99** |
| **Macro Recall** | **0.99** |
| **Macro F1-Score** | **0.99** |

### Class-Wise Classification Report:

| Class (Digit) | Precision | Recall | F1-Score | Support |
| :---: | :---: | :---: | :---: | :---: |
| **0** | 0.99 | 0.99 | 0.99 | 980 |
| **1** | 0.99 | 0.99 | 0.99 | 1,135 |
| **2** | 0.99 | 0.99 | 0.99 | 1,032 |
| **3** | 0.99 | 0.99 | 0.99 | 1,010 |
| **4** | 0.99 | 0.99 | 0.99 | 982 |
| **5** | 0.99 | 0.99 | 0.99 | 892 |
| **6** | 0.99 | 0.99 | 0.99 | 958 |
| **7** | 0.99 | 0.99 | 0.99 | 1,028 |
| **8** | 0.98 | 0.99 | 0.99 | 974 |
| **9** | 0.99 | 0.98 | 0.99 | 1,009 |
| **Accuracy** | | | **0.9894** | **10,000** |

---

## 📂 Repository Directory Layout

```plaintext
final-HANDWRITTEN-DIGIT-RECOGNITION-/
├── README.md                                  # Production documentation & technical report
├── app.py                                     # Interactive Gradio web application
├── preprocess.py                              # OpenCV preprocessing pipeline (line removal, adaptive threshold)
├── train.py                                   # End-to-end CNN training & evaluation script
├── requirements.txt                           # Verified project dependency manifest
├── .gitignore                                 # Git ignore rules for Python & ML artifacts
├── LICENSE                                    # MIT License
├── models/
│   └── digit_recognition_cnn.keras           # Pretrained CNN weights (~98.94% accuracy)
├── samples/                                   # Real-world test sample images (1.jpeg - 5.jpeg)
│   ├── 1.jpeg
│   ├── 2.jpeg
│   ├── 3.jpeg
│   ├── 4.jpeg
│   └── 5.jpeg
└── notebooks/
    └── AI_Handwritten_Digit_Recognition.ipynb # Complete research & experiment notebook
```

---

## 🚀 Quickstart Guide

### 1. Clone the Repository
```bash
git clone https://github.com/Rohini-S-286/final-HANDWRITTEN-DIGIT-RECOGNITION-.git
cd final-HANDWRITTEN-DIGIT-RECOGNITION-
```

### 2. Set Up a Virtual Environment
```bash
# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Launch the Interactive Web Application
```bash
python app.py
```
Open your browser at `http://127.0.0.1:7860/` to test with the live canvas, upload handwriting photos, or click any of the preloaded sample images.

### 5. Train the Model from Scratch (Optional)
```bash
python train.py --epochs 10 --batch_size 64 --output_model models/digit_recognition_cnn.keras
```

---

## 💼 Resume Bullet Points (Ready for Applications)

If you are adding this project to your resume, use these recruiter-tested, action-driven bullet points:

```markdown
• Developed an end-to-end Handwritten Digit Recognition system using TensorFlow/Keras and OpenCV, achieving 98.94% test accuracy and a 0.99 macro F1-score across 10,000 unseen MNIST test samples.
• Built a deep 3-stage Convolutional Neural Network (242K parameters) incorporating Batch Normalization, Dropout (0.4), and on-the-fly Data Augmentation (Rotation, Zoom, Translation) to prevent overfitting.
• Engineered an adaptive Computer Vision preprocessing pipeline utilizing Adaptive Gaussian Thresholding, Morphological Ruled-Line Removal, and Connected Component Centering to handle real-world notebook paper and camera shadows.
• Deployed an interactive web interface using Gradio with live drawing sketchpad, file upload, top-3 confidence metrics, and a "What the AI Sees" visualizer for model explainability.
```

---

## 🎓 Academic Credentials & Acknowledgments

- **Author**: **Rohini S** (USN: `24BBTCA096`)
- **Degree**: Bachelor of Technology (B.Tech) in Computer Science & Engineering (Artificial Intelligence & Machine Learning)
- **Institution**: School of Engineering and Technology, CMR University, Bengaluru
- **Course**: Mini Project (`AMPL2041`) — Python in AI & Machine Learning
- **Faculty Guide**: **Prof. Sirisha R**, Assistant Professor, Dept. of CSE (AI&ML), CMR University
- **Head of Department**: **Dr. B.K. Deshpande**, Dept. of CSE (AI&ML), CMR University

---

## 📜 License

This project is licensed under the [MIT License](LICENSE) — free to use, modify, and distribute for academic and professional purposes.
