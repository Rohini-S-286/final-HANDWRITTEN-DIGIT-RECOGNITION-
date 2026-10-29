"""
Preprocessing Pipeline for Handwritten Digit Recognition
Author: Rohini S (CMR University)

This module provides a robust computer vision preprocessing pipeline to bridge
the gap between real-world handwriting images (scanned paper, photos taken via phone,
pencil on ruled notebook paper) and the standardized MNIST dataset format (28x28 inverted grayscale).
"""

import cv2
import numpy as np


def preprocess_handwriting(image):
    """
    Transforms any raw input image (photo, scan, canvas sketch) into a clean,
    centered, 28x28 normalized MNIST-compatible tensor.

    Pipeline Steps:
    1. Grayscale Conversion: Handles RGB, BGR, or Alpha channels.
    2. Noise Reduction: Gentle 3x3 Gaussian blur to smooth high-frequency sensor noise.
    3. Adaptive Gaussian Thresholding: Isolates strokes even under non-uniform illumination
       and faint pencil pressure (unlike static Otsu thresholding).
    4. Morphological Line Removal: Detects and subtracts horizontal lined-notebook rulings
       using an aspect-tuned rectangular structuring element.
    5. Morphological Opening: Removes isolated specks, dust, and scan artifacts.
    6. Connected Component Analysis: Identifies the primary digit components (filtering area & height).
    7. Aspect-Ratio Preserving Square Canvas: Centers the isolated bounding box into a square
       to prevent digit distortion before resizing.
    8. MNIST Dimensioning: Resizes to 28x28 using INTER_AREA interpolation and normalizes to [0.0, 1.0].

    Parameters:
        image (numpy.ndarray): Input image in BGR, RGB, or Grayscale format.

    Returns:
        numpy.ndarray: 28x28 float32 array with pixel values in range [0.0, 1.0].
    """
    if image is None:
        return np.zeros((28, 28), dtype=np.float32)

    # 1. Grayscale Conversion
    if isinstance(image, np.ndarray):
        if image.ndim == 3:
            if image.shape[2] == 4:
                # RGBA -> Gray
                gray = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
            else:
                # RGB / BGR -> Gray
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
    else:
        gray = np.array(image, dtype=np.uint8)

    # If canvas background is already dark (like some sketchpads), invert if predominantly white
    # In natural paper photos, background is white (>127) and ink is dark (<127)
    mean_val = gray.mean()
    if mean_val < 50:
        # Already white digit on dark background
        invert_mode = False
    else:
        invert_mode = True

    # 2. Gaussian Blur to reduce sensor and paper texture noise
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    # 3. Thresholding
    if invert_mode:
        # Natural photo: black ink on light paper -> invert so digit is white (255)
        binary = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            31,
            9
        )
    else:
        # Dark canvas: white ink on black background
        _, binary = cv2.threshold(blurred, 30, 255, cv2.THRESH_BINARY)

    # 4. Detect and remove notebook horizontal lines
    line_kernel_width = max(25, gray.shape[1] // 8)
    kernel_horizontal = cv2.getStructuringElement(cv2.MORPH_RECT, (line_kernel_width, 1))
    horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_horizontal)
    clean = cv2.subtract(binary, horizontal_lines)

    # 5. Remove small speckle noise (morphological opening)
    clean = cv2.morphologyEx(clean, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))

    # 6. Connected component analysis to isolate digit
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(clean, 8)
    components = []
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        if area > 30 and h > 10:
            components.append((x, y, w, h, area))

    if not components:
        # Fallback if no components pass strict threshold: check for non-zero points
        points = cv2.findNonZero(clean)
        if points is None:
            return np.zeros((28, 28), dtype=np.float32)
        x, y, w, h = cv2.boundingRect(points)
    else:
        # Combine bounding boxes of all significant digit components
        x1 = min(c[0] for c in components)
        y1 = min(c[1] for c in components)
        x2 = max(c[0] + c[2] for c in components)
        y2 = max(c[1] + c[3] for c in components)
        x, y, w, h = x1, y1, x2 - x1, y2 - y1

    # Add margin padding
    pad = 15
    y1_pad = max(0, y - pad)
    y2_pad = min(clean.shape[0], y + h + pad)
    x1_pad = max(0, x - pad)
    x2_pad = min(clean.shape[1], x + w + pad)

    digit_crop = clean[y1_pad:y2_pad, x1_pad:x2_pad]
    crop_h, crop_w = digit_crop.shape

    if crop_h == 0 or crop_w == 0:
        return np.zeros((28, 28), dtype=np.float32)

    # 7. Aspect-ratio preserving square canvas
    size = max(crop_h, crop_w)
    square = np.zeros((size, size), dtype=np.uint8)
    x_offset = (size - crop_w) // 2
    y_offset = (size - crop_h) // 2
    square[y_offset:y_offset + crop_h, x_offset:x_offset + crop_w] = digit_crop

    # 8. Resize to 28x28 and normalize to [0.0, 1.0]
    digit_28 = cv2.resize(square, (28, 28), interpolation=cv2.INTER_AREA)
    return digit_28.astype("float32") / 255.0
