import streamlit as st
from PIL import Image, ImageOps, ImageFilter
import numpy as np
import cv2

st.title("🖼 Simple Image Processing Website")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

# Filter options
filter_type = st.selectbox("Choose a filter:", [
    "Gray", "Blur", "Edge", "Sharpen", "Invert", "Sepia",
    "Brightness +", "Brightness -", "Contrast +", "Contrast -",
    "Sketch", "Gaussian Noise", "Canny Edge"
])

def apply_filter(img_cv, filter_name):
    if filter_name == "Gray":
        return cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    elif filter_name == "Blur":
        return cv2.GaussianBlur(img_cv, (15, 15), 0)
    elif filter_name == "Edge" or filter_name == "Canny Edge":
        return cv2.Canny(img_cv, 100, 200)
    elif filter_name == "Sharpen":
        kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])
        return cv2.filter2D(img_cv, -1, kernel)
    elif filter_name == "Invert":
        return cv2.bitwise_not(img_cv)
    elif filter_name == "Sepia":
        kernel = np.array([[0.272, 0.534, 0.131],
                           [0.349, 0.686, 0.168],
                           [0.393, 0.769, 0.189]])
        return cv2.transform(img_cv, kernel)
    elif filter_name == "Brightness +":
        return cv2.convertScaleAbs(img_cv, alpha=1.2, beta=30)
    elif filter_name == "Brightness -":
        return cv2.convertScaleAbs(img_cv, alpha=0.8, beta=-30)
    elif filter_name == "Contrast +":
        return cv2.convertScaleAbs(img_cv, alpha=1.5, beta=0)
    elif filter_name == "Contrast -":
        return cv2.convertScaleAbs(img_cv, alpha=0.7, beta=0)
    elif filter_name == "Sketch":
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        return cv2.Canny(gray, 50, 150)
    elif filter_name == "Gaussian Noise":
        row, col, ch = img_cv.shape
        mean = 0
        sigma = 25
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        gauss = gauss.reshape(row, col, ch)
        return cv2.add(img_cv, gauss.astype('uint8'))
    else:
        return img_cv

if uploaded_file:
    img = Image.open(uploaded_file)
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    processed = apply_filter(img_cv, filter_type)

    if len(processed.shape) == 2:
        processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2RGB)

    st.subheader("Original Image")
    st.image(img, use_column_width=True)
    st.subheader("Filtered Image")
    st.image(processed, use_column_width=True)
