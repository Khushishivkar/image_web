from flask import Flask, render_template, request
import cv2
import numpy as np
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Upload route
@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    filter_type = request.form['filter']

    # Clear old images in the uploads folder
    for f in os.listdir(app.config['UPLOAD_FOLDER']):
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], f)
        if os.path.isfile(file_path):
            os.remove(file_path)

    # Save uploaded file
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # Read image using OpenCV
    img = cv2.imread(filepath)

    # Apply selected filter
    if filter_type == 'gray':
        processed = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif filter_type == 'blur':
        processed = cv2.GaussianBlur(img, (15, 15), 0)
    elif filter_type == 'edge':
        processed = cv2.Canny(img, 100, 200)
    elif filter_type == 'sharpen':
        kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])
        processed = cv2.filter2D(img, -1, kernel)
    elif filter_type == 'invert':
        processed = cv2.bitwise_not(img)
    elif filter_type == 'sepia':
        kernel = np.array([[0.272, 0.534, 0.131],
                           [0.349, 0.686, 0.168],
                           [0.393, 0.769, 0.189]])
        processed = cv2.transform(img, kernel)
    elif filter_type == 'bright_plus':
        processed = cv2.convertScaleAbs(img, alpha=1.2, beta=30)
    elif filter_type == 'bright_minus':
        processed = cv2.convertScaleAbs(img, alpha=0.8, beta=-30)
    elif filter_type == 'contrast_plus':
        processed = cv2.convertScaleAbs(img, alpha=1.5, beta=0)
    elif filter_type == 'contrast_minus':
        processed = cv2.convertScaleAbs(img, alpha=0.7, beta=0)
    elif filter_type == 'sketch':
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        processed = cv2.Canny(gray, 50, 150)
    elif filter_type == 'noise':
        row, col, ch = img.shape
        mean = 0
        sigma = 25
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        gauss = gauss.reshape(row, col, ch)
        processed = cv2.add(img, gauss.astype('uint8'))
    elif filter_type == 'canny':
        processed = cv2.Canny(img, 100, 200)
    else:
        processed = img

    # Save processed image
    processed_filename = 'filtered_' + filename
    processed_path = os.path.join(app.config['UPLOAD_FOLDER'], processed_filename)
    cv2.imwrite(processed_path, processed)

    # Render template with images
    return render_template('index.html', original=filename, filtered=processed_filename)

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
