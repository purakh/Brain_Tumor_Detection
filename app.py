from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import cv2
import numpy as np
import os
from PIL import Image,UnidentifiedImageError

app = Flask(__name__)

# Load models
valid_mri_model = load_model('models/valid_mri_64x64.h5')
tumor_model = load_model('models/BrainTumor10EpochsCategorical.h5')

INPUT_SIZE_VALID = 64
INPUT_SIZE_TUMOR = 64

def preprocess_image(img_path, target_size):
    try:
        image = Image.open(img_path).convert("RGB")
    except UnidentifiedImageError:
        raise ValueError("Uploaded file is not a valid image.")
    
    image = image.resize((target_size, target_size))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file part", 400

    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400

    filepath = os.path.join("static", file.filename)
    file.save(filepath)

    # Step 1: Check if it's a valid MRI
    valid_input = preprocess_image(filepath, INPUT_SIZE_VALID)
    validity = valid_mri_model.predict(valid_input)[0][0]

    if validity < 0.5:
        os.remove(filepath)
        return "Please upload a valid brain MRI scan.", 400

    # Step 2: Predict tumor classification
    tumor_input = preprocess_image(filepath, INPUT_SIZE_TUMOR)
    prediction = tumor_model.predict(tumor_input)[0]

    classes = ['No Tumor', 'Tumor']
    result = classes[np.argmax(prediction)]

    return result

if __name__ == '__main__':
    app.run(debug=True)