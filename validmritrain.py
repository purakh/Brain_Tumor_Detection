import os
import cv2
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from keras.utils import normalize
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam

# Constants
INPUT_SIZE = 64
VALID_DIRS = ['datasets/no', 'datasets/yes']
INVALID_DIR = 'datasets/invalid'

valid_images = []
invalid_images = []

# Load valid brain MRI images (label = 1)
for folder in VALID_DIRS:
    for img_name in os.listdir(folder):
        if img_name.lower().endswith(('.jpg', '.png')):
            img_path = os.path.join(folder, img_name)
            try:
                img = Image.open(img_path).convert('RGB')
                img = img.resize((INPUT_SIZE, INPUT_SIZE))
                valid_images.append(np.array(img))
            except:
                pass

# Load non-MRI (invalid) images (label = 0)
for img_name in os.listdir(INVALID_DIR):
    if img_name.lower().endswith(('.jpg', '.png')):
        img_path = os.path.join(INVALID_DIR, img_name)
        try:
            img = Image.open(img_path).convert('RGB')
            img = img.resize((INPUT_SIZE, INPUT_SIZE))
            invalid_images.append(np.array(img))
        except:
            pass

# Combine and prepare data
all_images = np.array(valid_images + invalid_images)
all_labels = np.array([1]*len(valid_images) + [0]*len(invalid_images))

all_images = normalize(all_images, axis=1)

# Split into train and validation sets
train_images, val_images, train_labels, val_labels = train_test_split(
    all_images, all_labels, test_size=0.2, random_state=42)

# Build model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(INPUT_SIZE, INPUT_SIZE, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')  # Binary classifier: 1 = valid MRI, 0 = invalid
])

model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=10, validation_data=(val_images, val_labels))

model.save('model/valid_mri_64x64.h5')
print("✅ Saved: model/valid_mri_64x64.h5")
