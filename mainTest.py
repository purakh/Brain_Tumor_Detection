import cv2
from keras.models import load_model
from PIL import Image
import numpy as np

model = load_model('BrainTumor10EpochsCategorical.h5')

image = cv2.imread('C:\\Users\\Purushottam Singh\\Desktop\\Brain Tumor Detection Project\\pred\\pred5.jpg')

img = Image.fromarray(image)
img = img.resize((64, 64))
img = np.array(img)

input_img = np.expand_dims(img, axis=0)

# Use predict instead of predict_classes
result = model.predict(input_img)
predicted_class = np.argmax(result, axis=1)

print(predicted_class)
