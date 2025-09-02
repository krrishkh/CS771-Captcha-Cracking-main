#harshitptl21
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
model = tf.keras.models.load_model("model.h5")
def preprocess_image(image_path):
    image = Image.open(image_path)
    image = image.crop((355, 0, 455, 100)).convert("L")
    image = np.array(image) / 255.0
    image = image.reshape(-1, 100, 100, 1)
    return image
def predict_character(image_path):
    image = preprocess_image(image_path)
    prediction = model.predict(image)
    predicted_class_index = np.argmax(prediction)
    label_mapping = {0: 'EVEN', 1: 'ODD', 2: 'EVEN', 3: 'ODD', 4: 'EVEN', 5: 'ODD', 6: 'EVEN', 7: 'ODD', 8: 'EVEN', 9: 'ODD', 10: 'EVEN', 11: 'ODD', 12: 'EVEN', 13: 'ODD', 14: 'EVEN', 15: 'ODD'}
    predicted_class = label_mapping[predicted_class_index]
    return predicted_class
def decaptcha( filenames ):
    output = []
    for file in filenames:
        predicted_character = predict_character(file)
        output.append(predicted_character)
    return predicted_character
