
# CS771 Captcha Cracking Project

## Overview

This project demonstrates how to build, train, and deploy a deep learning solution for automatic captcha digit recognition using Convolutional Neural Networks (CNNs). The objective is to classify captcha images representing hexadecimal characters (0–9, A–F) as either **EVEN** or **ODD** digits.

---

## Project Structure

```
├── train/         # Training images of captcha characters
├── reference/     # Reference images for validation or visualization
├── test/          # Images for testing predictions
├── model.h5       # Trained CNN model in Keras HDF5 format
├── predict.py     # Script to preprocess images and predict captcha class
├── README.md      # Project description (this file)
```

---

## Data Preparation

- **Training Images**: All samples in `train/` are labeled by filename (e.g., "0.png").
- **Preprocessing Steps**:
    - Each image is cropped to focus on relevant character region (coordinates: 355, 0, 455, 100).
    - Converted to grayscale.
    - Scaled to 100×100 pixels.
    - Pixel values normalized to range [0, 1].
    - Reshaped for model input (100, 100, 1).

---

## Model Architecture and Summary

The model is built using Keras (TensorFlow backend) as a **Sequential CNN**:

| Layer (type)        | Output Shape         | Parameters    | Activation |
|---------------------|---------------------|---------------|------------|
| Conv2D              | (98, 98, 32)        | 320           | relu       |
| MaxPooling2D        | (49, 49, 32)        | 0             | —          |
| Conv2D_1            | (47, 47, 64)        | 18,496        | relu       |
| MaxPooling2D_1      | (23, 23, 64)        | 0             | —          |
| Flatten             | (33856,)            | 0             | —          |
| Dense               | (64,)               | 2,166,848     | relu       |
| Dense_1 (output)    | (16,)               | 1,040         | softmax    |

**Total parameters:** 2,186,706 (all trainable)

**Layer Explanations:**
- **Conv2D layers**: Extract features like edges and shapes with ReLU activation.
- **MaxPooling2D layers**: Reduce spatial dimensions, keeping essential features.
- **Flatten**: Converts multidimensional feature maps to a 1D vector.
- **Dense (64 units)**: Learns patterns from extracted features, uses ReLU for non-linearity.
- **Dense_1 (16 units, softmax)**: Outputs probabilities for each hexadecimal class.

---

## Training

- The model is trained using **Adam optimizer** and **sparse categorical cross-entropy loss**.
- Validation split is used to monitor training progress and prevent overfitting.
- After training, the model weights and architecture are saved in `model.h5`.

---

## Inference Flow

### 1. **Preprocessing Images**
- Crop, grayscale, resize, normalize, reshape to (100, 100, 1).

### 2. **Prediction**
- Load `model.h5`.
- Feed preprocessed image to model; model returns probability vector for 16 classes.
- Take `np.argmax` of output to select predicted class.
- Map predicted class index to "EVEN" or "ODD" using:

```
label_mapping = {
  0: 'EVEN',  1:'ODD', 2:'EVEN', 3:'ODD',
  4: 'EVEN',  5:'ODD', 6:'EVEN', 7:'ODD',
  8: 'EVEN',  9:'ODD', 10:'EVEN', 11:'ODD',
  12:'EVEN', 13:'ODD', 14:'EVEN', 15:'ODD'
}
```

### 3. **Batch Prediction**
- Iterate through a list of image filenames and return a list of predicted labels.

---

## Usage Example

```
from PIL import Image
import numpy as np
import tensorflow as tf

# Load trained model
model = tf.keras.models.load_model("model.h5")

def preprocess_image(image_path):
    image = Image.open(image_path)
    image = image.crop((355, 0, 455, 100)).convert("L")
    image = image.resize((100, 100))
    image = np.array(image) / 255.0
    image = image.reshape(-1, 100, 100, 1)
    return image

def predict_character(image_path):
    image = preprocess_image(image_path)
    prediction = model.predict(image)
    predicted_class_index = np.argmax(prediction)
    label_mapping = {...}  # See mapping above
    predicted_class = label_mapping[predicted_class_index]
    return predicted_class

def decaptcha(filenames):
    return [predict_character(file) for file in filenames]
```

---


## Notes & Best Practices

- Check filenames and cropping coordinates match your dataset structure.
- Always normalize and resize image appropriately before feeding to the model.
- Ensure your train and test images cover all possible digit classes.
- For batch predictions, ensure the function returns a **list of predictions** (not just the last value).

---

## Troubleshooting

- **Model only predicts one class:** Check class balance in training data and preprocessing output shape.
- **Shape errors:** Confirm images are reshaped as `(batch_size, 100, 100, 1)`.
- **Colab file path errors:** Always verify image and model paths match Colab's directory structure.

---

## Contact & Credits

Project by: [Krrish Khandelwal] | Course Project: CS771 |  Prof. Purushottam Kar (CSE Department, IITK)



```
