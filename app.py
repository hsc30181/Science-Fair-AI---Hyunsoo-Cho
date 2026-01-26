import tensorflow as tf
from keras.models import load_model
import numpy as np
from PIL import Image
import base64
from io import BytesIO

IMG_SIZE = 224

model_fifty = load_model("/models/fifty.h5")
model_hundred = load_model("/models/hundred.h5")
model_hund_fif = load_model("/models/hund_fif.h5")

true_labels = [0, 1, 2, 3, 4]
image_paths = []

def preprocess_image(img_path):
    img = Image.open(img_path).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))      
    img = np.array(img) / 255.0
    return np.expand_dims(img, axis=0)

def calculate_accuracy(model, true_labels, image_path):
    preds = []
    confs = []
    for img_path, true_label in zip(image_path, true_labels):
        predicted_class, confidence = test_model(img_path, model)
        preds.append(predicted_class)
        confs.append(confidence)
    preds = np.array(preds)
    confs = np.array(confs)
    true_labels = np.array(true_labels)
    correct = (preds == true_labels).astype(float)
    weighted_accuracy = np.sum(correct * confs) / len(image_path)
    return weighted_accuracy

def test_model(image_path, model):
    img = preprocess_image(image_path)
    y_pred = model.predict(img)
    predicted_class = int(np.argmax(y_pred, axis=1)[0])
    confidence = float(np.max(y_pred))
    return predicted_class, confidence

"""
Class 0: Chair,
Class 1: Computer,
Class 2: TV,
Class 3: T-shirt,
Class 4: Beaker
"""

for test in range(100):
    pass