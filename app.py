import tensorflow as tf
from keras.models import load_model
import numpy as np
from PIL import Image
import base64
from io import BytesIO
import os
import sys

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
classes = ["class_chair", "class_computer", "class_tv", "class_tshirt", "class_beaker"]
for class_name in classes:
    images = []
    for i in range(10):
        image_count = i
        image_name = f"test{i}.png"
        path = os.path.join("class_name", class_name, image_name)
        images.append(image_name)
    for image in images:
        print(f"Accuracy for Model Fifty, Class {class_name}: {calculate_accuracy(model_fifty, true_labels, image)}")
    for image in images:
        print(f"Accuracy for Model Hundred, Class {class_name}: {calculate_accuracy(model_hundred, true_labels, image)}")
    for image in images:
        print(f"Accuracy for Model Hund_Fif, Class {class_name}: {calculate_accuracy(model_hund_fif, true_labels, image)}")

print("Test ended.")
sys.exit()