import tensorflow as tf
import numpy as np
from PIL import Image
import base64
from io import BytesIO

IMG_SIZE = 224

def preprocess_image(img_path):
    img = Image.open(img_path).convert("RGB")
    img.resize((IMG_SIZE, IMG_SIZE))
    img = np.array(img) / 255.0
    return np.expand_dims(img, axis=0)
