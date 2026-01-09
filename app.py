from flask import Flask, render_template, redirect, request
import tensorflow as tf
import numpy as np
from PIL import Image
import base64
from io import BytesIOpre

app = Flask(__name__)
app.config["SECRET_KEY"] = "uytrfdcvbnjuytresdes"

IMG_SIZE = 224

def preprocess_image(img_path):
    img = Image.open(img_path).convert("RGB")
    img.resize((IMG_SIZE, IMG_SIZE))
    img = np.array(img) / 255.0
    return np.expand_dims(img, axis=0)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/model_<model>")
def model_selection():
    return render_template("model_test.html", model = model)

@app.route("/model_test_<model>")
def model_test():
    file = request.files["image"]
    img_bytes = file.read()
    img_base64 = base64.b64encode(img_bytes).decode("utf-8")
    mime_type = file.content_type
