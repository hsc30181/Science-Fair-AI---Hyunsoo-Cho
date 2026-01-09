from flask import Flask, render_template, redirect, request
import tensorflow as tf
import numpy as np
from PIL import Image

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

@app.route("/images_trained_<model>")
def model_selection():
    return render_template("model_test.html", model = model)

@app.route("/model_test")
def model_test():
    data = request.get_json()
    model = data["model"]
    tf.keras.models.load_model(f"/models/{model}")
