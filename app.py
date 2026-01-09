from flask import Flask, render_template, redirect, request
import tensorflow as tf

app = Flask(__name__)
app.config["SECRET_KEY"] = "uytrfdcvbnjuytresdes"

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