import numpy as np
import os
import sys
from PIL import Image
from keras.models import load_model

IMG_SIZE = 224

# ===============================
# Load models
# ===============================
model_fifty = load_model("models/fifty.keras")
model_hundred = load_model("models/hundred.keras")
model_hund_fif = load_model("models/hund_fif.keras")

# ===============================
# Classes
# ===============================
classes = [
    "class_chair",
    "class_computer",
    "class_airplane",
    "class_boat",
    "class_beaker"
]

# ===============================
# Image preprocessing
# ===============================
def preprocess_image(img_path):
    img = Image.open(img_path).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(img, axis=0)

# ===============================
# Single image inference
# ===============================
def test_model(image_path, model):
    img = preprocess_image(image_path)
    y_pred = model.predict(img, verbose=0)

    predicted_class = int(np.argmax(y_pred))
    confidence = float(np.max(y_pred))

    return predicted_class, confidence

# ===============================
# Weighted accuracy
# ===============================
def calculate_weighted_accuracy(model, image_paths, true_label):
    weighted_sum = 0.0

    for path in image_paths:
        pred, conf = test_model(path, model)
        if pred == true_label:
            weighted_sum += conf

    return weighted_sum / len(image_paths)

# ===============================
# Evaluation
# ===============================
MODELS = {
    "Model Fifty": model_fifty,
    "Model Hundred": model_hundred,
    "Model Hund_Fif": model_hund_fif
}

for class_idx, class_name in enumerate(classes):
    print(f"\n===== Class: {class_name} =====")

    image_paths = []
    for i in range(10):
        image_paths.append(
            os.path.join(class_name, f"test{i}.jpg")
        )

    for model_name, model in MODELS.items():
        w_acc = calculate_weighted_accuracy(
            model, image_paths, class_idx
        )
        print(f"{model_name} Weighted Accuracy: {w_acc:.4f}")

print("\nTest ended.")
sys.exit()
