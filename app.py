import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input
import numpy as np
import json, os
from flask import Flask, request, jsonify, render_template
from PIL import Image

app = Flask(__name__)

model = tf.keras.models.load_model(
    "garbage_efficientnet_final.keras",
    custom_objects={"preprocess_input": preprocess_input}
)

with open("class_names.json") as f:
    class_names = json.load(f)

IMG_SIZE = 224

def predict_image(img_path):
    img = Image.open(img_path).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img = np.array(img, dtype=np.float32)
    img = np.expand_dims(img, axis=0)
    probs = model.predict(img, verbose=0)[0]
    class_idx  = int(np.argmax(probs))
    confidence = float(probs[class_idx])
    label      = class_names[class_idx]
    top3_idx = np.argsort(probs)[::-1][:3]
    top3 = [(class_names[i], float(probs[i])) for i in top3_idx]

    return label, confidence, top3

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["file"]
    path = "temp.jpg"
    file.save(path)
    label, confidence, top3 = predict_image(path)
    return jsonify({"label": label, "confidence": confidence, "top3": top3})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3000)
