from flask import Flask, render_template, request, jsonify
import base64, io
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
app = Flask(__name__)

# ===========================
# 👉  plug in your algorithm here
# ===========================
K = 10
n = 784
epochs = 0
inputX = np.zeros((1, n))   
labelsY = np.zeros((1, K))

def classify_image(img_arr_28x28):
    """
    img_arr_28x28 : np.ndarray shape (28, 28), values 0–1
    return: (predicted_label:int, probabilities:list[float])
    """
    params = np.load("softmax_params.npz")
    weightsT = params["Theta"]
    biasB = params["b"]
    x = img_arr_28x28.flatten().reshape(1, -1)  # shape (1, 784)

    logitsZ = x @ weightsT + biasB # shape (m, K)
    predictions = np.array([np.clip(np.exp(logitsZ[logit, :] - np.max(logitsZ[logit, :])), 0, None) for logit in range(logitsZ.shape[0])]) #Softmax logits to get prediction vector of shape (m, K)
    predicted_label = np.argmax(predictions, axis=1)[0]
    confidence = predictions[0, predicted_label]
    return predictions / np.sum(predictions, axis=1, keepdims=True), predicted_label, confidence

# ===========================
#  Routes
# ===========================
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/classify", methods=["POST"])
def classify():
    data = request.json
    b64data = data["image"]
    header, encoded = b64data.split(",", 1)
    img_bytes = base64.b64decode(encoded)
    img = Image.open(io.BytesIO(img_bytes)).convert("L").resize((28, 28))
    arr = np.array(img, dtype=np.float32) / 255.0

    pred, probs = classify_image(arr)
    return jsonify({"prediction": pred, "probs": probs})

@app.route("/test")
def test():
    return "<h1>Hello from Flask</h1>"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
