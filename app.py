from flask import Flask, render_template, request, jsonify
import base64, io
from PIL import Image
import numpy as np

app = Flask(__name__)

# ===========================
# 👉  plug in your algorithm here
# ===========================
def classify_image(img_arr_28x28):
    """
    img_arr_28x28 : np.ndarray shape (28, 28), values 0–1
    return: (predicted_label:int, probabilities:list[float])
    """
    # EXAMPLE dummy softmax placeholder
    probs = np.random.rand(10)
    probs = np.exp(probs) / np.sum(np.exp(probs))
    pred = int(np.argmax(probs))
    return pred, probs.tolist()

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

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
