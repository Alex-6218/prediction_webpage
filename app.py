from flask import Flask, render_template, request, jsonify
import base64, io
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
app = Flask(__name__)

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
    predictions = predictions / np.sum(predictions, axis=1, keepdims=True)  # shape (m, K)
    predicted_label = np.argmax(predictions, axis=1)[0]
    confidence = predictions[0, predicted_label]
    return int(predicted_label), predictions.tolist()

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
    # Debug: save the received image to disk for inspection

    # Save as JPEG for easier viewing
    debug_path = "MNIST_Data/testSample/debug_latest.jpg"
    img_debug = Image.open(io.BytesIO(img_bytes)).convert("RGB").resize((28, 28))
    img_debug.save(debug_path, format="JPEG")

    img = Image.open(io.BytesIO(img_bytes)).convert("L")
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = 1.0 - arr  # Invert: black strokes become white, white background becomes black

    # --- Digit centering preprocessing ---
    # Threshold to binary image
    bw = (arr > 0.1).astype(np.uint8)
    coords = np.argwhere(bw)
    if coords.size > 0:
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0) + 1
        arr_cropped = arr[y0:y1, x0:x1]
        # Pad to square
        h, w = arr_cropped.shape
        size = max(h, w)
        pad_y = (size - h) // 2
        pad_x = (size - w) // 2
        arr_square = np.pad(arr_cropped, ((pad_y, size - h - pad_y), (pad_x, size - w - pad_x)), mode='constant')
    # Resize to 20x20, then paste into center of 28x28
    from PIL import Image as PILImage
    digit_img = PILImage.fromarray((arr_square * 255).astype(np.uint8)).resize((20, 20), PILImage.BILINEAR)
    arr_img = PILImage.new('L', (28, 28), color=0)
    arr_img.paste(digit_img, (4, 4))
    arr = np.array(arr_img, dtype=np.float32) / 255.0
    # Save the preprocessed image for inspection
    img_number = len(os.listdir("MNIST_Data/userInput/"))
    arr_img.save("MNIST_Data/userInput/img_" + str(img_number+1) + ".jpg", format="JPEG")

    pred, probs = classify_image(arr)
    return jsonify({"prediction": pred, "probs": probs})

@app.route("/test")
def test():
    return "<h1>Hello from Flask</h1>"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
