import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import sys, os

test_file = ''
K = 10
n = 784
epochs = 0
inputX = np.zeros((1, n))   
labelsY = np.zeros((1, K))
if os.path.isfile("softmax_params.npz"):
    print("Loading existing parameters from softmax_params.npz")
    params = np.load("softmax_params.npz")
    weightsT = params["Theta"]
    biasB = params["b"]
else:
    print("No existing parameters found. Please train the model first.")
    sys.exit(1)

def prediction(x, weights):
    #create logits
    logitsZ = x @ weights + biasB # shape (m, K)
    predictions = np.array([np.clip(np.exp(logitsZ[logit, :] - np.max(logitsZ[logit, :])), 0, None) for logit in range(logitsZ.shape[0])]) #Softmax logits to get prediction vector of shape (m, K)
    return predictions / np.sum(predictions, axis=1, keepdims=True)  # shape (m, K)

def predict():
    print("Please provide a test image number (between 1 and 28000): ")
    test_image_number = input().strip()
    if not test_image_number.isdigit():
        print(f"Invalid input: {test_image_number} is not a number.")
    else:
        test_image_number = int(test_image_number)
        test_file = f"./MNIST_Data/testSet/testSet/img_{test_image_number}.jpg"
        if not os.path.isfile(test_file):
            print(f"File not found: {test_file}")
            sys.exit(1)
        img = Image.open(test_file).convert("L")  # grayscale
        img = img.resize((28,28))                 # force 28x28 if needed
        arr = np.array(img).astype(np.float32) / 255.0
        testX = arr.flatten().reshape(1, -1)      # flatten 28x28 → 784 and reshape to (1, 784)

        pred = prediction(testX, weightsT)
        predicted_label = np.argmax(pred, axis=1)[0]
        confidence = pred[0, predicted_label]

        print(f"Predicted label: {predicted_label} with confidence {confidence:.4f}")

        plt.imshow(arr, cmap='gray')
        plt.title(f"Predicted: {predicted_label} (Confidence: {confidence:.4f})")
        plt.axis('off')
        plt.show()

if __name__ == "__main__":
    predict()