import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import sys, os

train_files_path = ''
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

#Load images from the training set folder and modifying them as necessary for use in training
# this was the weirdest part of this project for me lol
def load_images_from_folder(folder, img_size=(28,28)):
    data, labels = [], []
    for label in range(K):
        path = os.path.join(folder, str(label))
        print(f"Loading images from: {path}")
        for filename in os.listdir(path):
            if filename.endswith(".jpg"):
                print(f"Processing file: {filename}")
                img_path = os.path.join(path, filename)
                img = Image.open(img_path).convert("L")  # grayscale
                img = img.resize(img_size)               # force 28x28 if needed
                arr = np.array(img).astype(np.float32) / 255.0
                data.append(arr.flatten())               # flatten 28x28 → 784

                #create one-hot encoded label
                label_one_hot = np.zeros(K) 
                label_one_hot[label] = 1
                labels.append(label_one_hot)
    return np.array(data), np.array(labels)

#user input to determine if they want to train or test
choice = input("Press Enter for test or 't' for train: ").strip().lower()
if choice == 't':
    weightsT = np.random.randn(n, K) * 0.01  # small random values
    biasB = np.zeros((1, K))                 # initialize biases to zero
    epochs = 0
    inputX, labelsY = load_images_from_folder('./MNIST_Data/trainingSet/trainingSet')


def prediction(x, weights):
    #create logits
    logitsZ = x @ weights + biasB # shape (m, K)
    predictions = np.array([np.clip(np.exp(logitsZ[logit, :] - np.max(logitsZ[logit, :])), 0, None) for logit in range(logitsZ.shape[0])]) #Softmax logits to get prediction vector of shape (m, K)
    return predictions / np.sum(predictions, axis=1, keepdims=True)  # shape (m, K)

def grad_ThetaT(x, y, weights):
    gradient = np.zeros(weights.shape)  # shape (n, K)
    p = prediction(x, weights)
    gradient += x.T @ (p - y) # cross multiply images by difference of prediction and label, gives shape (n, K)
    return gradient / x.shape[0] # average over all m examples

def grad_B(x, y):
    gradient = np.zeros(biasB.shape)  # shape (1, K)
    p = prediction(x, weightsT)                  # shape (m, K)
    gradient = np.mean(p - y, axis=0, keepdims=True)  # shape (1, K)
    return gradient

def train():
    global inputX, labelsY, weightsT, biasB, epochs, choice
    #calculate number of batches to make
    num_batches = inputX.shape[0] // 64
    lr = 0.06 # learning rates for bias and weights

    print(f"Number of batches: {num_batches}")
    epoch_loss = 0
    max_epochs = 10

    #create minibatches of size 64 and train on each minibatch
    while epochs < max_epochs:
        for i in range(num_batches):
            minibatch = inputX[64*i:64*(i+1), :]
            #create corresponding labels for minibatch
            minibatch_labels = np.zeros((64, K))
            for l in range(len(minibatch)):
                minibatchlabel = labelsY[64*i + l, :]
                minibatch_labels[l, :] = minibatchlabel


            #calculate error for the minibatch with current weight matrix
            p = prediction(minibatch, weightsT)
            batch_loss = -np.mean(np.sum(minibatch_labels * np.log(p + 1e-12), axis=1))
            #adjust weight based on gradient
            weightsT -= lr * grad_ThetaT(minibatch, minibatch_labels, weightsT)
            #adjust bias based on gradient
            biasB -= lr * grad_B(minibatch, minibatch_labels)
            print(f"Minibatch {i+1}/{num_batches}: shape {minibatch.shape}, batch loss {batch_loss}")
            print(f"  Epoch {epochs}/{max_epochs}, epoch loss: {epoch_loss}")

        #increment epoch, save trained parameters, and randomize data for retraining
        shuffled_indices = np.random.permutation(len(inputX))
        inputX = inputX[shuffled_indices]
        labelsY = labelsY[shuffled_indices]
        

        np.savez("softmax_params.npz", Theta=weightsT, b=biasB)
        epoch_loss += batch_loss
        epochs += 1

#handle testing and displaying results
#this was like 70 percent ai generated lol sorry guys
if choice == 't':
    train()
else:
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