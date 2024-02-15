import numpy as np
import struct

def read_images(filename):
    with open(filename, 'rb') as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        images = np.fromfile(f, dtype=np.uint8).reshape(num, rows, cols)
    return images

def read_labels(filename):
    with open(filename, 'rb') as f:
        magic, num = struct.unpack(">II", f.read(8))
        labels = np.fromfile(f, dtype=np.uint8)
    return labels

train_images = read_images('./data/train_mnist')
train_labels = read_labels('./data/train_mnist_labels')

def ReLU(z):
    return np.maximum(0, z)

def dReLU(z):
    return np.where(z > 0, 1, 0)

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

class NeuralNetwork:
    def __init__(self) -> None:
        self.weights = [
            np.random.randn(32, 784),
            np.random.randn(32, 32),
            np.random.randn(10, 32)
        ]

        self.biases = [
            np.random.randn(32),
            np.random.randn(32),
            np.random.randn(10)
        ]

        self.activations = [
            np.zeros(32),
            np.zeros(32),
            np.zeros(10)
        ]

    def forward(self, x) -> np.ndarray:
        for i in range(len(self.weights)):
            if i == 0:
                preactivated_neuron = np.dot(self.weights[i], x) + self.biases[i]
                activated_neuron = ReLU(preactivated_neuron)
                self.activations[i] = {"preactivated_neuron": preactivated_neuron, "activated_neuron": activated_neuron}
                continue
            else:
                preactivated_neuron = np.dot(self.weights[i], self.activations[i - 1]["activated_neuron"]) + self.biases[i]
                activated_neuron = ReLU(preactivated_neuron)
                self.activations[i] = {"preactivated_neuron": preactivated_neuron, "activated_neuron": activated_neuron}
                continue

        return softmax(self.activations[len(self.weights)-1]["activated_neuron"])
    
    def backwards(self, x, y) -> np.ndarray:
        actual = np.zeros(10); actual[y] = 1
        predicted = self.forward(x)
        loss = predicted - actual

        gradients = {
            "weights": [np.zeros_like(w) for w in self.weights],
            "biases": [np.zeros_like(b) for b in self.biases]
        }

        for i in range(gradients["weights"][2].shape[0]):
            up_gradient = dReLU(self.activations[2]["preactivated_neuron"][i]) * loss[i]
            gradients["biases"][2][i] = up_gradient
            for j in range(gradients["weights"][2].shape[1]):
                gradients["weights"][2][i, j] = up_gradient * self.activations[1]["activated_neuron"][j]
        
        return gradients



nn = NeuralNetwork()
print(nn.forward(train_images[0, :, :].reshape(784) / 255))
print(train_labels[0])
print(nn.activations[2]["preactivated_neuron"][0])
print(nn.backwards(train_images[0, :, :].reshape(784) / 255, train_labels[0])["biases"][2])