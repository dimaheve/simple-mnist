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

class MNIST_NN:
    def relu(self, Z):
        return np.maximum(0, Z)
    
    def softmax(self, Z):
        Z_exp = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        sum_Z_exp = np.sum(Z_exp, axis=1, keepdims=True)
        softmax_Z = Z_exp / sum_Z_exp
        return softmax_Z

    def initialize_activations(self):
        for layer in self.layers:
            if layer["activation"] == "relu":
                layer["activation"] = self.relu
            elif layer["activation"] == "softmax":
                layer["activation"] = self.softmax

    def initialize_params(self):
        self.params = {}
        for i, layer in enumerate(self.layers):
            init_method = np.sqrt(2. / layer["input_dim"]) if layer["activation"] == self.relu else np.sqrt(1. / layer["input_dim"]) 
            self.params[f"W{i+1}"] = np.random.randn(layer["input_dim"], layer["output_dim"]) * init_method
            self.params[f"b{i+1}"] = np.zeros((1, layer["output_dim"]))

    def __init__(self, layers) -> None:
        self.layers = layers
        self.initialize_activations()
        self.initialize_params()
    
    def forward_propagation(self, X, return_activations=False):
        A = X  
        activations = {}

        for i, layer in enumerate(self.layers):
            Z = np.dot(A, self.params[f"W{i+1}"]) + self.params[f"b{i+1}"]
            A = layer["activation"](Z)

            if return_activations:
                activations[f"Z{i+1}"] = Z
                activations[f"A{i+1}"] = A

        if return_activations:
            return A, activations
        return A
    

#       [
#           {"input_dim": 28*28, "output_dim": 128, "activation": self.relu},
#           {"input_dim": 128, "output_dim": 64, "activation": self.relu},
#           {"input_dim": 64, "output_dim": 10, "activation": self.softmax}
#       ]
