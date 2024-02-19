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

def one_hot_encode(y, num_classes=10):
    return np.eye(num_classes)[y]

def save_model_weights(model, filename="./data/model_weights.npz"):
    np.savez(filename, **model.params)

def load_model_weights(model, filename="./data/model_weights.npz"):
    with np.load(filename) as data:
        params = {key: data[key] for key in data.files}
    model.params = params

def calculate_accuracy(model, X_test, Y_test):
    predictions = model.forward_propagation(X_test)
    correct_predictions = np.argmax(predictions, axis=1) == np.argmax(Y_test, axis=1)
    accuracy = np.mean(correct_predictions)
    return accuracy

def train_model(learning_rate, epochs, batch_size, n_batches, model, train_mnist, train_mnist_labels, filename):
    for epoch in range(epochs):
        for i in range(n_batches):
            start = i * batch_size
            end = start + batch_size
            X, Y = train_mnist[start:end], train_mnist_labels[start:end]
            grads = model.backward_propagation(X, Y)
            model.update_params(grads, learning_rate)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {model.compute_loss(model.forward_propagation(train_mnist), train_mnist_labels)}")
    save_model_weights(model, filename)

train_mnist = read_images('./data/train_mnist'); train_mnist = train_mnist.reshape(train_mnist.shape[0], -1).astype(np.float32) / 255.0
train_mnist_labels = one_hot_encode(read_labels('./data/train_mnist_labels'))
test_mnist = read_images('./data/test_mnist'); test_mnist = test_mnist.reshape(test_mnist.shape[0], -1).astype(np.float32) / 255.0
test_mnist_labels = one_hot_encode(read_labels('./data/test_mnist_labels'))

class MNIST_NN:
    def relu(self, Z, derivative=False):
        if not derivative:
            return np.maximum(0, Z)

        return np.where(Z > 0, 1, 0)
    
    def softmax(self, Z, derivative=False):
        exps = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        softmax_out = exps / np.sum(exps, axis=1, keepdims=True)
        
        if not derivative:
            return softmax_out
        else:
            return 1

    def cross_entropy_loss(self, Y_pred, Y_true, derivative=False):
        m = Y_true.shape[0]
        Y_pred = np.clip(Y_pred, 1e-12, 1 - 1e-12)
        if not derivative:
            return -np.sum(Y_true * np.log(Y_pred) + (1 - Y_true) * np.log(1 - Y_pred)) / m
        else:
            return - (Y_true / Y_pred - (1 - Y_true) / (1 - Y_pred))
        
    def compute_loss(self, Y_pred, Y_true, derivative=False):
        return self.loss_function(Y_pred, Y_true, derivative)
    
    def init_dA(self, Y, output):
        if self.loss_function == self.cross_entropy_loss and self.layers[-1]["activation"] == self.softmax:
            return output - Y
        else:
            raise ValueError("Unknown combination of loss function and output activation function.")

    def initialize_activations(self):
        for layer in self.layers:
            try:
                layer["activation"] = getattr(self, layer["activation"])
            except AttributeError:
                raise ValueError(f"Unknown activation function: {layer['activation']}")

    def initialize_params(self):
        self.params = {}
        for i, layer in enumerate(self.layers):
            init_method = np.sqrt(2. / layer["input_dim"]) if layer["activation"] == self.relu else np.sqrt(1. / layer["input_dim"]) 
            self.params[f"W{i+1}"] = np.random.randn(layer["input_dim"], layer["output_dim"]) * init_method
            self.params[f"b{i+1}"] = np.zeros((1, layer["output_dim"]))

    def __init__(self, layers, loss_function="cross_entropy_loss") -> None:
        self.layers = layers
        self.loss_function = getattr(self, loss_function)
        self.initialize_activations()
        self.initialize_params()
    
    def forward_propagation(self, X, return_activations=False):
        A = X  
        activations = {"A0": A, "Z0": "X"}

        for i, layer in enumerate(self.layers):
            Z = np.dot(A, self.params[f"W{i+1}"]) + self.params[f"b{i+1}"]
            A = layer["activation"](Z)

            if return_activations:
                activations[f"Z{i+1}"] = Z
                activations[f"A{i+1}"] = A

        if return_activations:
            return A, activations
        return A
    
    def backward_propagation(self, X, Y):
        grads = {}
        A, activations = self.forward_propagation(X, return_activations=True)
        m = X.shape[0]
        dA = self.init_dA(Y, A)

        for i in reversed(range(len(self.layers))):
            dZ = dA * self.layers[i]["activation"](activations[f"Z{i+1}"], derivative=True)
            dW = np.dot(activations[f"A{i}"].T, dZ) / m
            db = np.sum(dZ, axis=0, keepdims=True) / m
            dA = np.dot(dZ, self.params[f"W{i+1}"].T)

            grads[f"dW{i+1}"] = dW
            grads[f"db{i+1}"] = db

        return grads
    
    def update_params(self, grads, learning_rate):
        n_layers = len(self.layers)
        for i in range(1, n_layers + 1):
            self.params[f"W{i}"] -= learning_rate * grads[f"dW{i}"]
            self.params[f"b{i}"] -= learning_rate * grads[f"db{i}"]
            
model = MNIST_NN([
    {"input_dim": 28*28, "output_dim": 128, "activation": "relu"},
    {"input_dim": 128, "output_dim": 64, "activation": "relu"},
    {"input_dim": 64, "output_dim": 10, "activation": "softmax"}
])

load_model_weights(model, "./data/model_weights_:).npz")
print("Model initialized.")
learning_rate = 0.01
epochs = 20
batch_size = 64
n_batches = train_mnist.shape[0] // batch_size

#train_model(learning_rate, epochs, batch_size, n_batches, model, train_mnist, train_mnist_labels, "./data/model_weights_:).npz")

accuracy = calculate_accuracy(model, test_mnist, test_mnist_labels)
print(f"Test Accuracy: {accuracy * 100}%")