import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import random

class SimpleNN(torch.nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.flatten = torch.nn.Flatten()
        self.fc1 = torch.nn.Linear(28*28, 128)
        self.fc2 = torch.nn.Linear(128, 64)    
        self.fc3 = torch.nn.Linear(64, 10)     

    def forward(self, x):
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)  
        return x

model = SimpleNN()
model.load_state_dict(torch.load('./data/mnist_model.pth'))
model.eval() 

transform = transforms.Compose([transforms.ToTensor(), 
                                transforms.Normalize((0.5,), (0.5,))])
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def display_random_image_and_prediction(loader):
    images, labels = next(iter(loader))
    images, labels = images.to(device), labels.to(device)
    output = model(images)
    _, predicted = torch.max(output, 1)

    plt.imshow(images.cpu().squeeze(), cmap="gray")
    plt.title(f"Actual: {labels.item()}, Predicted: {predicted.item()}")
    plt.show()

# Display a random image and prediction
display_random_image_and_prediction(test_loader)
