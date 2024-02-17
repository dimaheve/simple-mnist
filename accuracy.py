import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np

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
model.load_state_dict(torch.load('mnist_model.pth'))
model.eval()  

transform = transforms.Compose([transforms.ToTensor(), 
                                transforms.Normalize((0.5,), (0.5,))])
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def calculate_accuracy(loader):
    correct = 0
    total = 0
    with torch.no_grad():  
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

accuracy = calculate_accuracy(test_loader)
print(f'Accuracy of the model on the test images: {accuracy * 100}%')
