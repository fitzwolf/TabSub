import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import cv2
import numpy as np



# Load EMNIST dataset
print("Loading dataset")
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

trainset = torchvision.datasets.EMNIST(root='./data', split='byclass', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=True, num_workers=2)

testset = torchvision.datasets.EMNIST(root='./data', split='byclass', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

# Define the neural network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 62) # 62 classes in EMNIST byclass

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# Train the network
print("Beginning training")
num_epochs = 10
for epoch in range(num_epochs):
    print(f"Entering epoch: {epoch}")
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {running_loss / (i+1)}")

print("Training completed.")

# Test the network
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Accuracy: {100 * correct / total}%")

def preprocess_and_predict(images, model):
    # Preprocess images
    preprocessed_images = []
    for image in images:
        # Resize the image to 28x28
        resized_image = cv2.resize(image, (28, 28))
        
        # Normalize the image
        normalized_image = (resized_image / 255.0 - 0.5) / 0.5
        
        # Convert the image to PyTorch tensor format
        tensor_image = torch.from_numpy(normalized_image).unsqueeze(0).unsqueeze(0).float()
        preprocessed_images.append(tensor_image)
    
    # Predict characters
    predictions = []
    model.eval()
    with torch.no_grad():
        for image_tensor in preprocessed_images:
            output = model(image_tensor)
            _, predicted = torch.max(output.data, 1)
            predictions.append(predicted.item())
    
    return predictions


# Load images
image_paths = ["cropped_images/imageA.png", "cropped_images/image0.png", 
               "cropped_images/imageB.png", "cropped_images/imageC.png"]
images = [cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in image_paths]

# Predict characters
predictions = preprocess_and_predict(images, net)
print("Predicted characters:", predictions)


idx_to_class = {v: k for k, v in trainset.class_to_idx.items()}
predicted_characters = [idx_to_class[idx] for idx in predictions]
print("Predicted characters:", predicted_characters)
