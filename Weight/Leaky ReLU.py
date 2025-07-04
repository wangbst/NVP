import os
import pathlib
from datetime import datetime
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import wandb
import numpy as np
import torch.optim as optim
import torch.nn.utils as utils
import math
import torch.nn.utils.prune as prune  # Added import for weight pruning

# Define Cutout augmentation class
class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes=1, length=16):
 
    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),  
        transforms.RandomHorizontalFlip(),  
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        Cutout(n_holes=1, length=16)  
 
transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
                   

batch_size = 128

data_path = 'data'

trainset = torchvision.datasets.CIFAR10(root=data_path, train=True,
                                        download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=1)

testset = torchvision.datasets.CIFAR10(root=data_path, train=False,
                                       download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=1)

classes = 10

# Define the model
model = ResNet18()

# Move the model to the device
model = model.to(device)

import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim

# Configuration
learning_rate = 0.005
momentum = 0.9
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum,weight_decay=5e-4)

indices = [2, 5, 8, 11]
conv_modules = [module for module in model.modules() if isinstance(module, nn.Conv2d)]

entropies = [[] for _ in range(len(conv_modules))]
for idx in indices:
    entropies[idx] = [[] for _ in range(conv_modules[idx].out_channels)]

class ConvEntropyHook:
    def __init__(self, module, idx):
        
    def hook_fn(self, module, input, output):
        
    def close(self):
        self.hook.remove()


hooks = []
for idx in indices:
    hooks.append(ConvEntropyHook(conv_modules[idx], idx))
    
class LayerEntropyHook:
    def __init__(self, module, layer_name):
       
    def hook_fn(self, module, input, output):
        # Add a small epsilon to avoid log(0) issues
        
    def close(self):
        self.hook.remove()

layer_names = ['prepare'] + [f'layer{i}' for i in range(1, 5)] + ['fc']
entropy_hooks = [LayerEntropyHook(getattr(model, layer_name), name) for layer_name, name in zip(layer_names, layer_names)]

# Define a function to calculate entropy
def calculate_entropy(tensor):
    epsilon = 1e-10
    tensor = torch.abs(tensor) + epsilon
    entropy = -torch.sum(tensor * torch.log2(tensor), dim=tuple(range(1, tensor.dim())))
    return entropy

# Function to calculate importance score
def calculate_importance_score(weights, entropies,eta=2):
    return scores ** eta

# Pruning function
def prune_model(model, amount,eta=2):
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):

val_losses = []
train_losses = []
test_losses = []
best_test_accuracy = 0.0

# Training Loop
for epoch in range(100):  # loop over the dataset
    running_loss = 0.0
    correct_train = 0
    total_train = 0
    
    for i, data in enumerate(trainloader, 0):
        # Get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        # Use those GPUs!
        inputs, labels = inputs.to(device), labels.to(device)
        # Zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        
        optimizer.step()
        
        # Calculate accuracy
        _, predicted_train = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted_train == labels).sum().item()
        train_acc = 100 * correct_train // total_train

        # Print statistics
        running_loss += loss.item()
        
        
    # Calculate training error
    train_loss = running_loss / len(trainloader)
    train_losses.append(train_loss)
        
    print(f'Training Error at Epoch {epoch + 1}: {train_loss}')
        
    # Apply pruning based on importance score
    prune_model(model, amount=0.75,eta=2)
    
    model.eval()
    val_loss = 0.0
    correct_test = 0
    total_test = 0
    
    with torch.no_grad():
        for i, data in enumerate(testloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            
            # Calculate accuracy
            _, predicted_test = torch.max(outputs.data, 1)
            total_test += labels.size(0)
            correct_test += (predicted_test == labels).sum().item()
            
    # Calculate test error
    test_loss = val_loss / len(testloader.dataset)
    test_acc = round(100 * correct_test / total_test, 2)
    test_losses.append(test_loss)

    print(f'Test Error at Epoch {epoch + 1}: {test_loss}')
    print(f'Test Accuracy at Epoch {epoch + 1}: {test_acc}%')
    
    # Update best test accuracy
    if test_acc > best_test_accuracy:
        best_test_accuracy = test_acc
    
    model.train()

    val_losses.append(test_loss)

# Close all hooks
for hook in hooks:
    hook.close()

for hook in entropy_hooks:
    hook.close()

# Print best test accuracy
print(f"Best Test Accuracy: {best_test_accuracy}%")
