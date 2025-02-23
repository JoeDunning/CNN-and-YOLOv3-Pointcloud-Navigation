# Library Imports ---------------------------------------------------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F

# Model Framework ---------------------------------------------------------------------------------------------------------------------

class ClassCNN(nn.Module):
    def __init__(self):
        super(ClassCNN, self).__init__()
        
        # -- | First layer 
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)  
        self.pool = nn.MaxPool2d(2, 2)  
  
        # -- | Second layer
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2) 
        
        # -- | Third layer 
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  
        self.pool3 = nn.MaxPool2d(2, 2)  
        
        # -- | Adaptive pooling
        self.adaptive_pool = nn.AdaptiveAvgPool2d((8, 8))
        
        # -- | FC (Fully connected) layers 
        self.fc1 = nn.Linear(64*8*8, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 2)

    def forward(self, x):
        
        # -- | Convolutions and pooling layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        
        # -- | Apply adaptive pooling to standardize the output size & flatten features
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        
        # -- | ReLU activations 
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  
        
        return x