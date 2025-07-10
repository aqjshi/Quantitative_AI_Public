import torch
import torch.nn as nn
import torch.nn.functional as F



class Conv2DMultiBinary(nn.Module):
    def __init__(self, in_channels=1, base_filters=16, num_outputs=1):
        super().__init__()
        
        # --- Convolutional Block 1 ---
        # Input -> (base_filters) x H x W
        self.conv1 = nn.Conv2d(in_channels, base_filters, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(base_filters)
        self.pool1 = nn.MaxPool2d(2) # HxW -> H/2 x W/2

        # --- Convolutional Block 2 ---
        # -> (base_filters*2) x H/2 x W/2
        self.conv2 = nn.Conv2d(base_filters, base_filters * 2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(base_filters * 2)
        self.pool2 = nn.MaxPool2d(2) # H/2 x W/2 -> H/4 x W/4
        
        # --- Convolutional Block 3 ---
        # -> (base_filters*4) x H/4 x W/4
        self.conv3 = nn.Conv2d(base_filters * 2, base_filters * 4, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(base_filters * 4)
        self.pool3 = nn.MaxPool2d(2) # H/4 x W/4 -> H/8 x W/8
        
        # --- Convolutional Block 4 ---
        # -> (base_filters*8) x H/8 x W/8
        self.conv4 = nn.Conv2d(base_filters * 4, base_filters * 8, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(base_filters * 8)
        
        # --- Classifier Head ---
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1)) # Reduces spatial dims to 1x1
        self.dropout = nn.Dropout(0.5) # Regularization
        self.fc1 = nn.Linear(base_filters * 8, 128) # Heavier fully connected layer
        self.fc2 = nn.Linear(128, num_outputs) # Final output layer

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = F.relu(self.bn4(self.conv4(x))) # No pool after last conv
        
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1) # Flatten
        
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        
        return self.fc2(x)

