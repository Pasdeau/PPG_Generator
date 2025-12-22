import torch
import torch.nn as nn
import torch.nn.functional as F

class ResNet1D_Block(nn.Module):
    """1D ResNet Residual Block"""
    
    def __init__(self, in_channels, out_channels, kernel_size=7, stride=1):
        super(ResNet1D_Block, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, 
                              stride=stride, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(out_channels)
        
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                              stride=1, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )
    
    def forward(self, x):
        residual = x
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        out += self.shortcut(residual)
        out = F.relu(out)
        
        return out


class ResNet1D_Classifier(nn.Module):
    """
    ResNet1D Classifier - Deep network
    
    Suitable for large-scale datasets.
    """
    
    def __init__(self, input_length=8000, num_classes=5, dropout=0.5, in_channels=2):
        super(ResNet1D_Classifier, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=15, stride=2, padding=7)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(3, stride=2, padding=1)
        
        # Residual Blocks
        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)  # Add dropout
        self.fc = nn.Linear(256, num_classes)
    
    def _make_layer(self, in_channels, out_channels, num_blocks, stride=1):
        layers = []
        layers.append(ResNet1D_Block(in_channels, out_channels, stride=stride))
        for _ in range(1, num_blocks):
            layers.append(ResNet1D_Block(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)  # Apply dropout
        x = self.fc(x)
        
        return x
