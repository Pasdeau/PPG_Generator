#!/usr/bin/env python3
"""
PPG Signal Classification - Model Definitions
Supports various deep learning architectures
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from ml_training.models.unet_ppg import UNetPPG

class CNN1D_Classifier(nn.Module):
    """
    1D CNN Classifier - Recommended for general use
    
    Applicable for:
    - Waveform classification (5 classes)
    - Artifact classification (5 classes)
    - Rhythm classification (2 classes)
    """
    
    def __init__(self, input_length=8000, num_classes=5, in_channels=2):
        """
        Parameters:
        -----------
        input_length : int
            Input signal length (samples)
        num_classes : int
            Number of output classes
        in_channels : int
            Number of input channels (Default 2: Amplitude + Velocity)
        """
        super(CNN1D_Classifier, self).__init__()
        
        # Conv Layer 1: Extract low-level features
        self.conv1 = nn.Conv1d(
            in_channels=in_channels,      # Multi-channel PPG signal
            out_channels=32,    # 32 feature maps
            kernel_size=50,     # 50ms window (assuming 1000Hz sampling)
            stride=2,
            padding=25
        )
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(kernel_size=4, stride=4)
        
        # Conv Layer 2: Extract mid-level features
        self.conv2 = nn.Conv1d(32, 64, kernel_size=25, stride=2, padding=12)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(kernel_size=4, stride=4)
        
        # Conv Layer 3: Extract high-level features
        self.conv3 = nn.Conv1d(64, 128, kernel_size=10, stride=1, padding=5)
        self.bn3 = nn.BatchNorm1d(128)
        self.pool3 = nn.MaxPool1d(kernel_size=4, stride=4)
        
        # Calculate FC input dimension
        # input_length -> conv1 -> pool1 -> conv2 -> pool2 -> conv3 -> pool3
        fc_input_dim = self._get_fc_input_dim(input_length)
        
        # Fully Connected Layers
        self.fc1 = nn.Linear(fc_input_dim, 256)
        self.dropout1 = nn.Dropout(0.5)
        
        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(0.3)
        
        self.fc3 = nn.Linear(128, num_classes)
    
    def _get_fc_input_dim(self, input_length):
        """Calculate FC input dimension by forward pass"""
        # Simulate forward pass
        x = torch.zeros(1, self.conv1.in_channels, input_length)
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        return x.view(1, -1).size(1)
    
    def forward(self, x):
        """
        Forward pass
        
        Parameters:
        -----------
        x : torch.Tensor
            Input signal [batch_size, channels, signal_length]
        
        Returns:
        --------
        torch.Tensor
            Classification logits [batch_size, num_classes]
        """
        # Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)
        
        # Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)
        
        # Block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool3(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # FC Layers
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        
        x = self.fc3(x)
        
        return x


class LSTM_Classifier(nn.Module):
    """
    LSTM Classifier - Suitable for sequence modeling
    
    Applicable for:
    - Rhythm classification (long-term patterns)
    - Complex temporal features
    """
    
    def __init__(self, input_length=8000, num_classes=5, hidden_size=128, num_layers=2, in_channels=2):
        super(LSTM_Classifier, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM Layer
        self.lstm = nn.LSTM(
            input_size=in_channels,           # Features per time step
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3 if num_layers > 1 else 0,
            bidirectional=True      # Bi-directional LSTM
        )
        
        # FC Layer
        self.fc1 = nn.Linear(hidden_size * 2, 128)  # *2 for bidirectional
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)
    
    def forward(self, x):
        """
        Parameters:
        -----------
        x : torch.Tensor
            [batch_size, channels, signal_length]
        """
        # Convert to LSTM input format [batch, seq_len, features]
        x = x.transpose(1, 2)  # [batch, signal_length, channels]
        
        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use simple last time step output logic (or attention in advanced versions)
        # h_n: [num_layers*2, batch, hidden_size]
        # Get forward/backward hidden states from last layer
        forward_hidden = h_n[-2, :, :]
        backward_hidden = h_n[-1, :, :]
        hidden = torch.cat([forward_hidden, backward_hidden], dim=1)
        
        # FC Layers
        x = F.relu(self.fc1(hidden))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


class CNN_LSTM_Classifier(nn.Module):
    """
    CNN + LSTM Hybrid Model - Advanced
    
    CNN extracts local features, LSTM models temporal dependencies.
    """
    
    def __init__(self, input_length=8000, num_classes=5, in_channels=2):
        super(CNN_LSTM_Classifier, self).__init__()
        
        # CNN Feature Extractor
        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=50, stride=2, padding=25)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(4)
        
        self.conv2 = nn.Conv1d(64, 128, kernel_size=25, stride=2, padding=12)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(4)
        
        # LSTM Temporal Modellling
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=0.3,
            bidirectional=True
        )
        
        # Classification Head
        self.fc1 = nn.Linear(256, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)
    
    def forward(self, x):
        # CNN Features
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        
        # Convert to LSTM Input [batch, seq_len, features]
        x = x.transpose(1, 2)
        
        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use final hidden state
        forward_hidden = h_n[-2, :, :]
        backward_hidden = h_n[-1, :, :]
        hidden = torch.cat([forward_hidden, backward_hidden], dim=1)
        
        # Classification
        x = F.relu(self.fc1(hidden))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


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


def create_model(model_type='cnn', input_length=8000, num_classes=5, in_channels=34, **kwargs):
    """
    Model Factory Function
    
    Parameters:
    -----------
    model_type : str
        'cnn', 'lstm', 'cnn_lstm', 'resnet', 'unet'
    input_length : int
        Input signal length
    num_classes : int
        Number of output classes
    in_channels : int
        Number of input channels (Default 34: Amp + Vel + 32 CWT scales)
    
    Returns:
    --------
    nn.Module
        PyTorch Model
    """
    models = {
        'cnn': CNN1D_Classifier,
        'lstm': LSTM_Classifier,
        'cnn_lstm': CNN_LSTM_Classifier,
        'resnet': ResNet1D_Classifier,
        'unet': UNetPPG
    }
    
    if model_type not in models:
        raise ValueError(f"Unknown model type: {model_type}. Choose from {list(models.keys())}")
    
    # UNetPPG signature is different (kwargs handles extra args)
    if model_type == 'unet':
       return UNetPPG(in_channels=in_channels, **kwargs)
       
    model = models[model_type](input_length=input_length, num_classes=num_classes, in_channels=in_channels, **kwargs)
    return model


if __name__ == '__main__':
    # Test models
    print("=" * 70)
    print("PPG Model Factory Test")
    print("=" * 70)
    
    batch_size = 4
    signal_length = 8000
    num_classes = 5
    
    in_channels = 2
    
    # Create test input
    x = torch.randn(batch_size, in_channels, signal_length)
    
    models_to_test = ['cnn', 'lstm', 'cnn_lstm', 'resnet']
    
    for model_type in models_to_test:
        print(f"\nTesting {model_type.upper()} model:")
        model = create_model(model_type, signal_length, num_classes, in_channels=in_channels)
        
        # Forward pass
        output = model(x)
        
        # Parameter count
        num_params = sum(p.numel() for p in model.parameters())
        
        print(f"  Input shape: {x.shape}")
        print(f"  Output shape: {output.shape}")
        print(f"  Parameters: {num_params:,}")
        print(f"  [INFO] Test passed")
    
    print("\n" + "=" * 70)
    print("All models tested successfully!")
    print("=" * 70)
