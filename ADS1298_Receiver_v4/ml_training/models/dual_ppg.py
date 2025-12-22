import torch
import torch.nn as nn
from ml_training.models.resnet1d import ResNet1D_Classifier
from ml_training.models.unet_ppg import UNetPPG

class DualTaskPPG(nn.Module):
    """
    v4.0 Dual-Stream Architecture
    
    Combines:
    1. ResNet1D Stream: Specialized in Waveform Classification (Branch A)
       - Input: Time-domain signal (Amplitude, Velocity)
    2. UNet Stream: Specialized in Artifact Segmentation (Branch B)
       - Input: Full CWT Tensor (34 channels)
       
    The streams run in parallel to avoid Feature Preference/Bias.
    """
    def __init__(self, in_channels=34, n_classes_seg=5, n_classes_clf=5, attention=True):
        super(DualTaskPPG, self).__init__()
        
        # Branch A: Classification (The "Expert Cardiologist")
        # Uses standard ResNet1D features from raw signal
        self.resnet_branch = ResNet1D_Classifier(
            input_length=8000, 
            num_classes=n_classes_clf, 
            dropout=0.5, 
            in_channels=2 # Only Ampltidue + Velocity
        )
        
        # Branch B: Segmentation (The "Noise Hunter")
        # Uses CWT features for segmentation
        self.unet_branch = UNetPPG(
            in_channels=in_channels, # 34 channels (Amp + Vel + 32 CWT)
            n_classes_seg=n_classes_seg,
            n_classes_clf=n_classes_clf, # Not used for final output but kept for compatibility
            attention=attention
        )
        
    def forward(self, x):
        """
        x: [Batch, 34, Length]
           - x[:, 0:1, :] -> Amplitude
           - x[:, 1:2, :] -> Velocity
           - x[:, 2:, :]  -> CWT Scalogram
        """
        
        # Stream 1: ResNet (Classification)
        # Slices only the first 2 channels (Time Domain)
        # We need to detach it from the CWT computation path effectively? 
        # No, x is just the input tensor.
        x_time = x[:, 0:2, :] 
        logits_clf = self.resnet_branch(x_time)
        
        # Stream 2: UNet (Segmentation)
        # Uses the full "Compound Eye" input
        # We ignore the UNet's internal classification head output
        _, logits_seg = self.unet_branch(x)
        
        return logits_clf, logits_seg
