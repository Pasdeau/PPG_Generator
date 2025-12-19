#!/usr/bin/env python3
"""
PPG Dataset Loader
PyTorch Dataset and DataLoader
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import json
from pathlib import Path


class PPGDataset(Dataset):
    """
    PPG Signal Dataset
    
    Supports:
    - Waveform classification
    - Artifact classification
    - Rhythm classification
    """
    
    def __init__(self, data_dir, task='waveform', transform=None, max_length=None):
        """
        Parameters:
        -----------
        data_dir : str
            Data directory path
        task : str
            'waveform' - Waveform classification (5 classes)
            'artifact' - Artifact classification (5 classes)
            'rhythm' - Rhythm classification (2 classes)
            'multitask' - Returns signal, mask, and label
        transform : callable, optional
            Data augmentation function
        max_length : int, optional
            Signal max length (truncation or padding)
        """
        self.data_dir = Path(data_dir)
        self.task = task
        self.transform = transform
        self.max_length = max_length
        
        # Load all samples
        self.samples = self._load_samples()
        
        # Label mapping
        self.label_maps = {
            'waveform': {1: 0, 2: 1, 3: 2, 4: 3, 5: 4},  # pulse_type 1-5 -> 0-4
            'artifact': {
                'clean': 0,
                'device_disp': 1,
                'forearm': 2,
                'hand': 3,
                'poor_contact': 4
            },
            'rhythm': {'SR': 0, 'AF': 1}
        }
    
    def _load_samples(self):
        """Load all sample file paths and metadata"""
        samples = []
        
        # Find all .npz files
        npz_files = list(self.data_dir.glob('*.npz'))
        
        for npz_file in npz_files:
            # 1. Try finding metadata file in same directory (Legacy)
            meta_file = npz_file.with_name(npz_file.stem + '_meta.json')
            
            # 2. Try finding metadata in ../labels/ directory (New Structure)
            if not meta_file.exists():
                # Assuming structure is dataset/signals/*.npz and dataset/labels/*.json
                meta_file = npz_file.parent.parent / 'labels' / (npz_file.stem + '.json')
            
            if meta_file.exists():
                with open(meta_file, 'r') as f:
                    full_meta = json.load(f)
                    # Support both flat metadata and nested under 'metadata'/'labels'
                    if 'metadata' in full_meta and 'labels' in full_meta:
                        # Merge for convenience
                        metadata = full_meta['metadata']
                        metadata.update(full_meta['labels'])
                    else:
                        metadata = full_meta
                
                samples.append({
                    'npz_path': str(npz_file),
                    'metadata': metadata
                })
            else:
                # If no metadata, parse from filename
                # Format: sample_0000_p2_SR_poor_contact.npz
                parts = npz_file.stem.split('_')
                if len(parts) >= 5:
                    samples.append({
                        'npz_path': str(npz_file),
                        'metadata': {
                            'pulse_type': int(parts[2][1]),  # p2 -> 2
                            'rhythm': parts[3],
                            'artifact_type': '_'.join(parts[4:])
                        }
                    })
        
        print(f"Loaded {len(samples)} samples")
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        Get single sample
        
        Returns:
        --------
        signal : torch.Tensor
            PPG signal [1, signal_length]
        label : int
            Label
        """
        sample = self.samples[idx]
        
        # Load signal
        data = np.load(sample['npz_path'])
        if 'PPG' in data:
            signal = data['PPG']
        elif 'signal' in data:
            signal = data['signal']
        else:
             raise KeyError("No signal found in .npz (checked 'PPG' and 'signal')")
        
        # Select label based on task
        metadata = sample['metadata']
        seg_mask = None
        
        if self.task == 'waveform':
            label = self.label_maps['waveform'][metadata['pulse_type']]
        elif self.task == 'artifact':
            artifact_type = metadata['artifact_type']
            label = self.label_maps['artifact'].get(artifact_type, 0)
        elif self.task == 'rhythm':
            label = self.label_maps['rhythm'][metadata['rhythm']]
        elif self.task == 'multitask':
            label = self.label_maps['waveform'][metadata['pulse_type']]
            if 'artifact_timeline' in metadata:
                seg_mask = np.array(metadata['artifact_timeline'], dtype=int)
            else:
                 # Default to all zeros if missing (should not happen with new generator)
                 seg_mask = np.zeros(len(signal), dtype=int)
        else:
            raise ValueError(f"Unknown task: {self.task}")
        
        # Handle signal length
        if self.max_length is not None:
            if len(signal) > self.max_length:
                # Truncate
                signal = signal[:self.max_length]
                if seg_mask is not None:
                    seg_mask = seg_mask[:self.max_length]
            elif len(signal) < self.max_length:
                # Pad
                pad_len = self.max_length - len(signal)
                signal = np.pad(signal, (0, pad_len), mode='constant')
                if seg_mask is not None:
                    seg_mask = np.pad(seg_mask, (0, pad_len), mode='constant', constant_values=0)
        
        # Convert to tensor
        # Calculate 1st derivative (Velocity)
        derivative = np.diff(signal, prepend=signal[0])
        # Normalize derivative independently
        der_mean = np.mean(derivative)
        der_std = np.std(derivative) + 1e-6
        derivative = (derivative - der_mean) / der_std
        
        # Normalize raw signal again to be safe (or rely on generator)
        # It is safer to normalize here to mean=0, std=1
        sig_mean = np.mean(signal)
        sig_std = np.std(signal) + 1e-6
        signal = (signal - sig_mean) / sig_std
        
        # Stack channels: [2, length]
        # Channel 0: Amplitude
        # Channel 1: Velocity
        combined_signal = np.stack([signal, derivative], axis=0) # [2, length]
        
        signal_tensor = torch.FloatTensor(combined_signal) # [2, length]
        label = torch.LongTensor([label])[0]
        
        # Data augmentation
        if self.transform:
            signal_tensor = self.transform(signal_tensor)
        
        if seg_mask is not None:
             mask_tensor = torch.LongTensor(seg_mask)
             return signal_tensor, mask_tensor, label
        
        return signal_tensor, label
    
    def get_class_distribution(self):
        """Get class distribution"""
        from collections import Counter
        
        labels = []
        for sample in self.samples:
            metadata = sample['metadata']
            if self.task == 'waveform':
                labels.append(metadata['pulse_type'])
            elif self.task == 'artifact':
                labels.append(metadata['artifact_type'])
            elif self.task == 'rhythm':
                labels.append(metadata['rhythm'])
        
        return Counter(labels)


class DataAugmentation:
    """Data Augmentation"""
    
    @staticmethod
    def add_noise(signal, noise_level=0.01):
        """Add Gaussian noise"""
        noise = torch.randn_like(signal) * noise_level
        return signal + noise
    
    @staticmethod
    def time_shift(signal, shift_range=100):
        """Time shift"""
        shift = np.random.randint(-shift_range, shift_range)
        return torch.roll(signal, shift, dims=-1)
    
    @staticmethod
    def amplitude_scale(signal, scale_range=(0.8, 1.2)):
        """Amplitude scaling"""
        scale = np.random.uniform(*scale_range)
        return signal * scale
    
    @staticmethod
    def time_stretch(signal, stretch_range=(0.9, 1.1)):
        """Time stretch (Simplified)"""
        # Note: This is simplified, real application might need more complex interpolation
        stretch = np.random.uniform(*stretch_range)
        if stretch == 1.0:
            return signal
        
        # Simple resampling
        length = signal.shape[-1]
        new_length = int(length * stretch)
        indices = torch.linspace(0, length-1, new_length).long()
        stretched = signal[..., indices]
        
        # Pad or truncate to original length
        if new_length < length:
            pad = length - new_length
            stretched = torch.nn.functional.pad(stretched, (0, pad))
        else:
            stretched = stretched[..., :length]
        
        return stretched


def create_augmentation_transform(augment_prob=0.5):
    """Create data augmentation transform"""
    
    def transform(signal):
        if np.random.rand() < augment_prob:
            # Randomly select augmentation method
            aug_type = np.random.choice(['noise', 'shift', 'scale', 'stretch'])
            
            if aug_type == 'noise':
                signal = DataAugmentation.add_noise(signal)
            elif aug_type == 'shift':
                signal = DataAugmentation.time_shift(signal)
            elif aug_type == 'scale':
                signal = DataAugmentation.amplitude_scale(signal)
            elif aug_type == 'stretch':
                signal = DataAugmentation.time_stretch(signal)
        
        return signal
    
    return transform


def create_dataloaders(data_dir, task='waveform', batch_size=32, 
                       train_split=0.7, val_split=0.15, 
                       max_length=8000, num_workers=4, augment=True):
    """
    Create train, validation, and test data loaders
    
    Parameters:
    -----------
    data_dir : str
        Data directory
    task : str
        Task type
    batch_size : int
        Batch size
    train_split : float
        Train split ratio
    val_split : float
        Validation split ratio
    max_length : int
        Signal max length
    num_workers : int
        Number of workers
    augment : bool
        Whether to use augmentation
    
    Returns:
    --------
    train_loader, val_loader, test_loader
    """
    
    # Create full dataset
    full_dataset = PPGDataset(data_dir, task=task, max_length=max_length)
    
    # Split dataset
    dataset_size = len(full_dataset)
    train_size = int(train_split * dataset_size)
    val_size = int(val_split * dataset_size)
    test_size = dataset_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)  # Fixed random seed
    )
    
    # Add augmentation to training set
    if augment:
        # Note: Need to modify transform attribute here
        # Real application might need more complex handling
        pass
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"\nDataset Split:")
    print(f"  Train: {train_size} samples")
    print(f"  Val:   {val_size} samples")
    print(f"  Test:  {test_size} samples")
    
    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    # Test data loader
    print("=" * 70)
    print("PPG Dataset Loader Test")
    print("=" * 70)
    
    # Assume data in batch_demo directory
    data_dir = '../batch_demo'
    
    if os.path.exists(data_dir):
        # Test dataset
        dataset = PPGDataset(data_dir, task='waveform', max_length=8000)
        
        print(f"\nDataset size: {len(dataset)}")
        print(f"\nClass Distribution: {dataset.get_class_distribution()}")
        
        # Test single sample
        signal, label = dataset[0]
        print(f"\nSample shape: {signal.shape}")
        print(f"Label: {label}")
        
        # Test DataLoader
        train_loader, val_loader, test_loader = create_dataloaders(
            data_dir, task='waveform', batch_size=4
        )
        
        # Test one batch
        for signals, labels in train_loader:
            print(f"\nBatch shape: {signals.shape}")
            print(f"Labels: {labels}")
            break
        
        print("\n[INFO] Data loader successfully tested!")
    else:
        print(f"\n[WARN] Data directory not found: {data_dir}")
        print("Please run generate_training_data.py first to generate data")
    
    print("=" * 70)
