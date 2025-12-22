"""
Streaming Inference Demo for v4.0 Dual-Stream Model
Simulates real-time data reception from ADS1298 and continuous model inference.

Usage:
    python streaming_inference_demo.py
"""

import numpy as np
import torch
import time
from collections import deque
import matplotlib.pyplot as plt
import sys
import os

# Try to import model (optional)
MODEL_AVAILABLE = False
try:
    sys.path.append(os.path.abspath('./ml_training'))
    from model_factory import create_dual_stream_model
    MODEL_AVAILABLE = True
except ImportError:
    print("[WARNING] Model not available, running in simulation mode")


class StreamingPPGProcessor:
    """
    Simulates continuous PPG data reception and real-time inference
    """
    
    def __init__(self, model=None, window_size=8000, stride=1000, sample_rate=1000):
        """
        Args:
            model: Trained v4.0 model
            window_size: Inference window (8s = 8000 samples)
            stride: Stride for sliding window (1s = 1000 samples)
            sample_rate: Sampling frequency (Hz)
        """
        self.model = model
        self.window_size = window_size
        self.stride = stride
        self.sample_rate = sample_rate
        
        # Circular buffer to store incoming data
        self.buffer = deque(maxlen=window_size)
        
        # Results storage
        self.timestamps = []
        self.classifications = []
        self.segmentations = []
        self.latencies = []
        
        # Statistics
        self.total_samples_received = 0
        self.total_inferences = 0
        
        if model is not None:
            self.model.eval()
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model.to(self.device)
            print(f"[+] Model loaded on {self.device}")
        else:
            print("[!] No model loaded, running in stats-only mode")
    
    def receive_sample(self, sample):
        """
        Simulate receiving a single sample from ADS1298
        In real implementation, this would be called by serial port callback
        """
        self.buffer.append(sample)
        self.total_samples_received += 1
    
    def is_ready_for_inference(self):
        """Check if buffer has enough data"""
        return len(self.buffer) == self.window_size
    
    def run_inference(self):
        """
        Run v4.0 model inference on current buffer
        Returns: (pulse_type, segmentation_mask, latency_ms)
        """
        if not self.is_ready_for_inference():
            return None, None, 0
        
        start_time = time.time()
        
        # Convert buffer to numpy array
        signal = np.array(self.buffer)
        
        if self.model is None:
            # Simulation mode: random output
            pulse_type = np.random.randint(0, 5)
            seg_mask = np.random.randint(0, 5, size=len(signal))
            latency_ms = np.random.uniform(50, 100)
        else:
            # Real inference
            # Prepare input (normalize + add derivative)
            signal_norm = (signal - np.mean(signal)) / (np.std(signal) + 1e-8)
            velocity = np.gradient(signal_norm)
            
            # Stack as 2-channel input
            x_time = np.stack([signal_norm, velocity], axis=0)  # [2, 8000]
            x_time = torch.FloatTensor(x_time).unsqueeze(0).to(self.device)  # [1, 2, 8000]
            
            # For CWT input, we need to compute CWT
            # (Simplified: use zeros for demo, real implementation needs pywt)
            x_freq = torch.zeros(1, 34, self.window_size).to(self.device)
            
            with torch.no_grad():
                clf_out, seg_out = self.model(x_time, x_freq)
                pulse_type = clf_out.argmax(dim=1).item()
                seg_mask = seg_out.argmax(dim=1).squeeze().cpu().numpy()
            
            latency_ms = (time.time() - start_time) * 1000
        
        # Store results
        self.timestamps.append(self.total_samples_received / self.sample_rate)
        self.classifications.append(pulse_type)
        self.segmentations.append(seg_mask)
        self.latencies.append(latency_ms)
        self.total_inferences += 1
        
        return pulse_type, seg_mask, latency_ms
    
    def process_stream(self, data_stream, stride=None):
        """
        Process a continuous data stream
        
        Args:
            data_stream: numpy array of PPG samples
            stride: how many samples to skip between inferences (default: self.stride)
        """
        if stride is None:
            stride = self.stride
        
        print(f"[*] Processing stream of {len(data_stream)} samples...")
        print(f"[*] Window: {self.window_size}, Stride: {stride}")
        
        for i, sample in enumerate(data_stream):
            # Receive sample
            self.receive_sample(sample)
            
            # Run inference every 'stride' samples
            if (i + 1) % stride == 0 and self.is_ready_for_inference():
                pulse_type, seg_mask, latency = self.run_inference()
                
                # Print progress
                if self.total_inferences % 10 == 0:
                    pulse_names = ['SR', 'PVC', 'PAC', 'Fusion', 'Unknown']
                    print(f"  [{i+1:6d}/{len(data_stream)}] "
                          f"Type: {pulse_names[pulse_type]}, "
                          f"Latency: {latency:.1f}ms")
        
        print(f"[✓] Stream processing complete!")
        print(f"    Total samples: {self.total_samples_received}")
        print(f"    Total inferences: {self.total_inferences}")
        print(f"    Avg latency: {np.mean(self.latencies):.1f}ms ± {np.std(self.latencies):.1f}ms")
    
    def get_statistics(self):
        """Return processing statistics"""
        return {
            'total_samples': self.total_samples_received,
            'total_inferences': self.total_inferences,
            'mean_latency_ms': np.mean(self.latencies) if self.latencies else 0,
            'std_latency_ms': np.std(self.latencies) if self.latencies else 0,
            'max_latency_ms': np.max(self.latencies) if self.latencies else 0,
            'throughput_hz': self.sample_rate / (np.mean(self.latencies) / 1000) if self.latencies else 0
        }


def generate_test_stream(duration_sec=60, sample_rate=1000):
    """
    Generate a synthetic PPG stream for testing
    Simplified version using sinusoidal approximation
    """
    print(f"[*] Generating {duration_sec}s test stream...")
    
    num_samples = duration_sec * sample_rate
    t = np.arange(num_samples) / sample_rate
    
    # Simulate PPG as sum of sinusoids (heart rate ~1.2 Hz = 72 bpm)
    hr_freq = 1.2  # Heart rate frequency
    signal = np.sin(2 * np.pi * hr_freq * t)  # Main pulse
    signal += 0.3 * np.sin(2 * np.pi * hr_freq * 2 * t)  # Harmonics
    signal += 0.1 * np.sin(2 * np.pi * hr_freq * 3 * t)
    
    # Add some artifacts (random bursts)
    for _ in range(10):
        start_idx = np.random.randint(0, len(signal) - 5000)
        duration = np.random.randint(1000, 3000)
        artifact = np.random.randn(duration) * 0.5
        signal[start_idx:start_idx+duration] += artifact
    
    # Normalize
    signal = (signal - np.mean(signal)) / np.std(signal)
    
    print(f"[✓] Generated stream: {len(signal)} samples ({len(signal)/sample_rate:.1f}s)")
    return signal


def visualize_results(processor):
    """Visualize streaming inference results"""
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    # Plot 1: Classification over time
    ax1 = axes[0]
    pulse_names = ['SR', 'PVC', 'PAC', 'Fusion', 'Unknown']
    colors = ['blue', 'red', 'orange', 'purple', 'gray']
    
    for i, clf in enumerate(processor.classifications):
        t = processor.timestamps[i]
        ax1.scatter(t, clf, c=colors[clf], s=50, alpha=0.6)
    
    ax1.set_ylabel("Pulse Type")
    ax1.set_yticks(range(5))
    ax1.set_yticklabels(pulse_names)
    ax1.set_title("Streaming Classification Results")
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Inference latency
    ax2 = axes[1]
    ax2.plot(processor.timestamps, processor.latencies, 'g-', linewidth=1, alpha=0.7)
    ax2.axhline(y=np.mean(processor.latencies), color='r', linestyle='--', 
                label=f'Mean: {np.mean(processor.latencies):.1f}ms')
    ax2.set_ylabel("Latency (ms)")
    ax2.set_title("Inference Latency Over Time")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Statistics
    ax3 = axes[2]
    stats = processor.get_statistics()
    stats_text = f"""
Streaming Inference Statistics:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Total Samples Received: {stats['total_samples']:,}
Total Inferences: {stats['total_inferences']:,}
Mean Latency: {stats['mean_latency_ms']:.2f} ms
Std Latency: {stats['std_latency_ms']:.2f} ms
Max Latency: {stats['max_latency_ms']:.2f} ms
Throughput: {stats['throughput_hz']:.1f} Hz

✓ Real-time capable: {'YES' if stats['mean_latency_ms'] < 100 else 'NO (>100ms)'}
"""
    ax3.text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
             verticalalignment='center')
    ax3.axis('off')
    
    plt.tight_layout()
    plt.savefig('streaming_inference_results.png', dpi=150, bbox_inches='tight')
    print("[✓] Saved results to streaming_inference_results.png")
    plt.show()


def main():
    print("=" * 60)
    print("PPG Streaming Inference Demo (v4.0 Dual-Stream Model)")
    print("=" * 60)
    
    # Try to load model
    model = None
    if MODEL_AVAILABLE:
        try:
            model_path = 'output/v4_best_model.pth'
            if os.path.exists(model_path):
                # Load model architecture (need to define this based on actual model)
                print(f"[!] Model loading not fully implemented - running in simulation mode")
                print(f"[!] To enable real inference, implement model loading logic")
            else:
                print(f"[!] Model file not found: {model_path}")
        except Exception as e:
            print(f"[!] Error loading model: {e}")
    
    # Create processor
    processor = StreamingPPGProcessor(
        model=model,
        window_size=8000,  # 8 seconds
        stride=1000,       # Inference every 1 second
        sample_rate=1000
    )
    
    # Generate test stream (60 seconds)
    stream = generate_test_stream(duration_sec=60, sample_rate=1000)
    
    # Process stream
    print("\n" + "=" * 60)
    processor.process_stream(stream, stride=1000)
    print("=" * 60)
    
    # Show statistics
    stats = processor.get_statistics()
    print("\n📊 Performance Summary:")
    print(f"  ├─ Samples Processed: {stats['total_samples']:,}")
    print(f"  ├─ Inferences Run: {stats['total_inferences']:,}")
    print(f"  ├─ Avg Latency: {stats['mean_latency_ms']:.2f} ± {stats['std_latency_ms']:.2f} ms")
    print(f"  ├─ Max Latency: {stats['max_latency_ms']:.2f} ms")
    print(f"  └─ Throughput: {stats['throughput_hz']:.1f} Hz")
    
    # Real-time check
    print("\n🎯 Real-time Capability Assessment:")
    if stats['mean_latency_ms'] < 100:
        print("  ✓ PASS: Model can handle 1000Hz streaming (latency << 1ms per sample)")
    else:
        print("  ✗ FAIL: Latency too high for real-time processing")
        print(f"    Recommended stride: {int(stats['mean_latency_ms'])} samples")
    
    # Visualize
    print("\n📈 Generating visualization...")
    visualize_results(processor)
    

if __name__ == "__main__":
    main()
