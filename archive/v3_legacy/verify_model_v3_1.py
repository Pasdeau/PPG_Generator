#!/usr/bin/env python3
import torch
from ml_training.model_factory import create_model

def verify_v3_1():
    print("="*60)
    print("Verifying v3.1 SE-Attention UNet")
    print("="*60)
    
    # 1. Baseline v3.0 (No Attention)
    print("[-] Instantiating v3.0 (Baseline)...")
    model_v3 = create_model('unet', in_channels=34, attention=False)
    params_v3 = sum(p.numel() for p in model_v3.parameters())
    print(f"    v3.0 Parameters: {params_v3:,}")
    
    # 2. Upgrade v3.1 (With Attention)
    print("[-] Instantiating v3.1 (SE-Attention)...")
    model_v3_1 = create_model('unet', in_channels=34, attention=True)
    params_v3_1 = sum(p.numel() for p in model_v3_1.parameters())
    print(f"    v3.1 Parameters: {params_v3_1:,}")
    
    # 3. Comparison
    delta = params_v3_1 - params_v3
    percent = (delta / params_v3) * 100
    print(f"[-] Parameter Increase: +{delta:,} ({percent:.2f}%)")
    
    if delta > 0:
        print("[SUCCESS] SE-Blocks successfully integrated (Parameters increased).")
    else:
        print("[FAIL] No parameter increase detected. SE-Blocks might be missing.")
        
    # 4. Forward Pass Test
    print("[-] Testing Forward Pass (Input: [1, 34, 1000])...")
    dummy_input = torch.randn(1, 34, 1000)
    try:
        clf, seg = model_v3_1(dummy_input)
        print(f"    Output Clf: {clf.shape}")
        print(f"    Output Seg: {seg.shape}")
        
        if clf.shape == (1, 5) and seg.shape == (1, 5, 1000):
            print("[SUCCESS] Forward pass successful. Shape remains correct.")
        else:
            print("[FAIL] Output shape mismatch.")
            
    except Exception as e:
        print(f"[FAIL] Forward pass crashed: {e}")
    
    print("="*60)

if __name__ == "__main__":
    verify_v3_1()
