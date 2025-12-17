#!/usr/bin/env python3
"""
快速批量生成PPG信号 - 简化版
生成不同波形和噪声组合的PPG信号
"""

import numpy as np
import os
import json
from ppg_generator import gen_PPG
from ppg_artifacts import gen_PPG_artifacts
from data_loader import load_artifact_params, generate_synthetic_rr, save_ppg_data

def batch_generate_ppg(num_samples=20, output_dir='batch_output'):
    """
    批量生成PPG信号
    
    Parameters:
    -----------
    num_samples : int
        生成样本数量
    output_dir : str
        输出目录
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 70)
    print(f"批量生成 {num_samples} 个PPG样本")
    print("=" * 70)
    
    # 配置
    pulse_types = [1, 2, 3, 4, 5]  # 5种脉搏类型
    rhythms = ['SR', 'AF']          # 2种心律
    artifact_configs = [
        {'add': False, 'name': 'clean'},
        {'add': True, 'typ': [1, 0, 0, 0], 'name': 'device_disp'},
        {'add': True, 'typ': [0, 1, 0, 0], 'name': 'forearm'},
        {'add': True, 'typ': [0, 0, 1, 0], 'name': 'hand'},
        {'add': True, 'typ': [0, 0, 0, 1], 'name': 'poor_contact'},
    ]
    
    Fd = 1000
    num_beats = 30
    
    # 加载伪影参数
    try:
        artifact_params = load_artifact_params('data/artifact_param.mat')
        print("✓ 加载伪影参数")
    except:
        print("⚠️  使用默认伪影参数")
        artifact_params = {
            'RMS_shape': np.array([2.0, 2.0, 2.0, 2.0]),
            'RMS_scale': np.array([0.5, 0.5, 0.5, 0.5]),
            'slope_m': np.array([-1.0, -1.0, -1.0, -1.0])
        }
    
    print(f"\n配置:")
    print(f"  脉搏类型: {len(pulse_types)} 种")
    print(f"  心律类型: {len(rhythms)} 种")
    print(f"  伪影配置: {len(artifact_configs)} 种")
    print(f"  每样本心拍数: {num_beats}")
    print(f"  采样频率: {Fd} Hz")
    print()
    
    # 生成样本
    sample_count = 0
    
    for i in range(num_samples):
        # 随机选择配置
        pulse_type = np.random.choice(pulse_types)
        rhythm = np.random.choice(rhythms)
        artifact_cfg = np.random.choice(artifact_configs)
        
        # 生成RR间期
        if rhythm == 'SR':
            RR = generate_synthetic_rr(num_beats, mean_rr=800, std_rr=50)
        else:
            RR = generate_synthetic_rr(num_beats, mean_rr=800, std_rr=150)
        
        # 生成PPG
        PPGmodel, PPGpeakIdx, PPGpeakVal = gen_PPG(RR, pulse_type=pulse_type, Fd=Fd)
        
        # 生成伪影
        if artifact_cfg['add']:
            typ_artifact = np.array(artifact_cfg['typ'])
            prob = typ_artifact / np.sum(typ_artifact)
            dur_mu = np.array([15, 2, 2, 2, 2])  # 轻度伪影
            
            artifact = gen_PPG_artifacts(
                duration_sampl=len(PPGmodel),
                prob=prob,
                dur_mu=dur_mu,
                RMS_shape=artifact_params['RMS_shape'],
                RMS_scale=artifact_params['RMS_scale'],
                slope=artifact_params['slope_m'],
                Fd=Fd
            )
        else:
            artifact = np.zeros_like(PPGmodel)
        
        # 组合信号
        PPGmodel_norm = (PPGmodel - np.mean(PPGmodel)) / np.std(PPGmodel)
        PPG = PPGmodel_norm + artifact
        
        # 保存
        sample_id = f"sample_{i:04d}_p{pulse_type}_{rhythm}_{artifact_cfg['name']}"
        filepath = os.path.join(output_dir, sample_id)
        
        save_ppg_data(
            filepath, PPG, PPGmodel_norm, artifact,
            rr=RR, ppg_peak_idx=PPGpeakIdx, ppg_peak_val=PPGpeakVal,
            sampling_freq=Fd
        )
        
        # 保存元数据
        metadata = {
            'sample_id': sample_id,
            'pulse_type': int(pulse_type),
            'rhythm': rhythm,
            'artifact_type': artifact_cfg['name'],
            'num_beats': int(num_beats),
            'duration_sec': len(PPG) / Fd,
            'mean_hr_bpm': 60000 / np.mean(RR)
        }
        
        with open(f"{filepath}_meta.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        sample_count += 1
        
        if (i + 1) % 10 == 0:
            print(f"  生成进度: {i+1}/{num_samples} ({(i+1)/num_samples*100:.0f}%)")
    
    print(f"\n{'=' * 70}")
    print(f"✓ 完成！生成了 {sample_count} 个样本")
    print(f"  输出目录: {output_dir}/")
    print(f"  文件格式: .npz (信号) + .json (元数据)")
    print(f"\n查看生成的文件:")
    print(f"  ls {output_dir}/")
    print("=" * 70)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='批量生成PPG信号')
    parser.add_argument('--num_samples', type=int, default=20,
                       help='生成样本数量 (默认: 20)')
    parser.add_argument('--output_dir', type=str, default='batch_output',
                       help='输出目录 (默认: batch_output)')
    
    args = parser.parse_args()
    
    batch_generate_ppg(args.num_samples, args.output_dir)
