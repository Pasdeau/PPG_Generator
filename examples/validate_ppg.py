#!/usr/bin/env python3
"""
验证Python PPG生成器与MATLAB的一致性

此脚本：
1. 生成5种pulse type的波形
2. 与MATLAB模板对比
3. 计算误差
4. 生成对比可视化
"""

import numpy as np
import scipy.io as sio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ppg_generator import gen_PPG
from ppg_pulse import gen_PPGpulse

def compare_with_matlab():
    """对比Python生成的PPG与MATLAB模板"""
    
    print("="*70)
    print("Python PPG vs MATLAB 模板对比验证")
    print("="*70)
    print()
    
    # 加载MATLAB模板
    try:
        mat_data = sio.loadmat('pulse_templates.mat')
        pulse_templates = mat_data['pulse_templates'][0, 0]
        print("✓ 加载MATLAB模板成功")
    except FileNotFoundError:
        print("✗ 找不到 pulse_templates.mat")
        print("请先在MATLAB中运行 extract_pulse_templates.m")
        return
    
    # 设置测试参数（与MATLAB一致）
    RR_fixed = 1000  # ms
    Fd = 1000        # Hz
    
    # 创建对比图
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    results = []
    
    print("\n对比5种脉搏类型:\n")
    print(f"{'类型':<8} {'RMSE':<12} {'Peak误差':<12} {'形态相关性':<12} {'结果'}")
    print("-" * 70)
    
    for pulse_type in range(1, 6):
        # 获取MATLAB模板
        template_key = f'type{pulse_type}'
        template_struct = pulse_templates[template_key][0, 0]
        matlab_waveform = template_struct['waveform'].flatten()
        
        # 使用Python生成相同条件的PPG
        RR = np.array([RR_fixed] * 3)  # 3个相同的RR
        python_ppg, peak_idx, _ = gen_PPG(RR, pulse_type=pulse_type, Fd=Fd)
        
        # 提取中间脉搏（避免边缘效应）
        start_idx = RR_fixed
        end_idx = 2 * RR_fixed
        python_waveform = python_ppg[start_idx:end_idx]
        
        # 归一化到[0,1]
        python_waveform = (python_waveform - np.min(python_waveform)) / \
                          (np.max(python_waveform) - np.min(python_waveform))
        
        # 确保长度一致
        if len(python_waveform) != len(matlab_waveform):
            print(f"  警告: Type {pulse_type} 长度不一致")
            min_len = min(len(python_waveform), len(matlab_waveform))
            python_waveform = python_waveform[:min_len]
            matlab_waveform = matlab_waveform[:min_len]
        
        # 计算误差指标
        rmse = np.sqrt(np.mean((python_waveform - matlab_waveform) ** 2))
        
        # 峰值位置和值对比
        python_peak_idx = np.argmax(python_waveform)
        matlab_peak_idx = np.argmax(matlab_waveform)
        peak_error = abs(python_peak_idx - matlab_peak_idx)
        
        # 计算相关系数
        correlation = np.corrcoef(python_waveform, matlab_waveform)[0, 1]
        
        # 判断结果
        status = "✅ 优秀" if rmse < 0.01 else ("⚠️ 可接受" if rmse < 0.05 else "❌ 需改进")
        
        print(f"Type {pulse_type:<3} {rmse:<12.6f} {peak_error:<12d} {correlation:<12.6f} {status}")
        
        results.append({
            'type': pulse_type,
            'rmse': rmse,
            'peak_error': peak_error,
            'correlation': correlation
        })
        
        # 绘制对比
        ax = axes[pulse_type - 1]
        t = np.linspace(0, 1, len(matlab_waveform))
        
        ax.plot(t, matlab_waveform, 'b-', linewidth=2.5, label='MATLAB', alpha=0.7)
        ax.plot(t, python_waveform, 'r--', linewidth=1.5, label='Python', alpha=0.8)
        
        # 标记峰值
        ax.plot(t[matlab_peak_idx], matlab_waveform[matlab_peak_idx], 'bo', 
                markersize=8, label='MATLAB Peak')
        ax.plot(t[python_peak_idx], python_waveform[python_peak_idx], 'rs', 
                markersize=8, label='Python Peak')
        
        ax.set_title(f'Type {pulse_type} (RMSE: {rmse:.6f}, Corr: {correlation:.4f})', 
                    fontsize=11, fontweight='bold')
        ax.set_xlabel('Normalized Time')
        ax.set_ylabel('Normalized Amplitude')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    # 隐藏第6个子图
    axes[5].axis('off')
    
    plt.suptitle('Python PPG vs MATLAB Template Comparison', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('output/python_matlab_comparison.png', dpi=150)
    print("\n✓ 保存对比图: output/python_matlab_comparison.png")
    
    # 统计总结
    print("\n" + "="*70)
    print("总体统计:")
    print("="*70)
    rmses = [r['rmse'] for r in results]
    corrs = [r['correlation'] for r in results]
    print(f"平均RMSE: {np.mean(rmses):.6f}")
    print(f"最大RMSE: {np.max(rmses):.6f}")
    print(f"平均相关性: {np.mean(corrs):.6f}")
    print(f"最小相关性: {np.min(corrs):.6f}")
    
    if np.mean(rmses) < 0.01:
        print("\n🎉 结果: 优秀！Python实现与MATLAB高度一致")
    elif np.mean(rmses) < 0.05:
        print("\n✅ 结果: 良好！Python实现与MATLAB基本一致")
    else:
        print("\n⚠️ 结果: 需要进一步校准")
    
    return results


def test_different_sampling_frequencies():
    """测试不同采样频率"""
    
    print("\n" + "="*70)
    print("测试不同采样频率 (Fd)")
    print("="*70)
    print()
    
    RR = np.array([800] * 10)
    pulse_type = 1
    
    frequencies = [500, 1000, 2000]
    
    fig, axes = plt.subplots(len(frequencies), 1, figsize=(12, 10))
    
    for idx, Fd in enumerate(frequencies):
        ppg, peaks, _ = gen_PPG(RR, pulse_type=pulse_type, Fd=Fd)
        t = np.arange(len(ppg)) / Fd
        
        axes[idx].plot(t, ppg, 'b-', linewidth=1)
        axes[idx].plot(peaks/Fd, ppg[peaks], 'ro', markersize=4)
        axes[idx].set_title(f'Fd = {Fd} Hz ({len(ppg)} samples, {len(ppg)/Fd:.2f}s)')
        axes[idx].set_xlabel('Time (s)')
        axes[idx].set_ylabel('Amplitude')
        axes[idx].grid(True, alpha=0.3)
        
        print(f"Fd = {Fd} Hz: {len(ppg)} samples, {len(peaks)} peaks")
    
    plt.tight_layout()
    plt.savefig('output/different_sampling_freq.png', dpi=150)
    print("\n✓ 保存图像: output/different_sampling_freq.png")
    print("\n💡 结论: Fd可以修改,但建议使用1000Hz以保持与MATLAB一致")


def main():
    """主验证程序"""
    
    # 对比验证
    results = compare_with_matlab()
    
    # 测试不同采样频率
    test_different_sampling_frequencies()
    
    print("\n" + "="*70)
    print("验证完成!")
    print("="*70)
    print("\n生成的文件:")
    print("  - output/python_matlab_comparison.png  (5种类型对比)")
    print("  - output/different_sampling_freq.png   (不同Fd测试)")


if __name__ == "__main__":
    main()
