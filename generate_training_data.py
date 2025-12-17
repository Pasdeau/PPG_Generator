#!/usr/bin/env python3
"""
训练数据生成器 - PPG信号分类器数据集

生成用于训练噪声和波形分类器的标注数据集。
支持批量生成、自动标注、FFT特征提取。

Usage:
    python generate_training_data.py --num_samples 1000 --output_dir dataset/
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import signal as sp_signal
import os
import json
import argparse
from datetime import datetime
from ppg_generator import gen_PPG
from ppg_artifacts import gen_PPG_artifacts
from data_loader import load_artifact_params, generate_synthetic_rr


class PPGDatasetGenerator:
    """PPG训练数据集生成器"""
    
    def __init__(self, output_dir='dataset', Fd=1000):
        """
        初始化数据集生成器
        
        Parameters:
        -----------
        output_dir : str
            输出目录
        Fd : int
            采样频率 (Hz)
        """
        self.output_dir = output_dir
        self.Fd = Fd
        
        # 创建目录结构
        self.signal_dir = os.path.join(output_dir, 'signals')
        self.label_dir = os.path.join(output_dir, 'labels')
        self.viz_dir = os.path.join(output_dir, 'visualizations')
        self.fft_dir = os.path.join(output_dir, 'fft_features')
        
        for d in [self.signal_dir, self.label_dir, self.viz_dir, self.fft_dir]:
            os.makedirs(d, exist_ok=True)
        
        # 加载伪影参数
        try:
            self.artifact_params = load_artifact_params('artifact_param.mat')
        except:
            print("Warning: Could not load artifact_param.mat, artifacts will be disabled")
            self.artifact_params = None
    
    def compute_fft(self, signal_data):
        """
        计算FFT特征
        
        Returns:
        --------
        fft_features : dict
            包含freq, magnitude, phase, power_spectrum等
        """
        # FFT
        fft_result = np.fft.rfft(signal_data)
        freq = np.fft.rfftfreq(len(signal_data), 1/self.Fd)
        
        magnitude = np.abs(fft_result)
        phase = np.angle(fft_result)
        power_spectrum = magnitude ** 2
        
        # 找主频率成分
        dominant_freq_idx = np.argmax(power_spectrum[1:]) + 1  # 排除DC
        dominant_freq = freq[dominant_freq_idx]
        
        # 频率范围统计
        freq_bands = {
            'dc': (0, 0.5),
            'pulse': (0.5, 3.0),      # 脉搏频率范围30-180 bpm
            'breathing': (0.15, 0.4),  # 呼吸频率
            'high_freq': (3.0, freq[-1])  # 高频噪声
        }
        
        band_power = {}
        for band_name, (f_low, f_high) in freq_bands.items():
            band_mask = (freq >= f_low) & (freq <= f_high)
            band_power[band_name] = np.sum(power_spectrum[band_mask])
        
        return {
            'freq': freq,
            'magnitude': magnitude,
            'phase': phase,
            'power_spectrum': power_spectrum,
            'dominant_freq': dominant_freq,
            'band_power': band_power,
            'sampling_freq': self.Fd
        }
    
    def generate_sample(self, sample_id, pulse_type=None, artifact_config=None,
                       num_beats=50, rhythm_type='SR', mean_rr=800, std_rr=50):
        """
        生成单个训练样本
        
        Parameters:
        -----------
        sample_id : int or str
            样本ID
        pulse_type : int or None
            脉搏类型1-5，None表示随机
        artifact_config : dict or None
            伪影配置 {'types': [0,0,0,1], 'dur_mu': 6}
        num_beats : int
            心跳数量
        rhythm_type : str
            'SR' or 'AF'
        mean_rr : float
            平均RR间期
        std_rr : float
            RR标准差
            
        Returns:
        --------
        sample_data : dict
            包含信号、标签、元数据等
        """
        # 随机选择脉搏类型（如果未指定）
        if pulse_type is None:
            pulse_type = np.random.randint(1, 6)
        
        # 生成RR间期
        RR = generate_synthetic_rr(num_beats, rhythm_type=rhythm_type,
                                   mean_rr=mean_rr, std_rr=std_rr)
        
        # 生成PPG信号
        PPGmodel, PPGpeakIdx, PPGpeakVal = gen_PPG(RR, pulse_type=pulse_type, Fd=self.Fd)
        
        # 归一化
        PPGmodel_norm = (PPGmodel - np.mean(PPGmodel)) / np.std(PPGmodel)
        
        # 生成伪影和标签
        artifact_signal = np.zeros_like(PPGmodel_norm)
        artifact_labels = np.zeros(len(PPGmodel_norm), dtype=int)  # 0=clean, 1-4=artifact types
        
        if artifact_config is not None and self.artifact_params is not None:
            typ_artifact = np.array(artifact_config.get('types', [0, 0, 0, 0]))
            
            if np.sum(typ_artifact) > 0:
                # 生成伪影
                prob = typ_artifact * (1 / np.sum(typ_artifact))
                dur_mu0 = artifact_config.get('dur_mu0', 10)
                dur_mu = artifact_config.get('dur_mu', 6)
                dur_mu_vec = np.array([dur_mu0] + [dur_mu] * 4)
                
                slope = np.zeros(4)
                for n in range(4):
                    slope[n] = self.artifact_params['slope_sd'][n] * np.random.randn() + \
                              self.artifact_params['slope_m'][n]
                
                # 生成伪影（带状态追踪）
                artifact_signal, artifact_states = self._gen_artifacts_with_labels(
                    len(PPGmodel_norm), prob, dur_mu_vec,
                    self.artifact_params['RMS_shape'],
                    self.artifact_params['RMS_scale'],
                    slope
                )
                
                artifact_labels = artifact_states
        
        # 合并信号
        PPG_final = PPGmodel_norm + artifact_signal
        
        # 计算FFT特征
        fft_features = self.compute_fft(PPG_final)
        
        # 创建标签结构
        labels = {
            'pulse_type': int(pulse_type),
            'rhythm_type': rhythm_type,
            'has_artifact': bool(np.any(artifact_labels > 0)),
            'artifact_timeline': artifact_labels.tolist(),  # 每个采样点的伪影类型
            'artifact_summary': self._summarize_artifacts(artifact_labels),
            'peak_indices': PPGpeakIdx.tolist(),
            'rr_intervals': RR.tolist(),
            'num_beats': num_beats,
            'sampling_freq': self.Fd,
            'duration_seconds': len(PPG_final) / self.Fd
        }
        
        # 元数据
        metadata = {
            'sample_id': str(sample_id),
            'generation_time': datetime.now().isoformat(),
            'mean_rr': float(np.mean(RR)),
            'std_rr': float(np.std(RR)),
            'mean_hr': float(60000 / np.mean(RR)),
            'signal_mean': float(np.mean(PPG_final)),
            'signal_std': float(np.std(PPG_final)),
            'snr_db': float(self._calculate_snr(PPGmodel_norm, artifact_signal))
        }
        
        return {
            'signal': PPG_final,
            'clean_signal': PPGmodel_norm,
            'artifact_signal': artifact_signal,
            'labels': labels,
            'metadata': metadata,
            'fft_features': fft_features
        }
    
    def _gen_artifacts_with_labels(self, duration_samples, prob, dur_mu,
                                   RMS_shape, RMS_scale, slope):
        """生成伪影并同时记录状态标签"""
        from ppg_artifacts import gen_PPG_artifacts
        
        # 马尔科夫链状态转换矩阵
        P = np.zeros((5, 5))
        P[0, :] = [0] + list(prob)  # 从无伪影到各类伪影
        P[1:, 0] = 1  # 从伪影回到无伪影
        
        states_vec = []
        artifact_signal = []
        current_state = 0  # 开始无伪影
        
        while len(states_vec) < duration_samples:
            # 生成持续时间
            T = max(10, int(np.random.exponential(dur_mu[current_state])))
            
            if current_state > 0:  # 有伪影
                # 生成伪影段
                from scipy.signal import firls
                # 设计FIR滤波器
                f = np.array([0, 0.5, 0.51, 1])
                a = np.array([1, 10**(-slope[current_state-1]/20), 0, 0])
                b = firls(201, f, a)
                
                # 生成噪声并滤波
                noise = np.random.randn(T + 200)
                filtered = np.convolve(noise, b, mode='valid')[:T]
                filtered = (filtered - np.mean(filtered)) / np.std(filtered)
                
                # 伽马分布幅度
                R = np.random.gamma(RMS_shape[current_state-1], RMS_scale[current_state-1])
                segment = R * filtered
                
                artifact_signal.extend(segment)
                states_vec.extend([current_state] * T)
            else:  # 无伪影
                artifact_signal.extend([0] * T)
                states_vec.extend([0] * T)
            
            # 状态转换
            u = np.random.rand()
            cumsum = np.cumsum(P[current_state, :])
            current_state = np.where(u < cumsum)[0][0]
        
        # 截断到指定长度
        artifact_signal = np.array(artifact_signal[:duration_samples])
        states_vec = np.array(states_vec[:duration_samples], dtype=int)
        
        return artifact_signal, states_vec
    
    def _summarize_artifacts(self, artifact_labels):
        """总结伪影时段"""
        segments = []
        if len(artifact_labels) == 0 or np.all(artifact_labels == 0):
            return segments
        
        # 找到伪影段
        artifact_mask = artifact_labels > 0
        changes = np.diff(np.concatenate([[0], artifact_mask.astype(int), [0]]))
        starts = np.where(changes == 1)[0]
        ends = np.where(changes == -1)[0]
        
        for start_idx, end_idx in zip(starts, ends):
            # 找这一段的伪影类型（取众数）
            segment_types = artifact_labels[start_idx:end_idx]
            artifact_type = int(np.bincount(segment_types).argmax())
            
            segments.append({
                'start_sample': int(start_idx),
                'end_sample': int(end_idx),
                'start_time': float(start_idx / self.Fd),
                'end_time': float(end_idx / self.Fd),
                'duration': float((end_idx - start_idx) / self.Fd),
                'artifact_type': artifact_type  # 1-4
            })
        
        return segments
    
    def _calculate_snr(self, clean, noise):
        """计算信噪比"""
        if np.std(noise) == 0:
            return np.inf
        signal_power = np.mean(clean ** 2)
        noise_power = np.mean(noise ** 2)
        if noise_power == 0:
            return np.inf
        return 10 * np.log10(signal_power / noise_power)
    
    def save_sample(self, sample_data, visualize=True, save_fft=True):
        """
        保存样本到磁盘
        
        Parameters:
        -----------
        sample_data : dict
            generate_sample()返回的数据
        visualize : bool
            是否生成可视化
        save_fft : bool
            是否保存FFT特征
        """
        sample_id = sample_data['metadata']['sample_id']
        
        # 保存信号数据 (.npz格式)
        signal_path = os.path.join(self.signal_dir, f'{sample_id}.npz')
        np.savez_compressed(
            signal_path,
            signal=sample_data['signal'],
            clean_signal=sample_data['clean_signal'],
            artifact_signal=sample_data['artifact_signal']
        )
        
        # 保存标签 (JSON格式，易读)
        label_path = os.path.join(self.label_dir, f'{sample_id}.json')
        with open(label_path, 'w') as f:
            json.dump({
                'labels': sample_data['labels'],
                'metadata': sample_data['metadata']
            }, f, indent=2)
        
        # 保存FFT特征
        if save_fft:
            fft_path = os.path.join(self.fft_dir, f'{sample_id}_fft.npz')
            np.savez_compressed(
                fft_path,
                freq=sample_data['fft_features']['freq'],
                magnitude=sample_data['fft_features']['magnitude'],
                power_spectrum=sample_data['fft_features']['power_spectrum'],
                dominant_freq=sample_data['fft_features']['dominant_freq'],
                band_power=sample_data['fft_features']['band_power']
            )
        
        # 可视化
        if visualize:
            self._visualize_sample(sample_data)
    
    def _visualize_sample(self, sample_data):
        """生成样本可视化"""
        sample_id = sample_data['metadata']['sample_id']
        signal = sample_data['signal']
        clean = sample_data['clean_signal']
        artifact = sample_data['artifact_signal']
        artifact_labels = np.array(sample_data['labels']['artifact_timeline'])
        fft = sample_data['fft_features']
        
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(4, 2, hspace=0.3, wspace=0.3)
        
        t = np.arange(len(signal)) / self.Fd
        
        # Plot 1: 完整信号
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(t, signal, 'k-', linewidth=0.8, label='Complete Signal')
        # 标注伪影段
        for seg in sample_data['labels']['artifact_summary']:
            ax1.axvspan(seg['start_time'], seg['end_time'], 
                       alpha=0.3, color=f'C{seg["artifact_type"]}',
                       label=f'Artifact Type {seg["artifact_type"]}' if seg==sample_data['labels']['artifact_summary'][0] else '')
        ax1.set_ylabel('Amplitude')
        ax1.set_title(f'Sample {sample_id} | Pulse Type {sample_data["labels"]["pulse_type"]} | '
                     f'{sample_data["labels"]["rhythm_type"]}', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper right', fontsize=8)
        
        # Plot 2: 纯净信号
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.plot(t, clean, 'b-', linewidth=0.8)
        peak_idx = sample_data['labels']['peak_indices']
        ax2.plot(np.array(peak_idx)/self.Fd, clean[peak_idx], 'ro', markersize=3)
        ax2.set_ylabel('Amplitude')
        ax2.set_title('Clean PPG Signal')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: 伪影
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.plot(t, artifact, 'r-', linewidth=0.8)
        ax3.set_ylabel('Amplitude')
        ax3.set_title('Artifact Signal')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: FFT Magnitude
        ax4 = fig.add_subplot(gs[2, 0])
        ax4.plot(fft['freq'], fft['magnitude'], 'b-', linewidth=1)
        ax4.axvline(fft['dominant_freq'], color='r', linestyle='--', 
                   label=f'Dominant: {fft["dominant_freq"]:.2f} Hz')
        ax4.set_xlabel('Frequency (Hz)')
        ax4.set_ylabel('Magnitude')
        ax4.set_title('FFT Magnitude Spectrum')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        ax4.set_xlim([0, min(10, fft['freq'][-1])])
        
        # Plot 5: Power Spectrum
        ax5 = fig.add_subplot(gs[2, 1])
        ax5.semilogy(fft['freq'], fft['power_spectrum'], 'g-', linewidth=1)
        ax5.set_xlabel('Frequency (Hz)')
        ax5.set_ylabel('Power (log scale)')
        ax5.set_title('Power Spectrum')
        ax5.grid(True, alpha=0.3)
        ax5.set_xlim([0, min(10, fft['freq'][-1])])
        
        # Plot 6: Artifact Labels Timeline
        ax6 = fig.add_subplot(gs[3, :])
        ax6.plot(t, artifact_labels, 'k-', linewidth=1, drawstyle='steps-post')
        ax6.set_xlabel('Time (s)')
        ax6.set_ylabel('Artifact Type')
        ax6.set_yticks([0, 1, 2, 3, 4])
        ax6.set_yticklabels(['Clean', 'Type 1', 'Type 2', 'Type 3', 'Type 4'])
        ax6.set_title('Artifact Labels Timeline')
        ax6.grid(True, alpha=0.3)
        ax6.set_ylim([-0.5, 4.5])
        
        # 保存
        viz_path = os.path.join(self.viz_dir, f'{sample_id}.png')
        plt.savefig(viz_path, dpi=120, bbox_inches='tight')
        plt.close()
    
    def generate_dataset(self, num_samples, config=None, verbose=True):
        """
        批量生成数据集
        
        Parameters:
        -----------
        num_samples : int
            要生成的样本数量
        config : dict
            生成配置，包括pulse_types分布、artifact类型等
        verbose : bool
            是否显示进度
            
        Returns:
        --------
        dataset_info : dict
dataset统计信息
        """
        if config is None:
            config = {
                'pulse_types': [1, 2, 3, 4, 5],  # 均匀分布
                'artifact_ratios': [0.3, 0.7],  # 30%无伪影, 70%有伪影
                'rhythm_types': ['SR', 'AF'],  # 节律类型
                'rhythm_ratios': [0.8, 0.2]  # 80% SR, 20% AF
            }
        
        print(f"Generating {num_samples} samples...")
        print(f"Output directory: {self.output_dir}")
        print()
        
        dataset_info = {
            'num_samples': num_samples,
            'pulse_type_distribution': {},
            'artifact_distribution': {'clean': 0, 'with_artifact': 0},
            'rhythm_distribution': {},
            'generation_config': config
        }
        
        for i in range(num_samples):
            if verbose and (i % 10 == 0 or i == num_samples - 1):
                print(f"Progress: {i+1}/{num_samples} ({100*(i+1)/num_samples:.1f}%)")
            
            # 随机选择配置
            pulse_type = np.random.choice(config['pulse_types'])
            rhythm_type = np.random.choice(config['rhythm_types'], 
                                          p=config['rhythm_ratios'])
            
            # 是否添加伪影
            add_artifact = np.random.choice([False, True], 
                                           p=config['artifact_ratios'])
            
            if add_artifact:
                # 随机选择伪影类型组合
                artifact_config = {
                    'types': np.random.choice([0, 1], size=4, p=[0.7, 0.3]).tolist(),
                    'dur_mu0': np.random.uniform(8, 15),
                    'dur_mu': np.random.uniform(4, 10)
                }
            else:
                artifact_config = None
            
            # 生成样本
            sample_data = self.generate_sample(
                sample_id=f'sample_{i:06d}',
                pulse_type=pulse_type,
                artifact_config=artifact_config,
                num_beats=np.random.randint(40, 80),
                rhythm_type=rhythm_type,
                mean_rr=np.random.uniform(600, 1000),
                std_rr=np.random.uniform(30, 100)
            )
            
            # 保存
            self.save_sample(sample_data, visualize=(i < 10))  # 只可视化前10个
            
            # 更新统计
            pt = sample_data['labels']['pulse_type']
            dataset_info['pulse_type_distribution'][pt] = \
                dataset_info['pulse_type_distribution'].get(pt, 0) + 1
            
            if sample_data['labels']['has_artifact']:
                dataset_info['artifact_distribution']['with_artifact'] += 1
            else:
                dataset_info['artifact_distribution']['clean'] += 1
            
            rt = sample_data['labels']['rhythm_type']
            dataset_info['rhythm_distribution'][rt] = \
                dataset_info['rhythm_distribution'].get(rt, 0) + 1
        
        # 保存数据集信息
        info_path = os.path.join(self.output_dir, 'dataset_info.json')
        with open(info_path, 'w') as f:
            json.dump(dataset_info, f, indent=2)
        
        print(f"\n✓ Dataset generation complete!")
        print(f"  Total samples: {num_samples}")
        print(f"  Pulse types: {dataset_info['pulse_type_distribution']}")
        print(f"  Artifacts: {dataset_info['artifact_distribution']}")
        print(f"  Rhythms: {dataset_info['rhythm_distribution']}")
        print(f"  Saved to: {self.output_dir}")
        
        return dataset_info


def main():
    """主程序"""
    parser = argparse.ArgumentParser(description='Generate PPG training dataset')
    parser.add_argument('--num_samples', type=int, default=100,
                      help='Number of samples to generate')
    parser.add_argument('--output_dir', type=str, default='dataset',
                      help='Output directory')
    parser.add_argument('--Fd', type=int, default=1000,
                      help='Sampling frequency (Hz)')
    parser.add_argument('--seed', type=int, default=None,
                      help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    if args.seed is not None:
        np.random.seed(args.seed)
    
    # 创建生成器
    generator = PPGDatasetGenerator(output_dir=args.output_dir, Fd=args.Fd)
    
    # 生成数据集
    generator.generate_dataset(num_samples=args.num_samples)


if __name__ == "__main__":
    main()
