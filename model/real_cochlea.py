#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
真实音频耳蜗编码器 — 将音频波形通过FFT映射到32通道仿生耳蜗
输入：音频波形(numpy array)
输出：[timesteps, 160] spike张量（32ch × repeat 5）

耳蜗参数：80-8000Hz，32通道，对数间隔（仿生）
"""
import numpy as np
import torch


class RealCochlea:
    """
    真实音频耳蜗编码器
    音频波形 → FFT → 32频率通道 → spike
    """
    
    def __init__(self, n_channels=32, freq_range=(80, 8000), 
                 noise_level=0.03, spike_timesteps=10):
        self.n_channels = n_channels
        self.freq_lo, self.freq_hi = freq_range
        self.noise_level = noise_level
        self.spike_timesteps = spike_timesteps
        
        # 对数间隔的频率边界（仿生：人耳对低频更敏感）
        self.freq_edges = np.logspace(
            np.log10(self.freq_lo), 
            np.log10(self.freq_hi), 
            n_channels + 1
        )
        # 每个通道的中心频率
        self.center_freqs = np.sqrt(self.freq_edges[:-1] * self.freq_edges[1:])
    
    def encode_waveform(self, waveform, sample_rate):
        """
        音频波形 → 32通道spike → repeat到160维
        
        waveform: numpy array, shape (n_samples,) 或 (n_samples, n_channels)
        sample_rate: int, 采样率
        返回: torch.Tensor shape (spike_timesteps, 160)
        """
        # 确保单声道
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)
        waveform = waveform.astype(np.float32)
        
        # 归一化
        peak = np.abs(waveform).max()
        if peak > 0:
            waveform = waveform / peak
        
        # FFT
        n = len(waveform)
        if n < 2:
            return torch.zeros(self.spike_timesteps, 160)
        
        spectrum = np.abs(np.fft.rfft(waveform))
        freqs = np.fft.rfftfreq(n, d=1.0/sample_rate)
        
        # 映射到32个耳蜗通道
        channel_energy = np.zeros(self.n_channels, dtype=np.float32)
        for ch in range(self.n_channels):
            lo = self.freq_edges[ch]
            hi = self.freq_edges[ch + 1]
            mask = (freqs >= lo) & (freqs < hi)
            if mask.any():
                channel_energy[ch] = spectrum[mask].mean()
        
        # 归一化到 [0, 1]
        e_max = channel_energy.max()
        if e_max > 0:
            channel_energy = channel_energy / e_max
        
        # 加噪声
        noise = np.random.uniform(-self.noise_level, self.noise_level, 
                                   self.n_channels)
        channel_energy = np.clip(channel_energy + noise, 0, 1)
        
        # 能量 → spike概率 → spike张量
        spike_32 = torch.zeros(self.spike_timesteps, self.n_channels)
        for ch in range(self.n_channels):
            if channel_energy[ch] > 0.02:
                prob = min(0.8, channel_energy[ch] * 0.6)
                spike_32[:, ch] = (torch.rand(self.spike_timesteps) < prob).float()
        
        # 32 → 160 repeat
        spike_160 = spike_32.repeat(1, 5)
        return spike_160
    
    def describe(self, waveform, sample_rate):
        """调试：返回通道能量和spike统计"""
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)
        waveform = waveform.astype(np.float32)
        peak = np.abs(waveform).max()
        if peak > 0:
            waveform = waveform / peak
        
        n = len(waveform)
        spectrum = np.abs(np.fft.rfft(waveform))
        freqs = np.fft.rfftfreq(n, d=1.0/sample_rate)
        
        channel_energy = np.zeros(self.n_channels, dtype=np.float32)
        for ch in range(self.n_channels):
            lo = self.freq_edges[ch]
            hi = self.freq_edges[ch + 1]
            mask = (freqs >= lo) & (freqs < hi)
            if mask.any():
                channel_energy[ch] = spectrum[mask].mean()
        
        e_max = channel_energy.max()
        if e_max > 0:
            channel_energy_norm = channel_energy / e_max
        else:
            channel_energy_norm = channel_energy
        
        spike = self.encode_waveform(waveform * (peak if peak > 0 else 1), sample_rate)
        
        return {
            'channel_energy_raw': channel_energy.tolist(),
            'channel_energy_norm': channel_energy_norm.tolist(),
            'spike_sum': float(spike.sum()),
            'active_channels': int((channel_energy_norm > 0.05).sum()),
            'dominant_freq': float(self.center_freqs[np.argmax(channel_energy)]),
        }


# ========== 测试 ==========
if __name__ == '__main__':
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
    
    cochlea = RealCochlea(n_channels=32, freq_range=(80, 8000), noise_level=0.03)
    SR = 16000
    
    print("=" * 60)
    print("真实音频耳蜗编码器 — 测试")
    print(f"32通道, 80-8000Hz, 对数间隔")
    print(f"通道频率: {cochlea.center_freqs[0]:.0f}Hz ~ {cochlea.center_freqs[-1]:.0f}Hz")
    print("=" * 60)
    
    # Test 1: 纯440Hz正弦波（A4音）
    print("\n--- Test 1: 440Hz正弦波 ---")
    t = np.linspace(0, 0.5, SR // 2)
    wave_440 = np.sin(2 * np.pi * 440 * t)
    spike_440 = cochlea.encode_waveform(wave_440, SR)
    info_440 = cochlea.describe(wave_440, SR)
    print(f"  spike_sum={info_440['spike_sum']:.0f}")
    print(f"  active_channels={info_440['active_channels']}")
    print(f"  dominant_freq={info_440['dominant_freq']:.0f}Hz")
    
    # Test 2: 纯1000Hz
    print("\n--- Test 2: 1000Hz正弦波 ---")
    wave_1k = np.sin(2 * np.pi * 1000 * t)
    spike_1k = cochlea.encode_waveform(wave_1k, SR)
    info_1k = cochlea.describe(wave_1k, SR)
    print(f"  spike_sum={info_1k['spike_sum']:.0f}")
    print(f"  dominant_freq={info_1k['dominant_freq']:.0f}Hz")
    
    # Test 3: 440 vs 1000差异
    diff_freq = float(torch.norm(spike_440 - spike_1k))
    print(f"\n  440Hz vs 1000Hz 差异: {diff_freq:.2f}")
    print(f"  {'✅ 能区分' if diff_freq > 5 else '❌ 区分不够'}")
    
    # Test 4: 静音
    print("\n--- Test 3: 静音 ---")
    wave_silent = np.zeros(SR // 2)
    spike_silent = cochlea.encode_waveform(wave_silent, SR)
    print(f"  spike_sum={spike_silent.sum():.0f}")
    diff_silent = float(torch.norm(spike_440 - spike_silent))
    print(f"  440Hz vs 静音 差异: {diff_silent:.2f}")
    print(f"  {'✅ 能区分' if diff_silent > 5 else '❌ 区分不够'}")
    
    # Test 5: 模拟人声（多谐波）
    print("\n--- Test 4: 模拟人声(200Hz+谐波) ---")
    wave_voice = (np.sin(2*np.pi*200*t) + 0.5*np.sin(2*np.pi*400*t) + 
                  0.3*np.sin(2*np.pi*600*t) + 0.2*np.sin(2*np.pi*800*t))
    spike_voice = cochlea.encode_waveform(wave_voice, SR)
    info_voice = cochlea.describe(wave_voice, SR)
    print(f"  spike_sum={info_voice['spike_sum']:.0f}")
    print(f"  active_channels={info_voice['active_channels']}")
    print(f"  dominant_freq={info_voice['dominant_freq']:.0f}Hz")
    
    # Test 6: 白噪声
    print("\n--- Test 5: 白噪声 ---")
    wave_noise = np.random.randn(SR // 2) * 0.3
    spike_noise = cochlea.encode_waveform(wave_noise, SR)
    info_noise = cochlea.describe(wave_noise, SR)
    print(f"  spike_sum={info_noise['spike_sum']:.0f}")
    print(f"  active_channels={info_noise['active_channels']}")
    
    diff_voice_noise = float(torch.norm(spike_voice - spike_noise))
    print(f"  人声 vs 噪声 差异: {diff_voice_noise:.2f}")
    print(f"  {'✅ 能区分' if diff_voice_noise > 3 else '❌ 区分不够'}")
    
    print("\n" + "=" * 60)
    all_pass = diff_freq > 5 and diff_silent > 5 and diff_voice_noise > 3
    print(f"{'✅ 全部通过' if all_pass else '❌ 有未通过项'}")
    print("=" * 60)
