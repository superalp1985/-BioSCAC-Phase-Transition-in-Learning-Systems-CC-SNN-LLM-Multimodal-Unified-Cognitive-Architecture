#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
通用模态编码器框架 — 任何能数字化的感知都能接入CC-SNN
================================================================
原理：任何物理信号 → 传感器数字化 → 通道映射 → spike编码 → SNN

已实现：视觉(棱镜)、听觉(耳蜗)、文本(bigram)
可扩展：嗅觉、触觉、味觉、温度、加速度……

扩展规则：
  1. 定义传感器通道数（类比生物感受器数量）
  2. 定义通道→spike的映射（rate coding / temporal coding）
  3. 提供get_block_map()，让SNN知道怎么做局部连接
  4. 就这三步，其他全复用
"""
import numpy as np
import torch
from abc import ABC, abstractmethod


class ModalityEncoder(ABC):
    """所有模态编码器的基类"""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def total_dim(self) -> int: ...

    @abstractmethod
    def encode(self, raw_signal, timesteps=10) -> torch.Tensor:
        """原始信号 → (timesteps, total_dim) spike张量"""
        ...

    @abstractmethod
    def get_block_map(self, n_neurons_per_group=None):
        """返回 (block_map, total_neurons) 供SNN局部连接用"""
        ...

    def describe(self):
        return {'name': self.name, 'dim': self.total_dim}


class OlfactoryEncoder(ModalityEncoder):
    """
    嗅觉编码器 — 仿生嗅球
    生物基础：人类~400种嗅觉受体，每种对特定分子官能团响应
    数字化：电子鼻传感器阵列(MOX/QCM/CP)，每个传感器对一类气体敏感

    输入：n_sensors个传感器的响应值 (0-1)
    输出：spike张量

    block_map：每个传感器簇→独立神经元子群（仿嗅球 glomeruli）
    """

    def __init__(self, n_sensors=32, n_clusters=8, noise_level=0.03):
        self._n_sensors = n_sensors
        self.n_clusters = n_clusters  # 仿嗅球肾小球
        self.sensors_per_cluster = n_sensors // n_clusters
        self.noise_level = noise_level

    @property
    def name(self): return 'olfactory'
    @property
    def total_dim(self): return self._n_sensors

    def encode(self, sensor_values, timesteps=10):
        """sensor_values: (n_sensors,) array, 每个传感器0-1响应"""
        v = np.clip(np.array(sensor_values, dtype=np.float32), 0, 1)
        noise = np.random.uniform(-self.noise_level, self.noise_level, v.shape)
        v = np.clip(v + noise, 0, 1)
        spike = torch.zeros(timesteps, self._n_sensors)
        for ch in range(self._n_sensors):
            if v[ch] > 0.02:
                prob = min(0.8, v[ch] * 0.6)
                spike[:, ch] = (torch.rand(timesteps) < prob).float()
        return spike

    def get_block_map(self, n_neurons_per_group=None):
        if n_neurons_per_group is None:
            n_neurons_per_group = self.sensors_per_cluster
        bm = []
        for i in range(self.n_clusters):
            s = i * self.sensors_per_cluster
            e = (i+1) * self.sensors_per_cluster
            ns = i * n_neurons_per_group
            ne = (i+1) * n_neurons_per_group
            bm.append((s, e, ns, ne))
        return bm, self.n_clusters * n_neurons_per_group


class TactileEncoder(ModalityEncoder):
    """
    触觉编码器 — 仿生皮肤
    生物基础：4种机械感受器(Merkel/Meissner/Ruffini/Pacinian)
    数字化：压力传感器阵列/电子皮肤

    输入：(grid_h, grid_w) 压力图 + (grid_h, grid_w) 振动图
    输出：spike张量

    block_map：
      - 压力通道（慢适应，Merkel/Ruffini）→ 子群A
      - 振动通道（快适应，Meissner/Pacinian）→ 子群B
      - 每个子群内按空间区域再分（局部连接）
    """

    def __init__(self, grid_h=4, grid_w=4, noise_level=0.03):
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.spatial_dim = grid_h * grid_w
        self._total_dim = self.spatial_dim * 2  # 压力+振动
        self.noise_level = noise_level

    @property
    def name(self): return 'tactile'
    @property
    def total_dim(self): return self._total_dim

    def encode(self, pressure, vibration=None, timesteps=10):
        """
        pressure: (grid_h, grid_w) 0-1
        vibration: (grid_h, grid_w) 0-1, 可选
        """
        p = np.clip(np.array(pressure, dtype=np.float32).flatten(), 0, 1)
        if vibration is not None:
            vib = np.clip(np.array(vibration, dtype=np.float32).flatten(), 0, 1)
        else:
            vib = np.zeros(self.spatial_dim, dtype=np.float32)

        spike = torch.zeros(timesteps, self._total_dim)
        for ch in range(self.spatial_dim):
            # 压力（慢适应：持续发放）
            if p[ch] > 0.02:
                prob = min(0.8, p[ch] * 0.5)
                spike[:, ch] = (torch.rand(timesteps) < prob).float()
            # 振动（快适应：只在变化时发放）
            if vib[ch] > 0.02:
                prob = min(0.9, vib[ch] * 0.7)
                # 快适应：前半段密后半段稀
                t_half = timesteps // 2
                spike[:t_half, self.spatial_dim + ch] = (torch.rand(t_half) < prob).float()
                spike[t_half:, self.spatial_dim + ch] = (torch.rand(timesteps - t_half) < prob * 0.2).float()
        return spike

    def get_block_map(self, n_neurons_per_group=None):
        if n_neurons_per_group is None:
            n_neurons_per_group = self.spatial_dim
        return [
            (0, self.spatial_dim, 0, n_neurons_per_group),
            (self.spatial_dim, self._total_dim, n_neurons_per_group, 2*n_neurons_per_group)
        ], 2 * n_neurons_per_group


class ProprioceptionEncoder(ModalityEncoder):
    """
    本体感觉编码器 — 仿生关节/肌梭
    生物基础：肌梭(位置) + 高尔基腱器(力) + 关节感受器(角度)
    数字化：IMU(加速度/陀螺仪) + 力传感器

    输入：(n_joints,) 角度 + (n_joints,) 角速度 + (n_joints,) 力矩
    """

    def __init__(self, n_joints=6, noise_level=0.03):
        self.n_joints = n_joints
        self._total_dim = n_joints * 3  # 角度+角速度+力矩
        self.noise_level = noise_level

    @property
    def name(self): return 'proprioception'
    @property
    def total_dim(self): return self._total_dim

    def encode(self, angles, velocities=None, torques=None, timesteps=10):
        a = np.clip(np.array(angles, dtype=np.float32), 0, 1)
        v = np.clip(np.array(velocities if velocities is not None else np.zeros(self.n_joints), dtype=np.float32), 0, 1)
        t = np.clip(np.array(torques if torques is not None else np.zeros(self.n_joints), dtype=np.float32), 0, 1)
        combined = np.concatenate([a, v, t])
        noise = np.random.uniform(-self.noise_level, self.noise_level, combined.shape)
        combined = np.clip(combined + noise, 0, 1)
        spike = torch.zeros(timesteps, self._total_dim)
        for ch in range(self._total_dim):
            if combined[ch] > 0.02:
                spike[:, ch] = (torch.rand(timesteps) < min(0.8, combined[ch]*0.6)).float()
        return spike

    def get_block_map(self, n_neurons_per_group=None):
        nj = self.n_joints
        if n_neurons_per_group is None:
            n_neurons_per_group = nj
        return [
            (0, nj, 0, n_neurons_per_group),
            (nj, 2*nj, n_neurons_per_group, 2*n_neurons_per_group),
            (2*nj, 3*nj, 2*n_neurons_per_group, 3*n_neurons_per_group)
        ], 3 * n_neurons_per_group


# ══════════════ 多模态融合block_map ══════════════

def build_multimodal_block_map(encoders):
    """
    多模态融合：拼接所有编码器的block_map
    输入维度 = sum(enc.total_dim)
    每个编码器的通道→独立SNN神经元子群

    返回: (total_input_dim, block_map, total_neurons)
    """
    total_in = 0
    total_neurons = 0
    block_map = []

    for enc in encoders:
        bm, n_neurons = enc.get_block_map()
        for in_s, in_e, n_s, n_e in bm:
            block_map.append((
                in_s + total_in,
                in_e + total_in,
                n_s + total_neurons,
                n_e + total_neurons
            ))
        total_in += enc.total_dim
        total_neurons += n_neurons

    return total_in, block_map, total_neurons
