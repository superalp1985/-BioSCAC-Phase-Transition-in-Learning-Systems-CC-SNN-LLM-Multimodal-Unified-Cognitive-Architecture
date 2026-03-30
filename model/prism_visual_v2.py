#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PSA棱镜视觉编码器 v2 — RGB三原色 + 局部连接 + 空间保持
=========================================================
仿生原理：人眼视锥细胞三种(L/M/S)对应红绿蓝
棱镜分光：白光→RGB三通道 + 边缘 + 纹理 = 5通道

核心改进：
  1. RGB三原色保留色彩信息（不再是黑白）
  2. 分辨率可调（8×8默认），像素低但完整图像
  3. 空间拓扑保持：相邻像素→相邻spike通道
  4. block_map局部连接：每种特征→独立SNN子群

设计原则（老板定的）：
  "哪怕像素低点也是完整图像，不能糊成一团"
  "加上色彩，三原色构成"
"""
import numpy as np
import torch
from scipy.ndimage import convolve, uniform_filter


class PrismVisualEncoderV2:
    """
    v2: RGB三原色 + 边缘 + 纹理 = 5通道
    输出维度：grid_h × grid_w × 5
    默认 8×8×5 = 320维 spike

    5通道仿生对应：
      R(红) — L型视锥(长波, ~564nm)
      G(绿) — M型视锥(中波, ~534nm)
      B(蓝) — S型视锥(短波, ~420nm)
      边缘  — 视网膜神经节细胞(on-off center-surround)
      纹理  — V1初级视皮层(纹理/频率检测)
    """

    SOBEL_H = np.array([[ 1,  2,  1],
                        [ 0,  0,  0],
                        [-1, -2, -1]], dtype=np.float32) / 8.0
    SOBEL_V = np.array([[ 1, 0, -1],
                        [ 2, 0, -2],
                        [ 1, 0, -1]], dtype=np.float32) / 8.0

    def __init__(self, grid_h=8, grid_w=8, noise_level=0.02):
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.n_features = 5  # R, G, B, edge, texture
        self.noise_level = noise_level
        self.spatial_dim = grid_h * grid_w
        self.total_dim = self.spatial_dim * self.n_features
        self.feature_names = ['red', 'green', 'blue', 'edge', 'texture']

    def get_block_map(self, n_neurons_per_feature=None):
        """
        生成SNN局部连接的block_map。
        每种视觉特征→独立的神经元子群，互不串扰。

        n_neurons_per_feature: 每个特征分配的SNN神经元数
                               默认=spatial_dim（1:1映射）
        返回: block_map list + 总SNN神经元数
        """
        if n_neurons_per_feature is None:
            n_neurons_per_feature = self.spatial_dim

        block_map = []
        for fi in range(self.n_features):
            in_start = fi * self.spatial_dim
            in_end = (fi + 1) * self.spatial_dim
            n_start = fi * n_neurons_per_feature
            n_end = (fi + 1) * n_neurons_per_feature
            block_map.append((in_start, in_end, n_start, n_end))

        total_neurons = self.n_features * n_neurons_per_feature
        return block_map, total_neurons

    def get_spatial_block_map(self, n_neurons_per_feature=None):
        """
        更细粒度：每种特征内，按空间区域分块连接。
        例如8×8网格分成4个4×4象限，每个象限→独立神经元子群。
        """
        if n_neurons_per_feature is None:
            n_neurons_per_feature = self.spatial_dim

        block_map = []
        # 4个象限
        quad_h = self.grid_h // 2
        quad_w = self.grid_w // 2
        neurons_per_quad = n_neurons_per_feature // 4

        neuron_offset = 0
        for fi in range(self.n_features):
            feat_offset = fi * self.spatial_dim
            for qr in range(2):
                for qc in range(2):
                    # 该象限在展平后的输入索引
                    in_indices = []
                    for r in range(qr * quad_h, (qr + 1) * quad_h):
                        for c in range(qc * quad_w, (qc + 1) * quad_w):
                            in_indices.append(feat_offset + r * self.grid_w + c)
                    # 连续区间近似（block_map要求连续范围）
                    in_start = min(in_indices)
                    in_end = max(in_indices) + 1
                    n_start = neuron_offset
                    n_end = neuron_offset + neurons_per_quad
                    block_map.append((in_start, in_end, n_start, n_end))
                    neuron_offset += neurons_per_quad

        return block_map, neuron_offset

    def _to_grid(self, feature_map):
        """降采样到grid_h×grid_w，保持空间布局"""
        h, w = feature_map.shape
        bh = max(1, h // self.grid_h)
        bw = max(1, w // self.grid_w)
        grid = np.zeros((self.grid_h, self.grid_w), dtype=np.float32)
        for r in range(self.grid_h):
            for c in range(self.grid_w):
                block = feature_map[r*bh:min((r+1)*bh, h), c*bw:min((c+1)*bw, w)]
                if block.size > 0:
                    grid[r, c] = block.mean()
        return grid

    def _extract_features(self, image_rgb):
        """
        RGB图→5通道物理特征，保持空间结构
        image_rgb: (H, W, 3) float32 [0,1]
        """
        r_ch = self._to_grid(image_rgb[:, :, 0])
        g_ch = self._to_grid(image_rgb[:, :, 1])
        b_ch = self._to_grid(image_rgb[:, :, 2])

        # 边缘：用亮度图算（和人眼一样，边缘检测靠明暗不靠颜色）
        gray = 0.299 * image_rgb[:,:,0] + 0.587 * image_rgb[:,:,1] + 0.114 * image_rgb[:,:,2]
        edge_h = np.abs(convolve(gray, self.SOBEL_H, mode='reflect'))
        edge_v = np.abs(convolve(gray, self.SOBEL_V, mode='reflect'))
        edge = self._to_grid(np.sqrt(edge_h**2 + edge_v**2))

        # 纹理：局部标准差
        local_mean = uniform_filter(gray, size=3)
        local_sq = uniform_filter(gray**2, size=3)
        local_std = np.sqrt(np.maximum(local_sq - local_mean**2, 0))
        texture = self._to_grid(local_std)

        return {'red': r_ch, 'green': g_ch, 'blue': b_ch,
                'edge': edge, 'texture': texture}

    def encode(self, image, timesteps=10):
        """
        图像→spike张量，RGB三原色+边缘+纹理
        image: (H,W,3) RGB [0,1]或[0,255], 或 (H,W) 灰度(自动扩展为灰色RGB)
        返回: (timesteps, total_dim) spike
        """
        rgb = self._to_rgb(image)
        features = self._extract_features(rgb)

        spike = torch.zeros(timesteps, self.total_dim)
        for fi, fname in enumerate(self.feature_names):
            grid = features[fname]
            values = grid.flatten()

            # RGB三通道保留绝对值，边缘和纹理归一化
            if fname in ('edge', 'texture'):
                vmin, vmax = values.min(), values.max()
                if vmax - vmin > 1e-6:
                    values = (values - vmin) / (vmax - vmin)
                else:
                    values = np.zeros_like(values)

            values = np.clip(values, 0, 1)
            noise = np.random.uniform(-self.noise_level, self.noise_level, values.shape)
            values = np.clip(values + noise, 0, 1)

            offset = fi * self.spatial_dim
            for ch in range(self.spatial_dim):
                if values[ch] > 0.02:
                    prob = min(0.85, values[ch] * 0.7)
                    spike[:, offset + ch] = (torch.rand(timesteps) < prob).float()

        return spike

    def decode_to_image(self, spike):
        """
        spike→RGB重建图像（验证SNN"看到"了什么）
        返回: (grid_h, grid_w, 3) numpy array [0,1]
        """
        if spike.dim() == 2:
            rates = spike.mean(dim=0)
        else:
            rates = spike

        img = np.zeros((self.grid_h, self.grid_w, 3), dtype=np.float32)
        for ci, color in enumerate(['red', 'green', 'blue']):
            offset = ci * self.spatial_dim
            channel_rates = rates[offset:offset + self.spatial_dim].numpy()
            img[:, :, ci] = np.clip(channel_rates / 0.7, 0, 1).reshape(self.grid_h, self.grid_w)
        return img

    def _to_rgb(self, image):
        """任意输入→(H,W,3) float32 [0,1]"""
        if isinstance(image, torch.Tensor):
            image = image.numpy()
        if image.ndim == 2:
            # 灰度→RGB（灰色）
            g = image / 255.0 if image.max() > 1.0 else image.astype(np.float32)
            image = np.stack([g, g, g], axis=2)
        elif image.max() > 1.0:
            image = image.astype(np.float32) / 255.0
        else:
            image = image.astype(np.float32)
        return np.clip(image, 0, 1)

    def describe(self, image, timesteps=10):
        rgb = self._to_rgb(image)
        features = self._extract_features(rgb)
        spike = self.encode(image, timesteps)
        recon = self.decode_to_image(spike)
        info = {}
        for fn, g in features.items():
            info[fn] = {'mean':float(g.mean()), 'std':float(g.std())}
        info['spike_sum'] = float(spike.sum())
        # RGB重建MSE
        orig_rgb_grid = np.stack([features['red'], features['green'], features['blue']], axis=2)
        info['recon_mse'] = float(((recon - orig_rgb_grid)**2).mean())
        return spike, info, recon
