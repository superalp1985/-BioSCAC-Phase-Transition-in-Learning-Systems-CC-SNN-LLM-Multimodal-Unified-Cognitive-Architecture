"""
BioSCAC 核心神经元与SCAC控制层
确定性LIF（D-LIF）：去除8条生物缺陷，保留7条FAME控制特性
"""
import torch
import torch.nn as nn
import numpy as np


class DLIFLayer(nn.Module):
    """确定性LIF神经元层 —— 无噪声、无不应期、无衰减"""
    
    def __init__(self, in_features, out_features, tau_m=10e-3, dt=1e-3,
                 v_threshold=1.0, v_reset=0.0, connectivity='full',
                 block_map=None):
        """
        connectivity: 'full'=全连接, 'block'=块对角精确投送
        block_map: list of (input_start, input_end, neuron_start, neuron_end)
                   定义哪些输入通道连接到哪些神经元
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.v_threshold = v_threshold
        self.v_reset = v_reset
        self.beta = np.exp(-dt / tau_m)
        self.connectivity = connectivity
        
        # 突触权重
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        nn.init.uniform_(self.weight, 0.3, 2.5)
        
        # 块对角连接掩码（精确投送）
        if connectivity == 'block' and block_map is not None:
            mask = torch.zeros(in_features, out_features)
            for i_start, i_end, n_start, n_end in block_map:
                mask[i_start:i_end, n_start:n_end] = 1.0
            self.register_buffer('mask', mask)
            # 把掩码外的权重清零
            with torch.no_grad():
                self.weight.data *= mask
        else:
            self.register_buffer('mask', torch.ones(in_features, out_features))
        
    def forward(self, spike_in, v_mem):
        """
        spike_in: (in_features,) 输入spike
        v_mem: (out_features,) 膜电位状态
        返回: (spike_out, v_mem_new)
        """
        # 应用连接掩码（精确投送：只有该连接的才有权重）
        effective_weight = self.weight * self.mask
        
        # 突触电流
        i_syn = spike_in @ effective_weight
        
        # 膜电位更新
        v_new = self.beta * v_mem + (1 - self.beta) * i_syn
        
        # 发放判定
        spike_out = (v_new >= self.v_threshold).float()
        
        # 重置
        v_new = v_new * (1 - spike_out) + self.v_reset * spike_out
        
        return spike_out, v_new


class SCACController:
    """
    SCAC控制层 —— 真正的Φ判定器 + κ自适应
    
    κ动态规则（仿生发育）：
    - 新奇度(novelty)驱动κ上升 → 遇到新刺激时重新打开学习通道
    - 稳定度(stability)驱动κ下降 → 已掌握的知识受保护
    - α > β（上升快，下降慢）→ 对新事物始终保持开放
    """
    
    def __init__(self, kappa_init=0.7, kappa_min=0.15, kappa_max=0.85,
                 alpha_novelty=0.06, beta_stability=0.015, gamma_oscillation=0.03,
                 phi_strictness=0.0):
        """
        kappa_init=0.7: OPT实验最优平衡点，对应人类婴儿可塑性
        kappa_min=0.15: 老了也能学（永不封死）
        kappa_max=0.85: 安全边界（OPT: κ≤0.8有余量）
        alpha: novelty上升速率（快）
        beta: stability下降速率（慢）
        gamma: oscillation下降速率（检测到振荡时压κ）
        """
        self.kappa = kappa_init
        self.kappa_min = kappa_min
        self.kappa_max = kappa_max
        self.alpha = alpha_novelty
        self.beta = beta_stability
        self.gamma = gamma_oscillation
        self.phi_strictness = phi_strictness
        
        # Adaptive tracking
        self.distance_ema = None  # Initialize from first observation
        self.ema_alpha = 0.2
        self.recent_distances = []  # Last N for oscillation detection
        self.window_size = 10
        
        # Stats
        self.total_updates = 0
        self.accepted_updates = 0
        self.rejected_updates = 0
        self.distance_history = []
        self.kappa_history = [kappa_init]
    
    def phi_judge(self, current_distance, previous_distance):
        """
        Phi judge + adaptive kappa update
        Three forces on kappa:
          novelty (distance spike) -> kappa UP
          stability (distance near 0) -> kappa DOWN slowly
          oscillation (distance bouncing, not decreasing) -> kappa DOWN fast
        """
        self.total_updates += 1
        self.distance_history.append(current_distance)
        self.recent_distances.append(current_distance)
        if len(self.recent_distances) > self.window_size:
            self.recent_distances.pop(0)
        
        # ---- Adaptive kappa ----
        
        # Initialize EMA from first real observation
        if self.distance_ema is None:
            self.distance_ema = current_distance
        
        # Novelty: sudden jump above running average
        novelty = max(0, current_distance - self.distance_ema * 1.5)
        novelty_norm = min(1.0, novelty / max(self.distance_ema, 1.0))
        
        # Stability: distance near zero
        stability = max(0, 1.0 - current_distance / max(self.distance_ema, 1.0))
        
        # Oscillation: high variance but no trend downward
        oscillation = 0.0
        if len(self.recent_distances) >= 6:
            recent = self.recent_distances
            half = len(recent) // 2
            first_half_mean = sum(recent[:half]) / half
            second_half_mean = sum(recent[half:]) / (len(recent) - half)
            variance = sum((d - second_half_mean)**2 for d in recent[half:]) / (len(recent) - half)
            std = variance ** 0.5
            # Oscillating = high variance + no improvement
            if std > 10 and second_half_mean >= first_half_mean * 0.9:
                oscillation = min(1.0, std / max(second_half_mean, 1.0))
        
        # Update EMA
        self.distance_ema = (1 - self.ema_alpha) * self.distance_ema + self.ema_alpha * current_distance
        
        # Kappa update: three forces
        delta_kappa = (
            self.alpha * novelty_norm          # UP: new stuff
            - self.beta * stability            # DOWN slowly: mastered
            - self.gamma * oscillation         # DOWN fast: overshooting
        )
        self.kappa = np.clip(self.kappa + delta_kappa, self.kappa_min, self.kappa_max)
        self.kappa_history.append(self.kappa)
        
        # ---- Φ判定（不变） ----
        improvement = previous_distance - current_distance
        accept = improvement > self.phi_strictness
        
        if accept:
            self.accepted_updates += 1
        else:
            self.rejected_updates += 1
        
        return accept
    
    def compress_update(self, delta_w):
        """T压缩：ΔW_actual = κ × ΔW，保证收敛"""
        return self.kappa * delta_w
    
    def get_stats(self):
        """返回Φ判定统计"""
        rate = self.accepted_updates / max(1, self.total_updates)
        return {
            'total': self.total_updates,
            'accepted': self.accepted_updates,
            'rejected': self.rejected_updates,
            'accept_rate': rate,
            'distance_history': self.distance_history,
            'kappa_history': self.kappa_history,
        }


class SCACSTDP:
    """
    SCAC约束的STDP学习规则
    标准STDP + Φ判定 + κ压缩 + FAME惯性
    """
    
    def __init__(self, a_plus=0.01, a_minus=0.008, scac: SCACController = None):
        self.a_plus = a_plus
        self.a_minus = a_minus
        self.scac = scac or SCACController()
    
    def compute_update(self, pre_spike, post_spike):
        """计算标准STDP权重变化"""
        # 简化STDP：pre和post同时发放→强化，只有post→弱化
        # delta_w shape: (in_features, out_features)
        potentiation = self.a_plus * torch.outer(pre_spike, post_spike)
        # depression比potentiation弱一点，保持总体兴奋性
        depression = -self.a_minus * torch.outer(1 - pre_spike, post_spike)
        return potentiation + depression
    
    def apply_update(self, layer: DLIFLayer, pre_spike, post_spike, 
                     output_distance, prev_distance, fame_xi=0.01):
        """
        带SCAC约束的权重更新
        返回: (accepted: bool, delta_w_norm: float)
        """
        # 1. 计算标准STDP的ΔW
        delta_w = self.compute_update(pre_spike, post_spike)
        
        # 2. Φ判定
        accepted = self.scac.phi_judge(output_distance, prev_distance)
        
        if accepted:
            # 3. κ压缩
            delta_w_compressed = self.scac.compress_update(delta_w)
            # 4. FAME惯性衰减
            delta_w_final = delta_w_compressed * (1 - fame_xi)
            # 5. 应用更新（尊重连接掩码）
            with torch.no_grad():
                layer.weight.data += delta_w_final * layer.mask
                # 权重裁剪
                layer.weight.data.clamp_(0.0, 2.0)
        
        return accepted, delta_w.norm().item()
