"""
BioSCAC FAME 8D情感控制器
控制7条保留的生物特性，每个学习周期（不是每个时间步）更新一次
"""
import torch
import numpy as np


class FAMEController:
    """
    FAME 8D情感状态控制器
    更新频率：每个学习周期（一次完整的输入序列处理后），不是每个spike时间步
    """
    
    def __init__(self):
        # 8D状态向量
        self.mu = 0.5       # 满足感
        self.chi = 0.3      # 好奇心
        self.epsilon = 0.5  # 共情
        self.kappa = 0.7    # OPT optimal, synced from SCAC
        self.nu = 0.0       # 焦虑
        self.delta = 0.0    # 冲突
        self.rho = 0.0      # 疲劳
        self.lam = 0.0      # 挫败
        
        # 历史记录
        self.history = []
        self._record()
    
    def update(self, accepted, total_attempts, left_output=None, right_output=None,
               total_spikes=0, max_possible_spikes=1, scac_kappa=None):
        """
        每个学习周期结束后调用一次
        accepted: 本周期Φ接受的更新数
        total_attempts: 本周期总更新数
        scac_kappa: SCAC的自适应κ值（如果提供，同步到FAME）
        """
        success_rate = accepted / max(1, total_attempts)
        
        # μ满足 = 成功率的sigmoid
        self.mu = 0.7 * self.mu + 0.3 * success_rate
        
        # κ自信 = 从SCAC同步（SCAC是唯一的κ权威来源）
        if scac_kappa is not None:
            self.kappa = scac_kappa
        
        # χ好奇 = 1 - κ（不自信时更好奇）+ 小随机扰动
        self.chi = max(0, min(1, (1 - self.kappa) * 0.8 + np.random.uniform(-0.02, 0.02)))
        
        # ε共情 = 左右脑输出一致性
        if left_output is not None and right_output is not None:
            l_norm = left_output.norm()
            r_norm = right_output.norm()
            if l_norm > 0 and r_norm > 0:
                cos_sim = torch.nn.functional.cosine_similarity(
                    left_output.flatten().unsqueeze(0),
                    right_output.flatten().unsqueeze(0)
                ).item()
                self.epsilon = 0.7 * self.epsilon + 0.3 * ((cos_sim + 1) / 2)
        
        # κ自信由SCAC驱动（已在上面同步），不再自己调
        
        # ν焦虑 = 低成功率时升高
        self.nu = 0.8 * self.nu + 0.2 * (1 - success_rate) * 0.5
        
        # δ冲突 = 左右脑矛盾检测
        if left_output is not None and right_output is not None:
            l_norm = left_output.norm()
            r_norm = right_output.norm()
            if l_norm > 0 and r_norm > 0:
                cos_sim = torch.nn.functional.cosine_similarity(
                    left_output.flatten().unsqueeze(0),
                    right_output.flatten().unsqueeze(0)
                ).item()
                self.delta = max(0, -cos_sim)
            else:
                self.delta = 0.0
        
        # ρ疲劳 = 累计spike比例
        spike_ratio = total_spikes / max(1, max_possible_spikes)
        self.rho = 0.9 * self.rho + 0.1 * spike_ratio
        
        # λ挫败 = 持续不满足
        self.lam = 0.9 * self.lam + 0.1 * (1 - self.mu)
        
        self._record()
    
    def get_learning_modulation(self):
        """返回对学习参数的调制系数"""
        return {
            'lr_scale': 1.0 + 0.3 * self.chi - 0.2 * self.rho,      # 好奇加速，疲劳减速
            'search_range': (1 + self.chi) / (1 + self.kappa),        # 好奇扩大，自信缩小
            'xi_inertia': 0.01 + 0.05 * self.kappa,                  # 自信时惯性大（保守）
            'forget_rate': 0.001 * (1 + 0.2 * self.lam),             # 挫败加速遗忘
            'exchange_k': max(1, int(32 * (1 - self.kappa))),         # 低自信多交换
        }
    
    def get_state_vector(self):
        """返回8D状态向量"""
        return np.array([self.mu, self.chi, self.epsilon, self.kappa,
                         self.nu, self.delta, self.rho, self.lam])
    
    def _record(self):
        self.history.append(self.get_state_vector().copy())
    
    def get_history_array(self):
        return np.array(self.history)
