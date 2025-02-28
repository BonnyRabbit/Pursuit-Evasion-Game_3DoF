import os
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, List

class TensorboardTimeSeriesCallback(BaseCallback):
    """
    自定义时间序列记录回调，支持：
    - 多环境并行训练
    - 自动数据转换（角度/高度处理）
    - 关键指标统计
    - 数据采样控制
    """
    
    def __init__(self, 
                 log_dir: str, 
                 log_freq: int = 20,  # 记录频率
                 verbose: int = 0):
        super().__init__(verbose)
        self.log_dir = os.path.join(log_dir, "custom_metrics")
        self.log_freq = log_freq
        self.writer = None
        
        self.episode_buffers: Dict[int, dict] = {}
        self.episode_counter = 0
        
        self.action_names = ['dalpha', 'dbeta', 'dthr']
        self.obs_names = [
            'x', 'y', 'z', 'x_t', 'y_t', 'z_t', 'v',
            'alpha', 'beta', 'gamma', 'chi', 'chi_t', 'mu', 'thr',
            'rel_chi','rel_dist'
        ]
        
    def _init_callback(self) -> None:
        os.makedirs(self.log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.log_dir)
        
    def _convert_value(self, name: str, value: float) -> float:
        if name in ['dalpha', 'dbeta', 'alpha', 'beta', 'gamma', 'chi', 'chi_t', 'mu', 'rel_chi']:
            return np.rad2deg(value)
        elif name in ['z', 'z_t']:
            return -value
        return value
    
    def _on_step(self) -> bool:
        for env_idx in range(self.training_env.num_envs):
            done = self.locals['dones'][env_idx]
            
            if env_idx not in self.episode_buffers:
                self.episode_buffers[env_idx] = {
                    'actions': [],
                    'observations': [],
                    'rewards': []
                }
                
            buffer = self.episode_buffers[env_idx]
            buffer['actions'].append(self.locals['actions'][env_idx])
            buffer['observations'].append(self.locals['new_obs'][env_idx])
            buffer['rewards'].append(self.locals['rewards'][env_idx])
            
            if done:
                current_episode = self.episode_counter
                self._log_episode(env_idx, current_episode)
                self.episode_counter += 1  # 先记录再增加计数器
                self.episode_buffers[env_idx] = {'actions': [], 'observations': [], 'rewards': []}
                
        return True
    
    def _log_episode(self, env_idx: int, episode_num: int) -> None:
        """记录指定回合号的数据"""
        if episode_num % self.log_freq != 0:
            return
                
        buffer = self.episode_buffers[env_idx]
        
        obs = np.array(buffer['observations'])
            
        #  action和observation中的数据
        for step, (act, obs) in enumerate(zip(buffer['actions'], buffer['observations'])):
            # 动作序列
            for name, value in zip(self.action_names, act):
                self.writer.add_scalar(
                    f"episode_{episode_num}/actions_t/{name}",
                    self._convert_value(name, value),
                    step
                )
            # 观测序列
            for name, value in zip(self.obs_names, obs):
                self.writer.add_scalar(
                    f"episode_{episode_num}/observations_t/{name}",
                    self._convert_value(name, value),
                    step
                )

    def _on_close(self) -> None:
        for env_idx in self.episode_buffers:
            if len(self.episode_buffers[env_idx]['rewards']) > 0:
                self._log_episode(env_idx, self.episode_counter)
                self.episode_counter += 1
        if self.writer:
            self.writer.close()