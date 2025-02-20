from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
import os
from torch.utils.tensorboard import SummaryWriter

class TensorboardTimeSeriesCallback(BaseCallback):
    """
    自定义回调类,用于记录observation和action到tenserboard,物理时间作为横坐标
    """
    def __init__(self, log_dir, dt_AP=0.005, verbose=0):
        """
        :param log_dir: tensorboard 日志保存路径
        :param dt_AP: 每次模拟的时间步长
        :param verbose: 控制日志输出的详细程度
        """
        super(TensorboardTimeSeriesCallback, self).__init__(verbose)
        self.dt_AP =  dt_AP
        self.log_dir = log_dir
        self.writer = SummaryWriter(log_dir=log_dir)
        self.total_time = 0.0  # 记录物理时间
        self.current_episode = 0 # 用于区分回合
        self.action_names = ['alpha', 'beta', 'thr']
        self.observation_names = ['x', 'y', 'z', 'x_t', 'y_t', 'z_t', 'v', 'alpha', 'beta', 'gamma', 'chi', 'chi_t', 'mu', 'thr']

    def _on_step(self) -> bool:
        """
        获取当前的 Rollout Buffer，提取actions 和 observations，计算平均值后记录到tensorboard
        """
        rollout_buffer = self.locals.get("rollout_buffer", None)
        if rollout_buffer is not None:
            action = rollout_buffer.actions # shape: (n_steps, n_envs, action_dim)
            observation = rollout_buffer.observations # shape: (n_steps, n_envs, obs_dim)

            # 计算回调涉及的步数和物理时间
            n_steps, n_envs = rollout_buffer.actions.shape[:2]
            step_time = n_steps * n_envs * self.dt_AP
            self.total_time += step_time

            # 将数据转化为numpy数组
            action = action.reshape(-1, action.shape[-1])
            observation = observation.reshape(-1, observation.shape[-1])
            # 计算每个维度的平均值（你也可以计算其他统计量，如标准差等）
            avg_action = np.mean(action, axis=0)       # shape: (action_dim,)
            avg_observation = np.mean(observation, axis=0)  # shape: (obs_dim,)
            
            # 将数据记录到tensorboard
            for i, val in enumerate(avg_action):
                action_name = self.action_names[i]
                # action_name = self.action_names[i] if i < len(self.action_names) else f"Action_{i}"
                self.writer.add_scalar(f"{action_name}_Ep_{self.current_episode + 1}", val, self.total_time)

            for i, val in enumerate(avg_observation):
                observation_name = self.observation_names[i]
                # observation_name = self.observation_names[i] if i < len(self.observation_names) else f"Observation_{i}"
                self.writer.add_scalar(f"{observation_name}_Ep_{self.current_episode + 1}", val, self.total_time)
        
        if self.locals.get('done', None) is not None and np.any(self.locals.get('done')):
            # 回合结束时调用_on_episode_end
            self._on_episode_end()

        return True
    def _on_episode_end(self) -> None:
        """
        每个回合结束后，更新回合数，并在tb上标记
        """
        self.current_episode += 1
        self.writer.add_scalar("Episode/TotalTime", self.total_time, self.current_episode)
        self.total_time = 0.0

    def _on_training_end(self) -> None:
        """
        训练结束后关闭SummaryWriter
        """
        self.writer.close()