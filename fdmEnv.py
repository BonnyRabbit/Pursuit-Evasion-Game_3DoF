import numpy as np
import gym
from typing import Dict
from gym import spaces
from utils import scale
from fdm import Fdm3DoF

class PursuitEvasionGame(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array", None]}

    def __init__(self, render_mode=None):
        super(PursuitEvasionGame, self).__init__()

        self.render_mode = render_mode
        self.action_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-500, high=500, shape=(13,), dtype=np.float32) # 13个状态 限幅TODO
        self.fdm = Fdm3DoF()
        
        # 初始化状态
        self.state = self.reset()

    def step(self, action):

        # 渲染
        if self.render_mode == 'human':
            self.render()

        # 将(-1,1)动作映射到真实物理量输入
        # action = scale(action)
        # 3dof模块更新状态
        self.state = self.fdm.run(self.state, action)
        # 计算奖励
        reward = self.get_reward(self.state)
        # 检查是否完成
        done = self.get_done(self.state)
        # 获取额外信息
        info = self.get_info(self.state)

        return self.get_observation(), reward, done, info

    def reset(self):
        # 定义初始状态
        initial_state: Dict[str, float] = {
            'x': 0.0,
            'y': 0.0,
            'z': -4000.0,      # 初始高度 4000 米
            'x_t': 1e4,
            'y_t': 0.0, 
            'z_t': -5000.0,   # 目标高度 5000 米
            'v': 100.0,        # 初始速度
            'alpha': 4.0,   # 初始攻角 4°
            'beta': 0.0,    # 初始侧滑角 0°
            'gamma': 0.0,   # 初始航迹倾角 0°
            'chi': 90.0,    # 初始航向 90°
            'chi_t': 90.0,      # 目标航向 90°
            'mu': 0.0,    # 初始绕速度滚转角 0°
            'thr': 0.7        # 初始油门 70%
        }

        self.state = initial_state
        return self.get_observation()
    
    def get_observation(self):
        state = self.state
        observation = np.array([
            state['x'], state['y'], state['z'],
            state['x_t'], state['y_t'], state['z_t'],
            state['v'], state['alpha'], state['beta'],
            state['gamma'], state['chi'], state['chi_t'],
            state['mu']
        ], dtype=np.float32)

        return observation

    def render(self, mode='human'):
        pass
    
    def get_reward(self, state):
        pass

    def get_done(self, state):
        pass

    def get_info(self, state):
        return {}