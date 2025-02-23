import numpy as np
import gym
import math
from typing import Dict
from gym import spaces
from utils import scale, plot_rslt
from fdm import Fdm3DoF

class PursuitEvasionGame(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array", None]}

    def __init__(self, render_mode=None):
        super(PursuitEvasionGame, self).__init__()
        self.episode = 0
        self.time_step = 0
        self.max_time_step = 1000
        self.total_reward = 0.0
        # 初期训练时奖励函数权重
        self.w1 = 0.1   # alpha限幅
        self.w2 = 0.1   # gamma限幅
        self.w3 = 0.1   # hdot限幅
        self.w4 = 0.1   # 油门限幅
        self.w5 = 0.3   # 侧偏距
        self.w6 = 0.3   # 爬升
        # 记录日志
        self.alpha_log = []
        self.beta_log = []
        self.thr_log = []
        self.gamma_log = []
        self.x_log = []
        self.y_log = []
        self.z_log = []
        self.render_mode = render_mode
        self.action_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)
        # 定义14个状态量的范围
        low = np.array([
        -10000.0,  # x
        -10000.0,  # y
        -80000.0,  # z
        -10000.0,  # x_t
        -10000.0,  # y_t
        -80000.0,  # z_t
        0.0,       # v
        np.deg2rad(-6.0),     # alpha
        np.deg2rad(-6.0),     # beta
        np.deg2rad(-20.0),     # gamma
        np.deg2rad(-180.0),    # chi
        np.deg2rad(-180.0),    # chi_t
        np.deg2rad(-30.0),     # mu
        0.0        # thr
    ], dtype=np.float32)
        
        high = np.array([
            10000.0,   # x
            10000.0,   # y
            0.0,   # z
            10000.0,   # x_t
            10000.0,   # y_t
            0.0,   # z_t
            180.0,     # v
            np.deg2rad(12.0),      # alpha
            np.deg2rad(6.0),      # beta
            np.deg2rad(20.0),      # gamma
            np.deg2rad(180.0),    # chi
            np.deg2rad(180.0),     # chi_t
            np.deg2rad(30.0),      # mu
            1.0        # thr
        ], dtype=np.float32)

        self.observation_space = spaces.Box(low, high, dtype=np.float32) # 14个状态 限幅TODO
        self.fdm = Fdm3DoF()
        
        # 初始化状态
        self.state = self.reset()

    def step(self, action):
        self.time_step += 1
        # 渲染
        if self.render_mode == 'human':
            self.render()

        # 将(-1,1)动作映射到真实物理量输入
        # action = scale(action)
        # 3dof模块更新状态
        self.state = self.fdm.run(self.state, action)
        # 记录日志
        self.alpha_log.append(np.rad2deg(self.state['alpha']))
        self.beta_log.append(np.rad2deg(self.state['beta']))
        self.thr_log.append(self.state['thr'])
        self.gamma_log.append(np.rad2deg(self.state['gamma']))
        self.x_log.append(self.state['x'])
        self.y_log.append(self.state['y'])
        self.z_log.append(self.state['z'])

        # 计算奖励
        reward = self.get_reward(self.state)
        # 检查是否完成
        done = self.get_done(self.state)
        # 获取额外信息
        info = self.get_info(self.state)

        observation = self.get_observation()
        # 累计奖励 用于日志记录
        self.total_reward += reward
        return observation, reward, done, info

    def reset(self):
        self.time_step = 0
        self.prev_distance = 0
        self.total_reward = 0
        # 重置日志
        self.alpha_log = []
        self.beta_log = []
        self.thr_log = []
        self.gamma_log = []
        self.x_log = []
        self.y_log = []
        self.z_log = []
        # 定义初始状态
        initial_state: Dict[str, float] = {
            'x': 0.0,
            'y': 0.0,
            'z': -4000.0,      # 初始高度 4000 米
            'x_t': 3000.0,
            'y_t': 0.0, 
            'z_t': -5000.0,   # 目标高度 5000 米
            'v': 100.0,        # 初始速度
            'alpha': np.deg2rad(4.0),   # 初始攻角 4°
            'beta': np.deg2rad(0.0),    # 初始侧滑角 0°
            'gamma': np.deg2rad(0.0),   # 初始航迹倾角 0°
            'chi': np.deg2rad(0.0),    # 初始航向 90°
            'chi_t': np.deg2rad(0.0),      # 目标航向 90°
            'mu': np.deg2rad(0.0),    # 初始绕速度滚转角 0°
            'thr': 0.7        # 初始油门 0.7
        }

        self.state = initial_state
        observation = self.get_observation()
        return observation
    
    def get_observation(self):
        state = self.state
        observation = np.array([
            state['x'], state['y'], state['z'],
            state['x_t'], state['y_t'], state['z_t'],
            state['v'], state['alpha'], state['beta'],
            state['gamma'], state['chi'], state['chi_t'],
            state['mu'], state['thr']
        ], dtype=np.float32)

        return observation

    def render(self, mode='human'):
        # 启用渲染开关，在每个回合结束后画图
        plot_rslt(self.alpha_log,
                    self.beta_log,
                    self.thr_log,
                    self.gamma_log,
                    self.x_log,
                    self.y_log,
                    self.z_log,
                    self.total_reward)

    
    def get_reward(self, state):
        reward = 0.0
        persuer_pos = np.array([state['x'], state['y'], state['z']])
        evader_pos = np.array([state['x_t'], state['y_t'], state['z_t']])
        distance = np.linalg.norm(persuer_pos - evader_pos)
        # 接近敌机奖励
        if self.prev_distance is not None:
            if distance <= self.prev_distance:
                reward += 1e-3
            else:
                # reward -= 1e-9
                None
            self.prev_distance = distance

        # 击中奖励
        if distance < 100:
            reward += 10

        # 过低高度惩罚
        if state['z'] > -1000:
            reward -= 1

        # 逃脱惩罚
        if distance > 15000:
            reward -= 1

        self.total_reward += reward

        return self.total_reward

    def get_done(self, state):
        persuer_pos = np.array([state['x'], state['y'], state['z']])
        evader_pos = np.array([state['x_t'], state['y_t'], state['z_t']])
        distance = np.linalg.norm(persuer_pos - evader_pos)

        if distance < 10:
            return True
        if self.time_step > self.max_time_step:
            print(f"超过最大步数,距离: {distance}")
            return True
        if state['z'] > -1000:
            print(f"过低高度: {-state['z']} 步数: {self.time_step} alpha: {np.rad2deg(state['alpha']):.2f}\°\
 beta: {np.rad2deg(state['beta']):.2f}° gamma: {np.rad2deg(state['gamma']):.2f}° thr: {state['thr']:.2f}")
            return True
        if distance > 15000:
            print(f"逃脱攻击区:{distance} 步数: {self.time_step} alpha: {np.rad2deg(state['alpha']):.2f}°\
 beta: {np.rad2deg(state['beta']):.2f}° gamma: {np.rad2deg(state['gamma']):.2f}° thr: {state['thr']:.2f}")
            return True
        return False

    def get_info(self, state):
        return {}
    
class StableFlyingEnv(PursuitEvasionGame):
    def __init__(self, render_mode=None):
        super(StableFlyingEnv, self).__init__(render_mode)
        # 继承PEG，课程学习的初始阶段，只要求无人机平稳飞行

    def get_reward(self, state):
        reward_init = 2e-3
        step_reward = 0.0  # 当前步骤的即时奖励

        # 限制迎角
        if state['alpha'] > np.deg2rad(-3) and state['alpha'] < np.deg2rad(6):
            step_reward += reward_init * self.w1
        else:
            step_reward -= reward_init * self.w1

        # 限制航迹倾角
        if -np.deg2rad(15) < state['gamma'] < np.deg2rad(15):
            step_reward += reward_init * self.w2
        else:
            step_reward -= reward_init * self.w2

        # 限制爬升率
        h_dot = state['v'] * math.sin(state['gamma'])
        if 0 < h_dot < 40:
            step_reward += reward_init * self.w3
        else:
            step_reward -= reward_init * self.w3

        # 保持油门
        if state['thr'] >= 0.66:
            step_reward += reward_init * self.w4
        else:
            step_reward -= reward_init * self.w4

        # 限制侧偏距
        if abs(state['y']) < 100:
            step_reward += reward_init * self.w5
        else:
            step_reward -= reward_init * self.w5

        # 鼓励爬升
        if 0 < state['gamma'] < np.deg2rad(20):
            step_reward += reward_init * self.w6
        else:
            step_reward -= reward_init * self.w6

        return step_reward  # 返回当前步骤的即时奖励
    
    def get_done(self, state):
        # 指定飞行时间结束
        # if self.time_step >= self.max_time_step:
#             self.episode += 1
#             print(f"Episode:{self.episode} Reward:{self.total_reward:.2f} alpha: {np.rad2deg(state['alpha']):.2f}°\
#  beta: {np.rad2deg(state['beta']):.2f}° gamma: {np.rad2deg(state['gamma']):.2f}° y: {state['y']:.2f}m\
#  hdot: {math.sin(state['gamma']) * state['v']:.2f}m/s thr: {state['thr']:.2f}")
#             return True
#         return False
        return self.time_step >= self.max_time_step

    
class TargetFlyingEnv(PursuitEvasionGame):
    def __init__(self, render_mode=None):
        super(TargetFlyingEnv, self).__init__(render_mode)
        # 添加目标点的飞行任务

    def get_reward(self, state):
        reward = super().get_reward(state)  # 使用父类的奖励函数
        target_position = np.array([5000.0, 0.0, -4000.0])  # 设定目标点
        current_position = np.array([state['x'], state['y'], state['z']])
        distance_to_target = np.linalg.norm(current_position - target_position)
        if distance_to_target < 500:
            reward += 10  # 达到目标点奖励
        return reward

    def get_done(self, state):
        # 达到目标点时任务结束
        target_position = np.array([5000.0, 0.0, -4000.0])
        current_position = np.array([state['x'], state['y'], state['z']])
        if np.linalg.norm(current_position - target_position) < 500:
            return True
        return False