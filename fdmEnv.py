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
        80.0,       # v
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
            250.0,     # v
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
        reward = self.get_reward(self.state, action)
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
            'z': -1000.0,      # 初始高度 1000 米
            'x_t': 3000.0,
            'y_t': 0.0, 
            'z_t': -5000.0,   # 目标高度 5000 米
            'v': 165.0,        # 初始速度
            'alpha': np.deg2rad(3.0),   # 初始攻角 3°
            'beta': np.deg2rad(0.0),    # 初始侧滑角 0°
            'gamma': np.deg2rad(3.0),   # 初始航迹倾角 0°
            'chi': np.deg2rad(0.0),    # 初始航向 0°
            'chi_t': np.deg2rad(0.0),      # 目标航向 0°
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

    
    def get_reward(self, state, action):
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
        # 调整权重系数
        self.alpha_range = (2.0, 4.0)  # 目标迎角范围
        self.target_speed = 170.0      # 目标巡航速度
        self.w = {
            'alpha': 2.0,   # 迎角核心指标
            'beta': 1.5,     # 侧滑角零值保持
            'gamma': 5.0,    # 航迹倾角零值保持
            'speed': 3,    # 速度保持
            'offset': 0.8,   # 侧偏距抑制
            'action': 0.1,  # 动作平滑
            'thr':   0.5     # 油门限幅
        }

    def get_reward(self, state, action):
        # 单位统一转换为角度制
        alpha_deg = np.rad2deg(state['alpha'])
        beta_deg = np.rad2deg(state['beta'])
        gamma_deg = np.rad2deg(state['gamma'])
        
        # 核心奖励项设计
        # 1. 迎角维持奖励（钟形曲线奖励）
        alpha_center = (self.alpha_range[0] + self.alpha_range[1])/2
        alpha_dev = max(self.alpha_range[1] - self.alpha_range[0], 1e-5)
        alpha_reward = np.exp(-4*((alpha_deg - alpha_center)/alpha_dev)**2)
        
        # 2. 侧滑角抑制（二次惩罚）
        beta_penalty = (beta_deg/6.0)**2  # 归一化到0-1范围
        
        # 3. 航迹倾角抑制（二次惩罚）
        gamma_penalty = (gamma_deg/20.0)**2  # 归一化到0-1范围
        
        # 4. 速度保持奖励（分段函数）
        speed_error = abs(state['v'] - self.target_speed)
        if speed_error <= 5:  # 允许±5m/s误差
            speed_reward = 1.0
        else:
            speed_reward = np.exp(-0.01*speed_error)
        
        # 5. 侧向偏移抑制（指数衰减）
        lateral_offset = abs(state['y'] * math.sin(state['chi']))
        offset_reward = np.exp(-0.0001*lateral_offset)
        
        # 6. 动作平滑惩罚（变化率抑制）
        if hasattr(self, 'prev_action'):
            action_change = np.mean(np.abs(action - self.prev_action))
        else:
            action_change = 0.0
        self.prev_action = action.copy()

        # 7. 油门限幅(分段函数)
        # 0.5~0.9获得最大奖励,<0.4或>0.9惩罚
        thr = state['thr']
        if 0.5 <= thr <= 0.9:
            thr_reward = np.exp(-5*(thr-0.7)**2)
        elif thr < 0.4:
            thr_reward = -2.0 + (thr - 0.4) * 5
        else:
            thr_reward = -abs(thr - 0.7)
        
        # 综合奖励计算
        reward = (
            self.w['alpha'] * alpha_reward
            - self.w['beta'] * beta_penalty
            - self.w['gamma'] * gamma_penalty
            + self.w['speed'] * speed_reward
            + self.w['offset'] * offset_reward
            - self.w['action'] * (np.mean(action**2) + 0.5*action_change)
            + self.w['thr'] * thr_reward
        )
        
        # 边界条件惩罚（硬约束）
        if state['z'] > -1000:  # 高度低于1000米
            reward -= 10.0
        if abs(beta_deg) > 6.0:  # 侧滑角超过±6°
            reward -= 3.0
        if abs(gamma_deg) > 25.0:   # 航迹倾角超过±25°
            reward -= 3.0

        return reward

    
    def get_done(self, state):
        # 指定飞行时间结束/增加任务结束条件
        if state['z'] > -1000 or abs(np.rad2deg(state['gamma'])) > 25 or abs(np.rad2deg(state['beta'])) > 3.0 or self.time_step >= self.max_time_step:
            return True
        return super().get_done(state)

    
class TargetFlyingEnv(PursuitEvasionGame):
    def __init__(self, render_mode=None):
        super(TargetFlyingEnv, self).__init__(render_mode)
        # 添加目标点的飞行任务

    def get_reward(self, state):
        reward = super().get_reward(state)  # 使用父类的奖励函数
        target_position = np.array([15000.0, 0.0, -8000.0])  # 设定目标点
        current_position = np.array([state['x'], state['y'], state['z']])
        distance_to_target = np.linalg.norm(current_position - target_position)
        if distance_to_target < 200:
            reward += 10  # 达到目标点奖励
        return reward

    def get_done(self, state):
        # 达到目标点时任务结束
        target_position = np.array([5000.0, 0.0, -4000.0])
        current_position = np.array([state['x'], state['y'], state['z']])
        if np.linalg.norm(current_position - target_position) < 500:
            return True
        return False