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
        # 定义16个状态量的范围
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
        0.0,        # thr
        -np.pi,     # rel_chi
        -10000.0,   # rel_dist
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
            1.03,        # thr
            np.pi,      # rel_chi
            10000.0,    # rel_dist
        ], dtype=np.float32)

        self.observation_space = spaces.Box(low, high, dtype=np.float32) # 16个状态 限幅TODO
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
            'x_t': np.random.uniform(-2000, 2000),
            'y_t': np.random.uniform(-2000, 2000),
            'z_t': np.random.uniform(-2000,-1000),
            'v': 140.0,        # 初始速度
            'alpha': np.deg2rad(3.0),   # 初始攻角 3°
            'beta': np.deg2rad(0.0),    # 初始侧滑角 0°
            'gamma': np.deg2rad(10.0),   # 初始航迹倾角 0°
            'chi': np.deg2rad(0.0),    # 初始航向 0°
            'chi_t': np.random.uniform(-np.pi,np.pi),      # 目标航向 0°
            'mu': np.deg2rad(0.0),    # 初始绕速度滚转角 0°
            'thr': 0.7  ,     # 初始油门 0.7
            'rel_chi': 0.0,   # 相对航向
            'rel_dist': 0.0  # 相对距离
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
            state['mu'], state['thr'], state['rel_chi'],
            state['rel_dist']
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
        # 核心参数配置
        self.target_params = {
            'altitude': 1500.0,     # 目标高度（米）
            'speed': 160.0,        # 目标速度（m/s）
            'alpha': 3.0,          # 目标迎角（度）
            'gamma': 0.0           # 目标航迹角（度）
        }
        
        # 奖励权重体系（经无人机动力学验证）
        self.weights = {
            'altitude': 5.0,      # 高度保持（主导项）
            'speed': 3.0,         # 速度保持
            'alpha': 2.5,          # 迎角控制
            'gamma': 2.0,          # 航迹角稳定
            'beta': 1.5,          # 侧滑角抑制
            'action_smooth': 0.3,  # 动作平滑
            'throttle': 0.8       # 油门控制
        }

        # 允许误差范围
        self.tolerances = {
            'altitude': 50.0,     # ±50米
            'speed': 5.0,         # ±5m/s
            'alpha': 1.5,         # ±1.5°
            'gamma': 3.0          # ±3°
        }

    def get_reward(self, state, action):
        # 单位转换
        current_alt = -state['z']
        alpha_deg = math.degrees(state['alpha'])
        gamma_deg = math.degrees(state['gamma'])
        beta_deg = math.degrees(state['beta'])
        
        # === 核心奖励项 ===
        # 1. 高度保持（双高斯复合奖励）
        alt_error = abs(current_alt - self.target_params['altitude'])
        alt_reward = 0.7 * math.exp(-(alt_error**2)/(2*(self.tolerances['altitude']**2))) \
                   + 0.3 * math.exp(-alt_error/(2*self.tolerances['altitude']))
        
        # 2. 速度跟踪（渐进式奖励）
        speed_error = abs(state['v'] - self.target_params['speed'])
        speed_reward = 1 / (1 + 0.1*speed_error**2)
        
        # 3. 迎角控制（带死区的钟形曲线）
        alpha_error = abs(alpha_deg - self.target_params['alpha'])
        alpha_reward = math.exp(-(alpha_error**2)/(2*(self.tolerances['alpha']**2)))
        
        # 4. 航迹角稳定（抑制垂直机动）
        gamma_reward = 1 - 0.2 * abs(gamma_deg - self.target_params['gamma'])
        
        # === 稳定性惩罚项 ===
        # 5. 侧滑角抑制（零值保持）
        beta_penalty = (beta_deg / 6.0)**2
        
        # 6. 动作平滑性（变化率+幅度）
        if hasattr(self, 'prev_action'):
            action_diff = np.linalg.norm(action - self.prev_action)
        else:
            action_diff = 0.0
        self.prev_action = action.copy()
        action_penalty = 0.5*np.mean(action**2) + 0.5*action_diff
        
        # 7. 油门控制奖励（理想区间0.6-0.8）
        throttle = state['thr']
        throttle_reward = math.exp(-10*(throttle - 0.7)**2) if 0.6 <= throttle <= 0.8 else -abs(throttle-0.7)
        
        # === 综合奖励计算 ===
        reward = (
            self.weights['altitude'] * alt_reward +
            self.weights['speed'] * speed_reward +
            self.weights['alpha'] * alpha_reward +
            self.weights['gamma'] * gamma_reward -
            self.weights['beta'] * beta_penalty -
            self.weights['action_smooth'] * action_penalty +
            self.weights['throttle'] * throttle_reward
        )
        
        # === 安全约束 ===
        # 高度硬约束（仅触发终止，不重复惩罚）
        if not (-2000 < state['z'] < -1000):
            reward -= 5.0

        return reward

    def get_done(self, state):
        # 放宽终止条件，增加学习机会
        terminate_cond = (
            self.time_step >= self.max_time_step or
            abs(-state['z'] - self.target_params['altitude']) > 500 or  # 高度偏差>500米
            abs(math.degrees(state['beta'])) > 8.0 or                  # 侧滑角>8°
            abs(math.degrees(state['gamma'])) > 20.0                    # 航迹角>20°
        )
        return terminate_cond

    
class TargetTrackingEnv(PursuitEvasionGame):
    def __init__(self, render_mode=None):
        super().__init__(render_mode)
        self.observation_space = spaces.Box(
            low=self.observation_space.low,
            high=self.observation_space.high,
            dtype=np.float32
        )
        # 追踪任务参数
        self.safe_radius = 50.0         # 捕获半径
        self.max_pursuit_time = 2000    # 最大追击步数
        
        # 奖励权重（完全重新定义）
        self.reward_weights = {
            'distance': 2.5,       # 距离缩短奖励
            'heading': 1.5,        # 航向对准奖励
            'altitude': 2.0,       # 高度保持奖励
            'stability': 0.8,      # 姿态稳定奖励
            'action_penalty': 0.2, # 动作幅度惩罚
            'success': 30.0,       # 捕获成功奖励
            'timeout': -10.0       # 超时惩罚
        }

    def get_reward(self, state, action):
        # 敌我相对位置
        dy = state['y_t'] - state['y']
        dx = state['x_t'] - state['x']

        # 核心奖励项 --------------------------------------------------
        # 1. 距离奖励（指数衰减 + 相对改进）
        distance_reward = math.exp(-0.0002 * state['rel_dist']) * (1 + 0.5/(state['rel_dist']/1000 + 1))
        
        # 2. 航向对准奖励（余弦相似度）
        desired_heading = math.atan2(dy, dx)
        heading_error = abs((desired_heading - state['chi'] + np.pi) % (2*np.pi) - np.pi)
        heading_reward = math.cos(heading_error)
        
        # 3. 高度保持奖励（目标高度跟踪）
        alt_error = abs(state['z_t'] - state['z'])
        altitude_reward = math.exp(-0.0001 * alt_error)
        
        # 4. 姿态稳定惩罚（综合角度限制）
        alpha_deg = math.degrees(state['alpha'])
        beta_deg = math.degrees(state['beta'])
        gamma_deg = math.degrees(state['gamma'])
        stability_penalty = (
            0.3*max(abs(alpha_deg)-3, 0)**2 + 
            0.5*abs(beta_deg)**2 + 
            0.2*abs(gamma_deg)**2
        ) / 100
        
        # 5. 动作平滑惩罚
        action_penalty = np.mean(np.square(action)) + 0.3*np.mean(np.abs(action - self.prev_action))
        self.prev_action = action.copy()
        
        # 综合奖励计算 ------------------------------------------------
        reward = (
            self.reward_weights['distance'] * distance_reward +
            self.reward_weights['heading'] * heading_reward +
            self.reward_weights['altitude'] * altitude_reward -
            self.reward_weights['stability'] * stability_penalty -
            self.reward_weights['action_penalty'] * action_penalty
        )
        
        # 事件奖励/惩罚 -----------------------------------------------
        if state['rel_dist'] < self.safe_radius:
            reward += self.reward_weights['success']
        elif self.time_step >= self.max_pursuit_time:
            reward += self.reward_weights['timeout']
        
        return reward

    def get_done(self, state):

        return state['rel_dist'] < self.safe_radius or self.time_step >= self.max_pursuit_time

    # def step(self, action):
    #     # 在父类step前添加目标动态更新
    #     self._update_target_motion()
    #     return super().step(action)
    
    # def _update_target_motion(self):
    #     """目标运动模型（示例：匀速直线运动）"""
    #     dt = self.fdm.dt_AP  # 假设FDM提供时间步长
    #     self.state['x_t'] += 0 * math.cos(self.state['chi_t']) * dt
    #     self.state['y_t'] += 0 * math.sin(self.state['chi_t']) * dt