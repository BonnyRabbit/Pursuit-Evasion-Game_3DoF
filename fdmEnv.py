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
        self.state_prev = None

        self.render_mode = render_mode
        self.action_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)
        # 定义16个状态量的范围
        low = np.array([
        -15000.0*9.8/(340**2),  # x
        -15000.0*9.8/(340**2),  # y
        -10000.0*9.8/(340**2),  # z
        -15000.0*9.8/(340**2),  # x_t
        -15000.0*9.8/(340**2),  # y_t
        -10000.0*9.8/(340**2),  # z_t
        80.0/340,               # v
        np.deg2rad(-6.0),     # alpha
        np.deg2rad(-6.0),     # beta
        np.deg2rad(-20.0),     # gamma
        np.deg2rad(-180.0),    # chi
        np.deg2rad(-180.0),    # chi_t
        np.deg2rad(-30.0),     # mu
        0.0,                     # thr
        -np.pi,                 # rel_chi
        -10000.0*9.8/(340**2),   # rel_dist
    ], dtype=np.float32)
        
        high = np.array([
            15000.0*9.8/(340**2),   # x
            15000.0*9.8/(340**2),   # y
            0.0,                     # z
            15000.0*9.8/(340**2),   # x_t
            15000.0*9.8/(340**2),   # y_t
            0.0,   # z_t
            250.0/340,     # v
            np.deg2rad(12.0),      # alpha
            np.deg2rad(6.0),      # beta
            np.deg2rad(20.0),      # gamma
            np.deg2rad(180.0),    # chi
            np.deg2rad(180.0),     # chi_t
            np.deg2rad(30.0),      # mu
            1.03,                   # thr
            np.pi,                  # rel_chi
            10000.0*9.8/(340**2),    # rel_dist
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

        # 保存当前状态作为上一刻的值
        state_prev = self.state

        # 3DoF模块更新状态
        self.state = self.fdm.run(self.state, action)

        # 计算奖励
        reward = self.get_reward(self.state, action, state_prev)
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
        # 定义初始状态
        initial_state: Dict[str, float] = {
            'x': 0.0,
            'y': 0.0,
            'z': -1500.0,      # 初始高度 1500 米
            # 'x_t': np.random.uniform(-2000, 2000),
            # 'y_t': np.random.uniform(-2000, 2000),
            # 'z_t': np.random.uniform(-2000,-1000),
            'x_t': 10000.0,
            'y_t': 10000.0,
            'z_t': -8000.0,
            'v': 140.0,        # 初始速度
            'alpha': np.deg2rad(3.0),   # 初始攻角 3°
            'beta': np.deg2rad(0.0),    # 初始侧滑角 0°
            'gamma': np.deg2rad(3.0),   # 初始航迹倾角 3°
            'chi': np.deg2rad(0.0),    # 初始航向 0°
            # 'chi_t': np.random.uniform(-np.pi,np.pi),      # 目标航向 0°
            'chi_t':np.deg2rad(0.0),
            'mu': np.deg2rad(0.0),    # 初始绕速度滚转角 0°
            'thr': 0.7  ,     # 初始油门 0.7
            'rel_chi': 0.0,   # 相对航向
            'rel_dist': 0.0  # 相对距离
        }

        self.state = initial_state
        observation = self.get_observation()
        state_prev = observation

        return observation
    
    def get_observation(self):
        state = self.state
        observation = np.array([
            state['x']*9.8/(340**2), state['y']*9.8/(340**2), state['z']*9.8/(340**2),
            state['x_t']*9.8/(340**2), state['y_t']*9.8/(340**2), state['z_t']*9.8/(340**2),
            state['v']/340, state['alpha'], state['beta'],
            state['gamma'], state['chi'], state['chi_t'],
            state['mu'], state['thr'], state['rel_chi'],
            state['rel_dist']*9.8/(340**2)
        ], dtype=np.float32)

        return observation

    def render(self, mode='human'):
        # 启用渲染开关，在每个回合结束后画图
        # plot_rslt(self.alpha_log,
        #             self.beta_log,
        #             self.thr_log,
        #             self.gamma_log,
        #             self.x_log,
        #             self.y_log,
        #             self.z_log,
        #             self.total_reward)
        pass

    
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
            'altitude': 8.0,      # 高度保持（主导项）
            'speed': 3.0,         # 速度保持
            'alpha': 2.5,          # 迎角控制
            'gamma': 3.0,          # 航迹角稳定
            'mu': 3.0,          # 绕速度滚转角抑制
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
        mu_deg = math.degrees(state['mu'])
        
        # === 核心奖励项 ===
        # 1. 高度保持（双高斯复合奖励）
        alt_error = abs(current_alt - self.target_params['altitude'])
        # alt_reward = 0.7 * math.exp(-(alt_error**2)/(2*(self.tolerances['altitude']**2))) \
        #            + 0.3 * math.exp(-alt_error/(2*self.tolerances['altitude']))
        alt_reward = -0.002 * alt_error
        
        # 2. 速度跟踪（渐进式奖励）
        speed_error = abs(state['v'] - self.target_params['speed'])
        speed_reward = 1 / (1 + 0.1*speed_error**2)
        
        # 3. 迎角控制（带死区的钟形曲线）
        alpha_error = abs(alpha_deg - self.target_params['alpha'])
        alpha_reward = math.exp(-(alpha_error**2)/(2*(self.tolerances['alpha']**2)))
        
        # 4. 航迹角稳定（抑制垂直机动）
        # gamma_reward = 1 - 0.2 * abs(gamma_deg - self.target_params['gamma'])
        gamma_penalty = (gamma_deg / 8)**2
        
        # === 稳定性惩罚项 ===
        # 5. 绕速度滚转角抑制（零值保持）
        mu_penalty = (mu_deg / 1)**2
        
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
        
        # 8. 存活奖励
        survive_reward = 1

        # === 综合奖励计算 ===
        reward = (
            self.weights['altitude'] * alt_reward +
            self.weights['speed'] * speed_reward * 0 +
            self.weights['alpha'] * alpha_reward * 0 -
            self.weights['gamma'] * gamma_penalty -
            self.weights['mu'] * mu_penalty -
            self.weights['action_smooth'] * action_penalty +
            self.weights['throttle'] * throttle_reward + 
            survive_reward
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
            abs(math.degrees(state['mu'])) > 2.5 or                  # 绕速度滚转角>2°
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
        self.max_pursuit_time = 5000    # 最大追击步数
        
        # 奖励权重（完全重新定义）
        self.reward_weights = {
            'distance': 2.5,       # 距离缩短奖励
            'heading': 1.5,        # 航向对准奖励
            'altitude': 2.0,       # 高度保持奖励
            'stability': 0.8,      # 姿态稳定奖励
            'action_penalty': 0.2, # 动作幅度惩罚
            'success': 30.0,       # 捕获成功奖励
            'timeout': -10.0,      # 超时惩罚
            'alt_protect':-10
        }

    def get_reward(self, state, action, state_prev):
        # 敌我相对位置
        dx = state['x_t'] - state['x']
        dy = state['y_t'] - state['y']
        dz = state['z_t'] - state['z']
        dist = - np.sqrt(dx**2 + dy**2)

        dx_prev = state_prev['x_t'] - state_prev['x']
        dy_prev = state_prev['y_t'] - state_prev['y']
        dz_prev = state_prev['z_t'] - state_prev['z']
        dist_prev = - np.sqrt(dx_prev**2 + dy_prev**2)

        # 核心奖励项 --------------------------------------------------
        # 1. 距离奖励（指数衰减 + 相对改进）
        # distance_reward = math.exp(-0.0002 * state['rel_dist']) * (1 + 0.5/(state['rel_dist']/1000 + 1))
        distance_reward = 0.1 * (dist-dist_prev)
        # distance_reward = (dist_prev - dist) / dist_prev
        
        # 2. 航向对准奖励（余弦相似度）
        desired_heading = math.atan2(dy, dx)
        heading_error = abs((desired_heading - state['chi'] + np.pi) % (2*np.pi) - np.pi)
        heading_reward = math.cos(heading_error)
        
        # 3. 高度保持奖励（目标高度跟踪）
        # alt_error = abs(state['z_t'] - state['z'])
        # altitude_reward = math.exp(-0.0001 * alt_error)
        alt_error = dz_prev - dz
        altitude_reward = - 0.1 * alt_error
        
        # 4. 姿态稳定惩罚（综合角度限制）
        alpha_deg = math.degrees(state['alpha'])
        mu_deg = math.degrees(state['beta'])
        gamma_deg = math.degrees(state['gamma'])
        stability_penalty = (
            0.3*max(abs(alpha_deg)-3, 0)**2 + 
            0.5*abs(mu_deg)**2 + 
            0.2*abs(gamma_deg)**2
        ) / 100
        
        # 5. 动作平滑惩罚
        if not hasattr(self, 'prev_action'):
            self.prev_action = np.zeros_like(action)
        else:
            self.prev_action = action.copy()
        action_penalty = np.mean(np.square(action)) + 0.3*np.mean(np.abs(action - self.prev_action))
        
        
        # 综合奖励计算 ------------------------------------------------
        reward = (
            self.reward_weights['distance'] * distance_reward +
            self.reward_weights['heading'] * heading_reward * 0 +
            self.reward_weights['altitude'] * altitude_reward  -
            self.reward_weights['stability'] * stability_penalty * 0 -
            self.reward_weights['action_penalty'] * action_penalty * 0
        )
        
        # 事件奖励/惩罚 -----------------------------------------------
        if state['rel_dist'] < self.safe_radius:
            reward += self.reward_weights['success']
            print('Mission Success!')
        if state['z'] > -500:
            reward += self.reward_weights['alt_protect']
        elif self.time_step >= self.max_pursuit_time:
            reward += self.reward_weights['timeout']
            print('Timeout!')
        
        return reward

    def get_done(self, state):

        return state['rel_dist'] < self.safe_radius or state['z'] > 0 or self.time_step >= self.max_pursuit_time

    # def step(self, action):
    #     # 在父类step前添加目标动态更新
    #     self._update_target_motion()
    #     return super().step(action)
    
    # def _update_target_motion(self):
    #     """目标运动模型（示例：匀速直线运动）"""
    #     dt = self.fdm.dt_AP  # 假设FDM提供时间步长
    #     self.state['x_t'] += 0 * math.cos(self.state['chi_t']) * dt
    #     self.state['y_t'] += 0 * math.sin(self.state['chi_t']) * dt