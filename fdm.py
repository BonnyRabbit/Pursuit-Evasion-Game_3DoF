import numpy as np
import math
from load_mdl import ModelLoader
from utils import scale, DCM
from aero import AeroForceCalculator
from prop import PropulsionCalculator

class Fdm3DoF:
    def __init__(self):
        # 初始化模型
        self.model_loader = ModelLoader()
        model_data = self.model_loader.load_mdl()
        # 预处理数据
        self.data_aero = model_data['data_aero']
        self.data_prop = model_data['data_prop']
        self.var_aero = model_data['var_aero']
        self.grid_axes_aero = model_data['grid_axes_aero']
        self.var_prop = model_data['var_prop']
        self.grid_axes_prop = model_data['grid_axes_prop']
        # 固定参数
        self.dt_AP = 0.1
        self.mass = 200
        self.g = 9.8
        self.RefArea = self.data_aero['RefArea']
        self.RefLen = self.data_aero['RefLen']
        self.RefSpan = self.data_aero['RefSpan']

        # 初始化气动、发动机力计算方法
        self.aero_force = AeroForceCalculator(
            data_aero=self.data_aero,
            var_aero=self.var_aero,
            grid_axes_aero=self.grid_axes_aero,
        )
        self.prop_force = PropulsionCalculator(
            data_prop=self.data_prop,
            var_prop=self.var_prop,
            grid_axes_prop=self.grid_axes_prop
        )
        # 逃逸者运动规律
        self.evader_velocity = np.array([80, 0, 0], dtype=np.float32)
        self.chi_t_dot = 0

    def calculate_mach(self, v, alt):
        temp = 273.15
        gamma = 1.4
        R = 287.05
        T = temp - 0.0065 * alt
        c = math.sqrt(gamma * R * T)
        mach = v / c
        return mach

    def calculate_aero_force(self, alpha, beta, mach, v, alt):
        D, L, Y = self.aero_force.cal_aero_force(
            alpha=alpha,
            beta=beta,
            mach=mach,
            v=v,
            alt=alt,
            RefArea=self.RefArea
        )
        D = D.item() if isinstance(D, np.ndarray) else D
        L = L.item() if isinstance(L, np.ndarray) else L
        Y = Y.item() if isinstance(Y, np.ndarray) else Y
        return D, L, Y

    def calculate_prop_force(self, mach, alt, thr):
        Thrust = self.prop_force.cal_prop(
            mach=mach,
            alt=alt,
            thr=thr
        )
        Thrust = Thrust.item() if isinstance(Thrust, np.ndarray) else Thrust
        return Thrust

    def calculate_dcm(self, gamma, chi, alpha, beta, mu):
        # 计算方向余弦矩阵
        Ly_gamma = DCM('y', gamma)
        Lz_chi = DCM('z', chi)
        L_kg = Ly_gamma @ Lz_chi

        Lx_mu = DCM('x', mu)
        L_ka = np.linalg.inv(Lx_mu)

        Ly_alpha = DCM('y', -alpha)
        Lz_beta = DCM('z', beta)
        L_ab = Lz_beta @ Ly_alpha

        L_gk = np.linalg.inv(L_kg)
        L_kb = L_ka @ L_ab

        return L_ka, L_gk, L_kb, L_kg
    
    def check_chi(self, chi):
        if chi > math.pi:
            chi -= 2 * math.pi
        elif chi < -math.pi:
            chi += 2 * math.pi
        return chi

    def run(self, state, action):
        """
        更新状态的方法
        :param state: 当前状态的字典
        :param action: 动作的字典
        :return: 更新后的状态字典
        """
        # 提取当前状态
        x, y, z = state['x'], state['y'], state['z']
        x_t, y_t, z_t = state['x_t'], state['y_t'], state['z_t']
        v, alpha, beta, gamma, chi, chi_t, mu = state['v'], state['alpha'], state['beta'], state['gamma'], state['chi'], state['chi_t'], state['mu']
        thr = state['thr']
        mass = self.mass
        alt = -z  # 假设 z 是向下为正

        # 提取动作
        delta_alpha, delta_beta, delta_thr = scale(action)  # 缩放动作

        # 更新控制输入
        alpha += math.radians(delta_alpha) * self.dt_AP
        beta += math.radians(delta_beta) * self.dt_AP
        thr += delta_thr * self.dt_AP

        alpha = np.clip(alpha, math.radians(-3), math.radians(12))
        beta = np.clip(beta, math.radians(-6), math.radians(6))
        thr = np.clip(thr, 0, 1.03)


        # 计算马赫数
        mach = self.calculate_mach(v, alt)

        # 计算气动力
        D, L, Y = self.calculate_aero_force(alpha, beta, mach, v, alt)
        # 计算推进力
        T = 1 * self.calculate_prop_force(mach, alt, thr)

        # 计算方向余弦矩阵
        L_ka, L_gk, L_kb, L_kg = self.calculate_dcm(gamma, chi, alpha, beta, mu)

        # # 速度向量在惯性系中的表示
        # v_g = L_gk @ np.array([[v], [0], [0]])
        # dx, dy, dz = v_g.flatten()

        # # 计算航迹轴下F
        # F = (L_kb @ np.array([[T], [0], [0]])
        #      + L_ka @ np.array([[-D], [Y], [-L]])
        #      + L_kg @ np.array([[0], [0], [mass * self.g]]))
        # Fx, Fy, Fz = F.flatten()

        # # 计算加速度和角速度
        # v_dot = Fx / mass
        # gamma_dot = -Fz / (mass * v)
        # wz = Fy / (mass * v)
        # chi_dot = wz / math.cos(gamma) if math.cos(gamma) != 0 else 0

        dx = v * math.cos(gamma) * math.cos(chi)
        dy = v * math.cos(gamma) * math.sin(chi)
        dz = -v * math.sin(gamma)
        v_dot = (T * math.cos(alpha) - D) / mass - self.g * math.sin(gamma)
        chi_dot = (T * (math.sin(alpha) * math.sin(mu) - math.cos(alpha) * math.sin(beta) * math.cos(mu)) + Y * math.cos(mu) + L * math.sin(mu))\
              / (mass * v * math.cos(gamma)) 
        gamma_dot = (T * (math.sin(alpha) * math.cos(mu) + math.cos(alpha) * math.sin(beta) * math.sin(mu)) - Y * math.sin(mu) + L * math.cos(mu)\
                      - mass * self.g * math.cos(gamma)) / (mass * v)

        # 更新追击者状态
        x += dx * self.dt_AP
        y += dy * self.dt_AP
        z += dz * self.dt_AP
        v += v_dot * self.dt_AP
        gamma += gamma_dot * self.dt_AP
        chi += chi_dot * self.dt_AP
        chi = self.check_chi(chi)

        # 更新逃避者状态（x_t, y_t, z_t）
        x_t += self.evader_velocity[0] * self.dt_AP
        y_t += self.evader_velocity[1] * self.dt_AP
        z_t += self.evader_velocity[2] * self.dt_AP
        chi_t += self.chi_t_dot * self.dt_AP
        chi_t = self.check_chi(chi_t)

        # 更新相对航向和距离
        dx = x_t - x
        dy = y_t - y
        dz = z_t - z

        rel_chi = math.atan2(dy, dx) - chi
        xy_dist = math.hypot(dx, dy)
        rel_dist = math.hypot(xy_dist, dz)

        # 返回更新后的状态
        state = {
            'x': x,
            'y': y,
            'z': z,
            'x_t': x_t,
            'y_t': y_t,
            'z_t': z_t,
            'v': v,
            'alpha': alpha,
            'beta': beta,
            'gamma': gamma,
            'chi': chi,
            'chi_t': chi_t,
            'mu': mu,
            'thr': thr,
            'rel_chi': rel_chi,
            'rel_dist': rel_dist
        }

        return state