from scipy.interpolate import RegularGridInterpolator
import numpy as np
from saturation import limit
import matplotlib.pyplot as plt

# 插值表
def interp(mat, var, grid_axes, method='linear'):
    """
    PARAMS:
    mat: .mat
    var: CD/CL/CY
    grid_axes: aoa/aos/ma
    method: linear/cubic

    Return:
    interp function
    """
    axes = [mat[axis].squeeze() for axis in grid_axes]
    data = mat[var]

    interpolator = RegularGridInterpolator(
        tuple(axes),
        data,
        method = method,
        bounds_error=False,
        fill_value=None
    )
    def interp_function(*args):
        point = np.array(args)
        return interpolator(point)
    
    return interp_function

# 限幅截断
def clamp(value, min_val, max_val):
    return np.minimum(np.maximum(value, min_val), max_val)

# 方向余弦矩阵
def DCM(axis, angle_deg):
    angle_rad = np.radians(angle_deg)
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)

    if axis.lower() == 'x':
        L = np.array([
            [1, 0, 0],
            [0, c, s],
            [0, -s, c]
        ])
    elif axis.lower() == 'y':
        L = np.array([
            [c, 0, -s],
            [0, 1, 0],
            [s, 0, c]
        ])
    elif axis.lower() == 'z':
        L = np.array([
            [c, s, 0],
            [-s, c, 0],
            [0, 0, 1]
        ])

    return    L   


def scale(action):
    action = np.array(action, dtype=float)
    """
    对输入的动作进行缩放。
    :param action: 输入动作（包含 alpha, beta, thr）
    :return: 缩放后的动作
    """
    delta_alpha = action[0] * np.deg2rad(15)
    delta_beta = action[1] * np.deg2rad(6)
    # delta_thr = clamp(action[2],0, 1.03)
    delta_thr = action[2]


    return delta_alpha, delta_beta, delta_thr

# 绘图
def plot_rslt(alpha_log,
              beta_log,
              thr_log,
              gamma_log,
              x_log,
              y_log,
              z_log,
              total_reward):
    """
    绘制一个episode的状态曲线
    """
    fig, axes = plt.subplots(3, 1, figsize=(10, 8))
    # 姿态
    axes[0].plot(alpha_log, label='alpha')
    axes[0].plot(beta_log, label='beta')
    axes[0].plot(thr_log, label='thr')
    axes[0].plot(gamma_log, label='gamma')
    axes[0].set_title('姿态 动作曲线')
    axes[0].set_xlabel('step')
    axes[0].legend()
    
    # 位置
    axes[1].plot(x_log, label='x')
    axes[1].plot(y_log, label='y') 
    axes[1].plot(z_log, label='z')
    axes[1].set_title('位置曲线')
    axes[1].legend()

    # 奖励
    axes[2].plot(total_reward, label='reward')
    axes[2].set_title('奖励曲线')
    axes[2].set_xlabel('step')
    axes[2].set_ylabel('reward')
    axes[2].legend()

    plt.tight_layout()
    plt.show()