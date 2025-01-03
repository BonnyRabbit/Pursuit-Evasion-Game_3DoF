from scipy.interpolate import RegularGridInterpolator
import numpy as np
from saturation import limit

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


def scale(action, action_type):
    action = np.array(action, dtype=float)
    # 得到物理范围
    min_val, max_val = limit(action_type)
    scaled_val = ((action + 1) / 2) * (max_val - min_val) + min_val

    return scaled_val
