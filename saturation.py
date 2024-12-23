import numpy as np
def saturation(action):
    limits = {
        'ny': (-3, 3),
        'nz': (-2.5, 2.5),
        # 根据需要添加更多的 action 类型和对应的 min_val, max_val
    }
    return limits.get(action, (-1.0, 1.0))