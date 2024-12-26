# Action限制在-1~1间，在这里进行还原
def limit(action_type):
    limits = {
        'ay': (-15, 15),
        'az': (-25, 25),
        'thr': (0, 1.03),
        'alpha': (-3, 12),
        'beta': (-6, 6)
        # 根据需要添加更多的 action 类型和对应的 min_val, max_val
    }
    return limits.get(action_type, (-1.0, 1.0))