import numpy as np
import gym
from fdmEnv import PursuitEvasionGame

def main():
    # 创建环境实例
    env = PursuitEvasionGame(render_mode=None)
    
    # 重置环境，获取初始观察
    initial_observation = env.reset()
    print("Initial State:")
    print_state(initial_observation)
    
    # 定义预定义的动作
    action1 = np.array([0.1, -0.1, 0.05], dtype=np.float32)  # 第一步的动作
    action2 = np.array([-0.2, 0.2, -0.05], dtype=np.float32)  # 第二步的动作
    
    # 第一步
    observation1, reward1, done1, info1 = env.step(action1)
    print("\nAfter Step 1:")
    print_state(observation1)
    print(f"Reward: {reward1}, Done: {done1}, Info: {info1}")
    
    # 第二步
    observation2, reward2, done2, info2 = env.step(action2)
    print("\nAfter Step 2:")
    print_state(observation2)
    print(f"Reward: {reward2}, Done: {done2}, Info: {info2}")
    
    # 关闭环境
    env.close()

def print_state(observation):
    state_labels = [
        'x', 'y', 'z',
        'x_t', 'y_t', 'z_t',
        'v', 'alpha', 'beta',
        'gamma', 'chi', 'chi_t',
        'mu'
    ]
    
    print("State Variables:")
    for label, value in zip(state_labels, observation):
        print(f"  {label}: {value:.4f}")

if __name__ == "__main__":
    main()
