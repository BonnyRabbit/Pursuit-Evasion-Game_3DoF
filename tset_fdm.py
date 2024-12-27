import numpy as np
from fdm import fdm_3DoF
import math

def test_take_action():
    model = fdm_3DoF()

    model.v = 100
    model.alt = 4000
    model.alpha = math.radians(5)
    model.beta = 0
    model.gamma = 1e-6
    model.chi = math.radians(90)
    model.mu = 1e-6
    model.thr = 0.7
    model.x = 0
    model.y = 0
    model.z = -4000

    print("初始状态:")
    print(f"位置: ({model.x}, {model.y}, {-model.z})")
    print(f"速度: {model.v} m/s")
    print(f"攻角: {np.rad2deg(model.alpha)}°")
    print(f"侧滑角: {np.rad2deg(model.beta)}°")
    print(f"偏航角: {np.rad2deg(model.chi)}°")
    print(f"航迹倾角: {np.rad2deg(model.gamma)}°")
    print(f"绕速度滚转角: {np.rad2deg(model.mu)}°")
    print(f"油门: {model.thr}")

    action = [0.5, 0, 0.1]

    model.take_action(action)

    print("第二拍状态:")
    print(f"位置: ({model.x}, {model.y}, {-model.z})")
    print(f"速度: {model.v} m/s")
    print(f"攻角: {np.rad2deg(model.alpha)}°")
    print(f"侧滑角: {np.rad2deg(model.beta)}°")
    print(f"偏航角: {np.rad2deg(model.chi)}°")
    print(f"航迹倾角: {np.rad2deg(model.gamma)}°")
    print(f"绕速度滚转角: {np.rad2deg(model.mu)}°")
    print(f"油门: {model.thr}")

if __name__ == "__main__":
    test_take_action()