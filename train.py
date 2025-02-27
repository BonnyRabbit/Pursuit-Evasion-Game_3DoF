import os
import gym
import time
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from fdmEnv import PursuitEvasionGame, StableFlyingEnv, TargetFlyingEnv
from callback import TensorboardTimeSeriesCallback

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def make_env(stage, render_mode='human'):
    """
    根据不同阶段选择环境。
    """
    def _init():
        if stage == 1:
            env = StableFlyingEnv(render_mode=render_mode)
        elif stage == 2:
            env = TargetFlyingEnv(render_mode=render_mode)
        else:
            env = PursuitEvasionGame(render_mode=render_mode)
        print(f"Environment created: {env}")
        return env
    return _init

def main():
    # 记录开始时间
    start_time = time.time()
    # 创建日志目录
    log_dir = "logs/"
    stage = 1 # 训练阶段
    os.makedirs(log_dir, exist_ok=True)

    # 创建评估环境
    eval_env = DummyVecEnv([make_env(stage=stage, render_mode=None)])

    # # 创建训练环境
    train_env = DummyVecEnv([make_env(stage=stage, render_mode=None)])

    # 可选：检查环境是否符合 Gym API
    # 这对于调试非常有用
    # check_env(PursuitEvasionGame())

    # 初始化 PPO 模型
    model = PPO(
        "MlpPolicy",              # 策略网络类型
        train_env,                # 训练环境
        device=device,            # 训练设备
        verbose=1,                # 显示详细信息的级别
        tensorboard_log=log_dir,  # TensorBoard 日志目录
        learning_rate=2e-4,       # 学习率
        n_steps=2048,             # 每次更新前在每个环境中运行的步数
        batch_size=64,            # 小批量大小
        n_epochs=10,              # 优化代理损失时的迭代次数
        gamma=0.99,               # 折扣因子
        gae_lambda=0.95,          # GAE lambda 参数
        clip_range=0.25,           # PPO 剪切参数
        ent_coef=0.0,             # 熵系数
        vf_coef=0.5,              # 值函数系数
        max_grad_norm=0.5,         # 梯度最大范数
        policy_kwargs=dict(
            net_arch=dict(
                pi=[256, 128, 64],
                vf=[256, 128, 64]
                )
            )  # pi、vf 网络结构
    )

    # 回调函数，用于保存模型、评估模型和查看自定义指标
    callbacks = [
        CheckpointCallback(
            save_freq=10000,                 # 每 10,000 步保存一次模型
            save_path=log_dir,               # 模型保存目录
            name_prefix='ppo_pursuit_evasion'  # 模型名称前缀
        ),
        EvalCallback(
            eval_env,
            best_model_save_path=os.path.join(log_dir, 'best_model/'),
            log_path=log_dir,
            eval_freq=5000,
            deterministic=True,
            render=False
        ),
        TensorboardTimeSeriesCallback(log_dir=log_dir)
    ]

    # 训练模型
    total_timesteps = 50_000  # 根据需要调整总步数
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        tb_log_name="ppo_pursuit_evasion"
    )
    
    # 记录结束时间
    end_time = time.time()
    elapsed_time = end_time - start_time
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = int(elapsed_time % 60)

    # 打印训练时长
    print(f"训练完成，耗时: {hours}h {minutes}min {seconds}s")

    # 保存最终模型
    model.save(os.path.join(log_dir, "ppo_pursuit_evasion_final"))
    print("训练完成并保存模型。")

if __name__ == "__main__":
    main()
