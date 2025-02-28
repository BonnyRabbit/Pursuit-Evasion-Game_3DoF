import os
import gym
import time
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv
from fdmEnv import PursuitEvasionGame, StableFlyingEnv, TargetTrackingEnv
from callback import TensorboardTimeSeriesCallback

device = 'cuda' if torch.cuda.is_available() else 'cpu'
stage = 1 # 训练阶段

def make_env(stage, render_mode='human', rank=0, seed=0):
    """
    根据不同阶段选择环境。
    """
    def _init():
        if stage == 1:
            env = StableFlyingEnv(render_mode=render_mode)
        elif stage == 2:
            env = TargetTrackingEnv(render_mode=render_mode)
        else:
            env = PursuitEvasionGame(render_mode=render_mode)
        print(f"Environment created: {env}")
        return env
    return _init

def main():
    # 记录开始时间
    start_time = time.time()
    # 创建日志目录
    log_dir = f"logs/stage{stage}/" # 按阶段区分日志目录
    
    os.makedirs(log_dir, exist_ok=True)

    # 预训练模型路径
    pretrained_path = os.path.join(log_dir, 'best_model/best_model.zip')

    # 设置并行环境
    n_envs = 4

    # 创建评估环境
    eval_env = SubprocVecEnv([make_env(stage=stage, render_mode=None,
                                        rank=i) for i in range(n_envs)])

    # # 创建训练环境
    train_env = SubprocVecEnv([make_env(stage=stage, render_mode=None)])

    if os.path.exists(pretrained_path):
        print(f"Loading pretrained model from {pretrained_path}")
        model = PPO.load(
            pretrained_path,
            env=train_env,
            device=device,
            tensorboard_log=log_dir,
            learning_rate=2e-5,
            n_steps=2048 // n_envs,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.25,
            ent_coef=0.0,
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs=dict(
                net_arch=dict(
                    pi=[256, 512, 128],
                    vf=[256, 512, 128]
                )
            )
        )
        print("Successfully loaded pretrained model!")
    else:
        print("No pretrained model found, initializing new model.")

        # 初始化 PPO 模型
        model = PPO(
            "MlpPolicy",              # 策略网络类型
            train_env,                # 训练环境
            device=device,            # 训练设备
            verbose=1,                # 显示详细信息的级别
            tensorboard_log=log_dir,  # TensorBoard 日志目录
            learning_rate=2e-5,       # 学习率
            n_steps=2048 // n_envs,   # 每次更新前在每个环境中运行的步数
            batch_size=64 * n_envs,   # 小批量大小
            n_epochs=10,              # 优化代理损失时的迭代次数
            gamma=0.99,               # 折扣因子
            gae_lambda=0.95,          # GAE lambda 参数
            clip_range=0.25,          # PPO 剪切参数
            ent_coef=0.0,             # 熵系数
            vf_coef=0.5,              # 值函数系数
            max_grad_norm=0.5,        # 梯度最大范数
            policy_kwargs=dict(
                net_arch=dict(
                    pi=[256, 512, 128],
                    vf=[256, 512, 128]
                    )
                )  # pi、vf 网络结构
        )

        # 回调函数，用于保存模型、评估模型和查看自定义指标
    callbacks = [
        CheckpointCallback(
            save_freq=20000 // n_envs,           # 每 20,000 步保存一次模型
            save_path=log_dir,                   # 模型保存目录
            name_prefix='PEG_stage{stage}'    # 模型名称前缀
        ),
        EvalCallback(
            eval_env,
            best_model_save_path=os.path.join(log_dir, f"best_model"),
            log_path=log_dir,
            eval_freq=10000 // n_envs,
            deterministic=True,
            render=False
        ),
        TensorboardTimeSeriesCallback(log_dir=log_dir)
    ]

    # 训练模型(增量训练)
    additional_timesteps = 50_000_000
    total_timesteps = model.num_timesteps + additional_timesteps
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            tb_log_name="PEG_stage{stage}",
            reset_num_timesteps=False
        )
    except KeyboardInterrupt:
        print("Training interrupted by YKJ.")
    finally:
        model.save(os.path.join(log_dir, f"PEG_stage{stage}_final"))
        print("Model saved before exit.")

    # 记录结束时间
    end_time = time.time()
    elapsed_time = end_time - start_time
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = int(elapsed_time % 60)

    # 打印训练时长
    print(f"训练完成，耗时: {hours}h {minutes}min {seconds}s")

    # 关闭环境
    train_env.close()
    eval_env.close()

if __name__ == "__main__":
    main()
