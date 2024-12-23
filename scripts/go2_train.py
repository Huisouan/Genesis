import argparse
import os
import pickle
import shutil
import sys
from rl_lab.env import *
from rl_lab.config import * 
from rsl_rl.runners import *
from rsl_rl.utils.wrappers import RslRlVecEnvWrapper

import genesis as gs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="Go2ActorCritic")
    parser.add_argument("-B", "--num_envs", type=int, default=4096)
    parser.add_argument("--max_iterations", type=int, default=10000)
    args = parser.parse_args()
    
    try:
        gs.init(logging_level="warning")
    except AttributeError as e:
        print(f"AttributeError: {e}")
        print(f"Available attributes in genesis: {dir(gs)}")

    log_dir = f"logs/{args.exp_name}"
    
    # 使用实验名称来初始化配置类
    cfg_cls: Go2ActorCritic = eval(f"{args.exp_name}({args.exp_name!r}, {args.max_iterations})")
    # 获取环境配置、观测配置、奖励配置和命令配置
    env_cfg, obs_cfg, reward_cfg, command_cfg = cfg_cls.get_env_cfg()
    train_cfg = cfg_cls.get_train_cfg()

    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)
    # 创建 Go2Env 环境
    env_class_name = env_cfg['env_name']
    env_class = getattr(sys.modules[__name__], env_class_name)
    env = env_class(num_envs=args.num_envs, env_cfg=env_cfg, obs_cfg=obs_cfg, reward_cfg=reward_cfg, command_cfg=command_cfg)
    # 使用 RslRlVecEnvWrapper 包装环境
    env = RslRlVecEnvWrapper(env)
    # 创建 OnPolicyRunner
    runner_class_name = train_cfg['runner_class_name']
    runner_class = getattr(sys.modules[__name__], runner_class_name)
    runner = runner_class(env, train_cfg, log_dir, device="cuda:0")

    pickle.dump(
        [env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg],
        open(f"{log_dir}/cfgs.pkl", "wb"),
    )

    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)


if __name__ == "__main__":
    main()