import argparse
import os
import pickle
import sys
import torch
from rl_lab.env import *
from rsl_rl.runners import *
from rsl_rl.utils.wrappers import RslRlVecEnvWrapper
import genesis as gs
from rsl_rl.utils.wrappers import (
    RslRlOnPolicyRunnerCfg,
    export_policy_as_jit,
    export_policy_as_onnx,
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="Go2Amp")
    parser.add_argument("--ckpt", type=int, default=100)
    args = parser.parse_args()

    gs.init()

    log_dir = f"logs/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(open(f"logs/{args.exp_name}/cfgs.pkl", "rb"))
    reward_cfg["reward_scales"] = {}

    env_class_name = env_cfg['env_name']
    env_class = getattr(sys.modules[__name__], env_class_name)
    env = env_class(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=True,
    )
    env = RslRlVecEnvWrapper(env)
    runner_class_name = train_cfg['runner_class_name']
    runner_class = getattr(sys.modules[__name__], runner_class_name)
    runner = runner_class(env, train_cfg, log_dir, device="cuda:0")

    resume_path = os.path.join(log_dir, f"model_{args.ckpt}.pt")


    runner.load(resume_path)
    policy = runner.get_inference_policy(device="cuda:0")
    export_model_dir = f"weights/{args.exp_name}"
    export_policy_as_jit(
        runner.alg.actor_critic, runner.obs_normalizer, path=export_model_dir, filename="policy.pt"
    )
    obs, _ = env.reset()
    with torch.no_grad():
        while True:
            actions = policy(obs)
            obs, rews, dones, infos = env.step(actions)


if __name__ == "__main__":
    main()

"""
# evaluation
python examples/locomotion/go2_eval.py -e go2-walking -v --ckpt 100
"""
