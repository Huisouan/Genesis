from . import Go2BaseEnv
from rl_lab.assets.loder_for_algs import AmpMotion

class Go2AseEnv(Go2BaseEnv):
    def __init__(self,num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg):
        super().__init__(num_envs=num_envs, env_cfg=env_cfg, obs_cfg=obs_cfg, reward_cfg=reward_cfg, command_cfg=command_cfg)
        self.amp_loader = AmpMotion(
            data_dir ="datasets/mocap_motions_go2",
            datatype="isaacgym",
            file_type="txt",
            data_spaces = None,
            env_step_duration = 0.005,
        )


