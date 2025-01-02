from . import *

class Go2Himloco():
    def __init__(self,exp_name, max_iterations):
        self.env_cfg = BaseEnvCfg()
        self.train_cfg = BaseTrainCfg(exp_name, max_iterations)
        self.train_cfg.runner_class_name = "HIMOnPolicyRunner"
        self.train_cfg.policy.class_name = "HIMActorCritic"
        self.train_cfg.algorithm.class_name = "HIMPPO"

        self.env_cfg.obs_cfg.num_one_step_observations = 45
        self.env_cfg.obs_cfg.encoder_steps = 6
        self.env_cfg.obs_cfg.num_obs = self.env_cfg.obs_cfg.num_one_step_observations * self.env_cfg.obs_cfg.encoder_steps
        self.num_one_step_privileged_obs = 45

        self.env_cfg.reward_cfg.reward_scales = {
            #root
            "tracking_lin_vel": 1.5,
            "tracking_ang_vel": 0.75,
            "lin_vel_z": -2.0,
            "orientation": -0.2,
            "base_height": -0.4,
            #joint
            "action_rate": -0.01,
            "similar_to_default": -0.05,
            "joint_power": -2e-5,
            
        }
    
    
    def get_env_cfg(self):
        return self.env_cfg.get_cfg()
    def get_train_cfg(self):
        return self.train_cfg.to_dict()
    
