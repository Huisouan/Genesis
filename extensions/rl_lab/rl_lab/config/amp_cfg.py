from . import *
import glob
class Go2Amp():
    def __init__(self,exp_name, max_iterations):
        self.env_cfg = BaseEnvCfg()
        self.train_cfg = BaseTrainCfg(exp_name, max_iterations)
        self.train_cfg.runner_class_name = "AmpOnPolicyRunnerg"
        self.train_cfg.policy.class_name = "ActorCritic"
        self.train_cfg.algorithm.class_name = "AMPPPO"

        self.train_cfg.amp_policy_cfg = {
            #PPO中的AMP算法的配置
            "amp_reward_coef":2.0,
            "amp_discr_hidden_dims":[1024,512],
            "amp_task_reward_lerp":0.3,
            "min_normalized_std":[0.01,0.01,0.01]*4,
            
        }
        
        self.env_cfg.env_cfg.env_name = "Go2PAmpEnv"
        self.env_cfg.env_cfg.dof_names = [
            #更改顺序与数据集相同
            "FL_hip_joint",
            "FR_hip_joint",
            "RL_hip_joint",
            "RR_hip_joint",
            "FL_thigh_joint",
            "FR_thigh_joint",
            "RL_thigh_joint",
            "RR_thigh_joint",
            "FL_calf_joint",
            "FR_calf_joint",
            "RL_calf_joint",
            "RR_calf_joint",
        ]
        self.env_cfg.env_cfg.amp = {
            #环境中关于AMP算法的配置
            "urdf_path":  "datasets/go2_description/urdf/go2_description.urdf",
            "amp_replay_buffer_size": 1000000,
            "ee_names" : ["FL_foot", "FR_foot", "RL_foot", "RR_foot"],
            "amp_motion_files":glob.glob("datasets/mocap_motions_go2/*"),
            "preload_transitions":True,
        }
        self.env_cfg.obs_cfg.num_amp_observations = 42
        
        self.env_cfg.command_cfg.lin_vel_x_range = [0, 3]
        self.env_cfg.command_cfg.lin_vel_y_range = [-0.3, 0.3]
        
        self.env_cfg.reward_cfg.reward_scales = {
            #root
            "tracking_lin_vel": 1.5*30,
            "tracking_ang_vel": 0.75*30,
            #"lin_vel_z": -2.0,
            #"orientation": -0.2,
            #"base_height": -0.4,
            #joint
            #"action_rate": -0.01,
            #"similar_to_default": -0.05,
            #"joint_power": -2e-5,

        }
    
    
    def get_env_cfg(self):
        return self.env_cfg.get_cfg()
    def get_train_cfg(self):
        return self.train_cfg.to_dict()
    
