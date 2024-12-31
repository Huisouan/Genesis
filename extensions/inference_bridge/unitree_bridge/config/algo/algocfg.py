class Algo_def_cfg:
    model_path = None
    
    policy_observation_dim = None

    policy_action_dim = None

    joint_order = None
    
    default_jointpos_bias = None

class A2CConfig(Algo_def_cfg):
    model_path = "weights/Go2ActorCritic/policy.pt"
    
    policy_observation_dim = 45

    policy_action_dim = 12

    joint_order = (
    "FR_hip",   # hip
    "FR_thigh", # thigh
    "FR_calf",  # calf
    "FL_hip",   # hip
    "FL_thigh", # thigh
    "FL_calf",  # calf
    "RR_hip",   # hip
    "RR_thigh", # thigh
    "RR_calf",  # calf
    "RL_hip",   # hip
    "RL_thigh", # thigh
    "RL_calf"   # calf
)
    
    default_jointpos_bias = [
        0.0,0.8,-1.5,
        0.0,0.8,-1.5,
        0.0,1.0,-1.5,
        0.0,1.0,-1.5,
    ]    
    
    base_ang_vel_scale = 0.1
    joint_pos_scale = 1.0
    joint_vel_scale = 0.05
    actions_scale = 0.25