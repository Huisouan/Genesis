class EnvConfig:
    def __init__(self):
        self.env_name = "Go2BaseEnv"
        self.num_actions = 12
        self.default_joint_angles = {  # [rad]
            "FL_hip_joint": 0.0,
            "FR_hip_joint": 0.0,
            "RL_hip_joint": 0.0,
            "RR_hip_joint": 0.0,
            "FL_thigh_joint": 0.8,
            "FR_thigh_joint": 0.8,
            "RL_thigh_joint": 1.0,
            "RR_thigh_joint": 1.0,
            "FL_calf_joint": -1.5,
            "FR_calf_joint": -1.5,
            "RL_calf_joint": -1.5,
            "RR_calf_joint": -1.5,
        }
        self.dof_names = [
            "FR_hip_joint",
            "FR_thigh_joint",
            "FR_calf_joint",
            "FL_hip_joint",
            "FL_thigh_joint",
            "FL_calf_joint",
            "RR_hip_joint",
            "RR_thigh_joint",
            "RR_calf_joint",
            "RL_hip_joint",
            "RL_thigh_joint",
            "RL_calf_joint",
        ]
        self.kp = 20.0
        self.kd = 0.5
        self.termination_if_roll_greater_than = 10  # degree
        self.termination_if_pitch_greater_than = 10
        self.base_init_pos = [0.0, 0.0, 0.42]
        self.base_init_quat = [0.0, 0.0, 0.0, 1.0]
        self.episode_length_s = 20.0
        self.resampling_time_s = 4.0
        self.action_scale = 0.25
        self.simulate_action_latency = True
        self.clip_actions = 100.0
        self.terraincfg = {
            'flat_terrain' :True,
            'mesh_type': 'trimesh',  # "heightfield" # none, plane, heightfield or trimesh
            'horizontal_scale': 0.1,  # [m]
            'vertical_scale': 0.005,  # [m]
            'border_size': 25,  # [m]
            'curriculum': True,
            'static_friction': 1.0,
            'dynamic_friction': 1.0,
            'restitution': 0.,
            # rough terrain only:
            'measure_heights': True,
            'measured_points_x': [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],  # 1mx1.6m rectangle (without center line)
            'measured_points_y': [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5],
            'selected': False,  # select a unique terrain type and pass all arguments
            'terrain_kwargs': None,  # Dict of arguments for selected terrain
            'max_init_terrain_level': 5,  # starting curriculum state
            'subterrain_size': 8.,  # size of subterrain in meters
            'num_rows': 10,  # number of terrain rows (levels)
            'num_cols': 20,  # number of terrain cols (types)
            # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete]
            'terrain_proportions': [0.1, 0.2, 0.3, 0.3, 0.1],
            # trimesh only:
            'slope_treshold': 0.75  # slopes above this threshold will be corrected to vertical surfaces
        }
class ObsConfig:
    def __init__(self):
        self.num_obs = 45
        self.use_privileged_obs = False
        self.obs_scales = {
            "lin_vel": 2.0,
            "ang_vel": 0.25,
            "dof_pos": 1.0,
            "dof_vel": 0.05,
        }

class RewardConfig:
    def __init__(self):
        self.tracking_sigma = 0.25
        self.base_height_target = 0.3
        self.feet_height_target = 0.075
        
        self.reward_scales = {
            "tracking_lin_vel": 1.0,
            "tracking_ang_vel": 0.2,
            "lin_vel_z": -1.0,
            "base_height": -50.0,
            "action_rate": -0.005,
            "similar_to_default": -0.1,
        }

class CommandConfig:
    def __init__(self):
        self.num_commands = 3
        self.lin_vel_x_range = [-1, 1]
        self.lin_vel_y_range = [-1, 1]
        self.ang_vel_range = [-3.14, 3.14]

class BaseEnvCfg:
    def __init__(self):
        self.env_cfg = EnvConfig()
        self.obs_cfg = ObsConfig()
        self.reward_cfg = RewardConfig()
        self.command_cfg = CommandConfig()

    def get_cfg(self):
        return (
            self.env_cfg.__dict__.copy()
,
            self.obs_cfg.__dict__.copy()
,
            self.reward_cfg.__dict__.copy()
,
            self.command_cfg.__dict__.copy()
,
        )