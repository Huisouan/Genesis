import genesis as gs
import torch
from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat, transform_quat_by_quat

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.go2.low_level.go2_pd_sim2sim import Go2_SIM2SIM
from unitree_bridge.a2c import A2C
from unitree_bridge.config.algo.algocfg import A2CConfig
from unitree_bridge.config.robot.bot_cfg import GO2
from unitree_bridge.process.joystick import *
# 默认网络接口名称
default_network = "lo"

base_pos = [0.0, 0.0, 0.42]
base_quat = [0.0, 0.0, 0.0, 1.0]

dof_names = [
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

def data_process(imu_state,motor_state,device):

    quaternion = torch.zeros(1,4)
    for i in range(4):
        quaternion[0][i] = imu_state.quaternion[i]
    gyroscope = torch.zeros(1,3)
    for i in range(3):
        gyroscope[0][i] = imu_state.gyroscope[i]
    joint_pos = torch.zeros(1,12)
    joint_vel = torch.zeros(1,12)
    for i in range(12):
        joint_pos[0][i] = motor_state[i].q
        joint_vel[0][i] = motor_state[i].dq
    
    joint_pos = joint_pos.to(device)
    joint_vel = joint_vel.to(device)
    gyroscope = gyroscope.to(device)
    quaternion = quaternion.to(device)

    return gyroscope,quaternion,joint_pos,joint_vel

def main(go2:Go2_SIM2SIM):
    gs.init()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dt = 0.02
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=dt, substeps=2),
        viewer_options=gs.options.ViewerOptions(
            max_FPS=int(0.5 / dt),
            camera_pos=(2.0, 0.0, 2.5),
            camera_lookat=(0.0, 0.0, 0.5),
            camera_fov=40,
        ),
        vis_options=gs.options.VisOptions(n_rendered_envs=1),
        rigid_options=gs.options.RigidOptions(
            dt=dt,
            constraint_solver=gs.constraint_solver.Newton,
            enable_collision=True,
            enable_joint_limit=True,
            
        ),
        show_viewer=True,
    )

    terrain = scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))        

    # add robot
    base_init_pos = torch.tensor(base_pos, device=device)
    base_init_quat = torch.tensor(base_quat, device=device)
    inv_base_init_quat = inv_quat(base_init_quat)
    robot = scene.add_entity(
        gs.morphs.URDF(
            file="urdf/go2/urdf/go2.urdf",
            pos=base_init_pos.cpu().numpy(),
            quat=base_init_quat.cpu().numpy(),
        ),
    )

    # build
    scene.build(n_envs=1)

    # names to indices
    motor_dofs = [robot.get_joint(name).dof_idx_local for name in dof_names]
    # 假设 device 已经定义，例如 device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    idx = torch.tensor([0], device=device)  # 修改这里，将 idx 转换为 1D 张量
    # PD control parameters
    robot.set_dofs_kp([20] * 12, motor_dofs)
    robot.set_dofs_kv([0.5] * 12, motor_dofs)
    while True:
        imu_state, motor_state = go2.return_obs()        
        
        
        gyroscope,quaternion,joint_pos,joint_vel= data_process(imu_state,motor_state,device)     
        
        robot.set_quat(quaternion, zero_velocity=True, envs_idx=idx)
        robot.set_dofs_position(
            position=joint_pos,
            dofs_idx_local=motor_dofs,
            zero_velocity=True,
            envs_idx=idx,
        )
        scene.visualizer.update()
        
if __name__ == "__main__":
    ChannelFactoryInitialize(1, default_network)

    # 创建Go2_SIM2SIM对象
    go2 = Go2_SIM2SIM()
    # 启动Go2_SIM2SIM对象
    go2.Start()
    
    main(go2)