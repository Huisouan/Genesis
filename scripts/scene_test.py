#!/usr/bin/env python3
import argparse
import numpy as np
import genesis as gs
import torch
import plotly.express as px  # 导入plotly.express
import plotly.graph_objects as go  # 导入plotly.graph_objects

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    
    args = parser.parse_args()

    default_joint_angles = {  # [rad]
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

    ########################## init ##########################
    gs.init(backend=gs.gpu)

    ########################## create a scene ##########################
    viewer_options = gs.options.ViewerOptions(
        camera_pos=(0, -3.5, 2.5),
        camera_lookat=(0.0, 0.0, 0.5),
        camera_fov=40,
        max_FPS=60,
    )

    scene = gs.Scene(
        viewer_options=viewer_options,
        sim_options=gs.options.SimOptions(
            dt=0.01,
        ),
        show_viewer=True,
    )

    ########################## entities ##########################
    plane = scene.add_entity(
        gs.morphs.Plane(),
    )

    # 提取关节角度值并转换为 NumPy 数组
    joint_angles_values = [default_joint_angles[name] for name in dof_names]
    pos = np.array([0,0,0.4])
    quat = np.array([0.0, 0.0, 0.0, 1.0])  # 默认四元数，假设机器人初始姿态为单位四元数

    robot = scene.add_entity(
        gs.morphs.URDF(
            file="urdf/go2/urdf/go2.urdf",
            pos=pos,
            quat=quat,
        ),
    )

    ########################## build ##########################
    scene.build()
    motor_dofs = [robot.get_joint(name).dof_idx_local for name in dof_names]
    joints = []
    data = torch.zeros(1000, 13, 3)  # 修改为正确的初始化方式
    for i in range(1000):

        AABB = robot.get_AABB()
        ang = robot.get_ang()
        contacts = robot.get_contacts()
        dofs_armature = robot.get_dofs_armature()
        dofs_control_force = robot.get_dofs_control_force()
        dofs_damping = robot.get_dofs_damping()
        dofs_force = robot.get_dofs_force()
        data[i] = robot.get_links_net_contact_force()
        scene.step()
    
    # 展平数据
    data_flat = data.view(1000, -1).numpy()  # 展平为 (1000, 39)
    
    # 可视化 data 中的数据，使用plotly绘制所有列
    time_steps = np.arange(1000)
    fig = go.Figure()

    for i in range(data_flat.shape[1]):
        fig.add_trace(go.Scatter(x=time_steps, y=data_flat[:, i], mode='lines', name=f'DOF {i}'))

    fig.update_layout(
        title='DOF Forces Over Time',
        xaxis_title='Time Step',
        yaxis_title='Force',
        legend_title='DOF',
        width=1200,
        height=800
    )

    fig.show()

    print("Done")

if __name__ == "__main__":
    main()