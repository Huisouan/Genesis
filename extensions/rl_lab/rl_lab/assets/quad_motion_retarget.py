import argparse
import os
import pandas as pd
import numpy as np
import csv
import genesis as gs
import tkinter as tk
from tkinter import ttk
import threading
import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as R
from rl_lab.assets.transformations import quaternion_from_matrix ,quaternion_multiply
from scipy.signal import butter, lfilter

INIT_ROT = np.array([0, 0, 0, 1.0])
class MotionRetarget:
    def __init__(self, mocap_file, urdf_file, name_list, point_names, joint_names, key_points, scale=[1,1,1]):
        self.mocap_file = mocap_file
        self.urdf_file = urdf_file
        self.name_list = name_list
        self.point_names = point_names
        self.joint_names = joint_names
        self.key_points = key_points
        self.scale = scale
        self.csv_data_list = []
        self.scene = None
        self.target_dict = {}
        self.point_list = []
        self.robot = None
        self.data_in_play = 0 # 正在播放的数据集的索引
        self.frame_in_play = 0 # 正在播放的数据集的帧索引
        self.start_frame = 0 
        self.end_frame = 0
        self.play = True
        self.back = False
        self.record = False
        self.record_stack = {
            'global_root_velocity': [],
            'global_root_angular_velocity': [],
            'global_translation': [],
            'global_rotation': [],
            'local_rotation': [],
            'global_velocity': [],
            'global_angular_velocity': [],
            'dof_pos': [],
            'dof_vels': [],
        }
        self.fps = 120
        self.init()

    def load_csv_files_from_folder(self):
        """
        从指定文件夹中读取所有的 CSV 文件，并提取表头信息。
        """
        for filename in os.listdir(self.mocap_file):
            if filename.endswith('.csv'):
                file_path = os.path.join(self.mocap_file, filename)
                data = pd.read_csv(file_path)
                header = data.columns.tolist()
                
                if self.point_names is not None:
                    filtered_header = [col for col in header if any(name in col for name in self.point_names)]
                    data = data[filtered_header]
                    header = filtered_header
                
                for point_name in self.point_names:
                        x_col = f"{point_name}.X"
                        y_col = f"{point_name}.Y"
                        z_col = f"{point_name}.Z"
                        
                        if x_col in data.columns and y_col in data.columns and z_col in data.columns:
                            # 应用缩放系数
                            data[x_col] *= self.scale[0]
                            data[y_col] *= self.scale[1]
                            data[z_col] *= self.scale[2]
                
                # 数据平滑处理
                window_size = 5  # 移动平均窗口大小
                data_smoothed = data.rolling(window=window_size, center=True).mean().fillna(method='bfill').fillna(method='ffill')
                
                # 检测和处理跳变
                threshold = 0.1  # 跳变阈值
                diff = data_smoothed.diff().abs()
                jumps = (diff > threshold).any(axis=1)
                
                # 处理跳变帧
                data_cleaned = data_smoothed.copy()
                i = 1
                while i < len(data_cleaned) - 1:
                    if jumps[i]:
                        # 找到连续跳变帧的开始和结束
                        start_jump = i
                        while i < len(data_cleaned) - 1 and jumps[i]:
                            i += 1
                        end_jump = i
                        
                        # 使用两侧帧的二次拟合来替代跳变帧
                        if start_jump > 1 and end_jump < len(data_cleaned) - 2:
                            x = np.array([start_jump-2, start_jump-1, end_jump+1, end_jump+2])
                            for col in data_cleaned.columns:
                                y = data_cleaned[col].iloc[x].values
                                coefficients = np.polyfit(x, y, 2)
                                poly = np.poly1d(coefficients)
                                x_new = np.arange(start_jump, end_jump)
                                y_new = poly(x_new)
                                data_cleaned[col].iloc[start_jump:end_jump] = y_new
                    i += 1
                
                self.csv_data_list.append({
                    'data': data_cleaned,
                    'length': data_cleaned.shape[0],  # 添加数据长度
                    'header': header,
                    'filename': filename,
                })


    def calculate_baseqps(self):
        pass

    def add_axis(self):
        """
        在场景中添加多个轴对象，并将它们存储在字典中。
        """
        self.target_dict = {}
        for name in self.name_list:
            target_name = name + "_target"
            target = self.scene.add_entity(
                gs.morphs.Mesh(
                    file="meshes/axis.obj",
                    scale=0.5,
                ),
                surface=gs.surfaces.Default(color=(1, 0.5, 0.5, 1)),
            )
            self.target_dict[target_name] = target

    def set_axis(self):
        self.base_pos = np.zeros((1, 3),  dtype=gs.tc_float)
        self.base_quat = np.zeros((1, 4),  dtype=gs.tc_float)
        self.base_pos[:] = self.robot.get_pos()
        self.base_quat[:] = self.robot.get_quat()           
        for name in self.name_list:
 
            target_name = name + "_target"
            self.target_dict[target_name].set_pos([0,0,0])        

    def add_points(self, csv_info):
        """
        根据 CSV 数据在场景中添加点对象，并返回一个包含点名称和对应点对象的列表。
        """
        data = csv_info['data']
        header = csv_info['header']
        
        columns = [col for col in header if col.endswith('.X') or col.endswith('.Y') or col.endswith('.Z')]
        unique_points = set()
        self.point_list = []
        
        for index, row in data.iterrows():
            for i in range(0, len(columns), 3):
                x_col = columns[i]
                point_name = x_col[:-2]
                
                if point_name not in unique_points:
                    unique_points.add(point_name)
                    
                    if self.key_points is not None and point_name in self.key_points:
                        color = (0.8, 0.0, 0.0, 1)
                    else:
                        color = (0.5, 0.5, 0.5, 1)
                    
                    point = self.scene.add_entity(
                        gs.morphs.Sphere(
                            radius=0.01,
                        ),
                        surface=gs.surfaces.Default(color=color),
                    )
                    self.point_list.append({'name': point_name, 'point': point})

    def initialize_scene(self, vis):
        """
        初始化场景并添加必要的实体。
        """
        gs.init(seed=0, precision="32", logging_level="debug")
        self.scene = gs.Scene(
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(2.5, 0.0, 1.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=40,
            ),
            show_viewer=vis,
            rigid_options=gs.options.RigidOptions(
                gravity=(0, 0, 0),
                enable_collision=False,
                enable_joint_limit=False,
            ),
        )
        self.scene.add_entity(gs.morphs.Plane())
        self.robot = self.scene.add_entity(
            gs.morphs.URDF(
                file=self.urdf_file,
                pos=(0, 0, 0.4),
                merge_fixed_links = True,
                links_to_keep = ['FL_calf_rotor', 'FR_calf_rotor', 'RL_calf_rotor', 'RR_calf_rotor',
                                 'FL_foot', 'FR_foot', 'RL_foot', 'RR_foot'],
                fixed = True
                
            ),
        )
        
        self.FL_link = self.robot.get_link("FL_foot")
        self.FR_link = self.robot.get_link("FR_foot")
        self.RL_link = self.robot.get_link('RL_foot')
        self.RR_link = self.robot.get_link('RR_foot')
        
        self.FL_sholder = self.robot.get_link('FL_calf_rotor')
        self.FR_sholder = self.robot.get_link('FR_calf_rotor')
        self.RL_sholder = self.robot.get_link('RL_calf_rotor')
        self.RR_sholder = self.robot.get_link('RR_calf_rotor')
        
        self.FL_thigh = self.robot.get_link('FL_calf')
        self.FR_thigh = self.robot.get_link('FR_calf')
        self.RL_thigh = self.robot.get_link('RL_calf')
        self.RR_thigh = self.robot.get_link('RR_calf')
        
      
        

        self.add_axis()
        self.add_points(self.csv_data_list[0])
        self.scene.build()
        self.scene.reset()

        self.motion_link_idxs = []
        for link in self.robot._links:
            if link.name not in ['FL_foot', 'FR_foot', 'RL_foot', 'RR_foot', 
                                 'FL_calf_rotor', 'FR_calf_rotor', 'RL_calf_rotor', 'RR_calf_rotor'
                                 ]:
                self.motion_link_idxs.append(link.idx_local)  

    def play_frame(self,csv_data,row):
        frame = csv_data['data'].iloc[row]
        for point_info in self.point_list:
            point_name = point_info['name']
            x_col = f"{point_name}.X"
            y_col = f"{point_name}.Y"
            z_col = f"{point_name}.Z"
            
            if x_col in frame and y_col in frame and z_col in frame:
                position = np.array([frame[x_col], frame[z_col], frame[y_col]])
                point_info['point'].set_pos(position)

    def set_color(self,points, color):
        for point_info in self.point_list:
            if point_info['name'] == points:
                point_info['point'].surface.color = color
        self.scene.step()

    def init(self):
        self.load_csv_files_from_folder()
        parser = argparse.ArgumentParser()
        parser.add_argument("-v", "--vis", action="store_true", default=True)
        args = parser.parse_args()

        self.initialize_scene(args.vis)

        target_quat = np.array([1, 0, 0, 0])
        
    def play_data(self):
        prev_curr = None
        while True:
            if self.frame_in_play >= self.csv_data_list[self.data_in_play]['length'] or self.frame_in_play >= self.end_frame:
                # 循环播放一个文件，如果到达文件末尾，重新播放
                self.record = False
                self.frame_in_play = self.start_frame

            dt = 1/120
            self.play_frame(self.csv_data_list[self.data_in_play], self.frame_in_play)
            curr = self.IK()

            if self.record:
                if prev_curr is None :
                    prev_curr = curr
                else:
                    data_to_record = self.collect_data(
                        prev_curr,
                        curr,
                        dt=dt
                    )
                    self.record_stack['global_root_velocity'].append(data_to_record['global_root_velocity'])
                    self.record_stack['global_root_angular_velocity'].append(data_to_record['global_root_angular_velocity'])
                    self.record_stack['global_translation'].append(data_to_record['global_translation'])
                    self.record_stack['global_rotation'].append(data_to_record['global_rotation'])
                    self.record_stack['local_rotation'].append(data_to_record['local_rotation'])
                    self.record_stack['global_velocity'].append(data_to_record['global_velocity'])
                    self.record_stack['global_angular_velocity'].append(data_to_record['global_angular_velocity'])
                    self.record_stack['dof_pos'].append(data_to_record['dof_pos'])
                    self.record_stack['dof_vels'].append(data_to_record['dof_vels'])
                    
                    
                prev_curr = curr

            self.scene.step()
            if self.play:
                self.frame_in_play += 1

    def calculate_linear_velocity(self, root_pos, prev_root_pos, dt):
        """
        计算根节点的位置线速度。

        参数:
        - root_pos (np.ndarray): 当前帧的根节点位置。
        - prev_root_pos (np.ndarray): 上一帧的根节点位置。
        - dt (float): 时间步长。

        返回:
        - linear_velocity (np.ndarray): 根节点的位置线速度。
        """
        linear_velocity = (root_pos - prev_root_pos) / dt
        return linear_velocity

    def calculate_angular_velocity(self, global_rotation, prev_global_rotations, dt):
        """
        计算根节点的角速度。

        参数:
        - root_rot (np.ndarray): 当前帧的根节点四元数旋转表示。
        - prev_root_rot (np.ndarray): 上一帧的根节点四元数旋转表示。
        - dt (float): 时间步长。

        返回:
        - angular_velocity (np.ndarray): 根节点的角速度。
        """
        # 计算四元数差
        angular_velocity = np.zeros([global_rotation.shape[0], 3])
        for i in range(global_rotation.shape[0]):
            delta_quat = quaternion_multiply(global_rotation[i], quaternion_multiply(prev_global_rotations[i], torch.tensor([0, 0, 0, -1])))
            delta_quat = delta_quat / torch.linalg.norm(delta_quat)
            delta_quat = delta_quat.cpu().numpy()
            # 计算角速度
            theta = 2 * np.arccos(delta_quat[0])
            if theta < 1e-6:
                angular_velocity[i] = np.array([0, 0, 0])
            else:
                sin_half_theta = np.sin(theta / 2)
                axis = delta_quat[1:] / sin_half_theta
                angular_velocity[i] = axis * theta / dt
        
        return angular_velocity

    def calculate_joint_angular_velocities(self, qpos, prev_qpos, dt):
        """
        计算每个关节的角速度。

        参数:
        - qpos (np.ndarray): 当前帧的关节位置。
        - prev_qpos (np.ndarray): 上一帧的关节位置。
        - dt (float): 时间步长。

        返回:
        - joint_angular_velocities (np.ndarray): 每个关节的角速度。
        """
        joint_angular_velocities = (qpos - prev_qpos) / dt
        return joint_angular_velocities

    def collect_data(self, prev_curr, curr, dt):
        """
        收集当前帧的数据，并计算线速度、角速度和关节角速度。

        参数:
        - prev_curr (dict): 上一帧的数据字典，包含 'global_translation', 'global_rotation', 'local_rotation', 'dof_pos'。
        - curr (dict): 当前帧的数据字典，包含 'global_translation', 'global_rotation', 'local_rotation', 'dof_pos'。
        - dt (float): 时间步长。

        返回:
        - data (dict): 包含所有数据的字典。
        """
        global_translation = curr['global_translation']
        global_rotation = curr['global_rotation']
        dof_pos = curr['dof_pos']
        
        prev_global_translations = prev_curr['global_translation']
        prev_global_rotations = prev_curr['global_rotation']
        prev_dof_pos = prev_curr['dof_pos']

        # 计算线速度
        linear_velocity = self.calculate_linear_velocity(global_translation, prev_global_translations, dt)
        _linear_velocity = torch.tensor(linear_velocity).cpu()

        # 计算角速度
        angular_velocity = self.calculate_angular_velocity(global_rotation, prev_global_rotations, dt)
        _angular_velocity = torch.tensor(angular_velocity).cpu()

        # 计算关节角速度
        joint_angular_velocities = self.calculate_joint_angular_velocities(dof_pos, prev_dof_pos, dt)
        _joint_angular_velocities = torch.tensor(joint_angular_velocities).cpu()

        data = {
            'global_root_velocity': torch.tensor(linear_velocity[0]).cpu().tolist(),
            'global_root_angular_velocity': torch.tensor(angular_velocity[0]).cpu().tolist(),

            'global_translation': curr['global_translation'],
            'global_rotation': curr['global_rotation'],
            'local_rotation': curr['local_rotation'],

            'global_velocity': _linear_velocity.tolist(),
            'global_angular_velocity': _angular_velocity.tolist(),
                        
            'dof_pos': curr['dof_pos'],
            'dof_vels': _joint_angular_velocities.tolist(),
        }

        return data


    def calculate_root_state(self):
        """
        根据参考关节位置计算当前帧的根节点位置和旋转。
        
        返回:
        - root_pos (np.ndarray): 计算得到的当前帧的根节点位置。
        - root_rot (np.ndarray): 当前帧的根节点的四元数旋转表示。
        """
        # 获取当前帧数据
        csv_data = self.csv_data_list[self.data_in_play]['data']
        frame = csv_data.iloc[self.frame_in_play]

        # 定义关键点名称
        key_point_names = [
            "Bip01",
            "b__Neck",
            "b_LeftArm",
            "b_RightArm",
            "b_LeftLegUpper",
            "b_RightLegUpper"
        ]

        # 初始化关键点位置字典
        key_point_positions = {}

        # 获取关键点位置
        for point_name in key_point_names:
            x_col = f"{point_name}.X"
            y_col = f"{point_name}.Y"
            z_col = f"{point_name}.Z"
            
            if x_col in frame and y_col in frame and z_col in frame:
                position = np.array([frame[x_col], frame[z_col], frame[y_col]], dtype=np.float64)
                key_point_positions[point_name] = position

        # 确保所有关键点都存在
        required_key_points = set(key_point_names)
        if not required_key_points.issubset(key_point_positions.keys()):
            raise ValueError("Required key points are not present in the current frame.")

        # 提取关键点位置
        pelvis_pos = key_point_positions["Bip01"]
        neck_pos = key_point_positions["b__Neck"]
        left_shoulder_pos = key_point_positions["b_LeftArm"]
        right_shoulder_pos = key_point_positions["b_RightArm"]
        left_hip_pos = key_point_positions["b_LeftLegUpper"]
        right_hip_pos = key_point_positions["b_RightLegUpper"]

        # 计算前向方向
        forward_dir = neck_pos - pelvis_pos
        forward_dir = forward_dir / np.linalg.norm(forward_dir)

        # 计算左右方向
        delta_shoulder = left_shoulder_pos - right_shoulder_pos
        delta_hip = left_hip_pos - right_hip_pos
        dir_shoulder = delta_shoulder / np.linalg.norm(delta_shoulder)
        dir_hip = delta_hip / np.linalg.norm(delta_hip)
        left_dir = 0.5 * (dir_shoulder + dir_hip)

        # 确保正交性
        up_dir = np.cross(left_dir,forward_dir)
        up_dir = up_dir / np.linalg.norm(up_dir)
        
        
        left_dir = np.cross(up_dir, forward_dir)
        #left_dir[2] = 0.0  # make the base more stable
        left_dir = left_dir / np.linalg.norm(left_dir)

        rot_mat = np.array(
            [
                [forward_dir[0], left_dir[0], up_dir[0], 0],
                [forward_dir[1], left_dir[1], up_dir[1], 0],
                [forward_dir[2], left_dir[2], up_dir[2], 0],
                [0, 0, 0, 1],
            ]
        )

        #root_pos = 0.5 * (pelvis_pos + neck_pos)
        root_pos = 0.25 * (left_shoulder_pos + right_shoulder_pos + left_hip_pos + right_hip_pos)
        root_rot = quaternion_from_matrix(rot_mat)
        #root_rot = quaternion_multiply(root_rot, INIT_ROT)
        root_rot = root_rot / np.linalg.norm(root_rot)
        
        # 将 root_rot 的最后一列放到第一列
        root_rot = np.roll(root_rot, 1)
        return root_pos, root_rot
    def offset(self, root_rot, point_name, offset_amount):
        """
        根据局部坐标系下的偏移量对指定点进行偏移。

        参数:
        - root_pos (np.ndarray): 根节点的位置。
        - root_rot (np.ndarray): 根节点的四元数旋转表示。
        - point_name (str): 要偏移的点的名称。
        - offset_amount (np.ndarray): 局部坐标系下的偏移量。
        """
        # 检查点是否存在
        point_info = next((p for p in self.point_list if p['name'] == point_name), None)
        if point_info is None:
            print(f"Point {point_name} not found.")
            return

        # 获取点的当前位置
        current_pos = point_info['point'].get_pos()

        # 确保 current_pos 是 numpy.ndarray
        if isinstance(current_pos, torch.Tensor):
            current_pos = current_pos.cpu().numpy()

        # 将偏移量从局部坐标系转换到全局坐标系
        rotation_matrix = R.from_quat(root_rot).as_matrix()
        # 将 offset_amount 从 torch.Tensor 转换为 numpy.ndarray
        # 首先将 tensor 从 CUDA 设备复制到 CPU 内存
        offset_amount_np = offset_amount.cpu().numpy()
        global_offset = rotation_matrix @ offset_amount_np

        # 计算新的位置
        new_pos = current_pos + global_offset
        
        # 设置点的新位置
        point_info['point'].set_pos(new_pos)
    def IK(self):
        root_pos, root_rot = self.calculate_root_state()
        self.target_dict['Base_target'].set_pos(root_pos)
        self.target_dict['Base_target'].set_quat(root_rot)
        
        self.robot.set_pos(root_pos)
        self.robot.set_quat(root_rot)
        #self.scene.step()   
        
        pos = self.robot.get_links_pos()
        quat = self.robot.get_links_quat()

        self.target_dict['FL_target'].set_pos(pos[self.FL_thigh.idx_local])
        self.target_dict['FR_target'].set_pos(pos[self.FR_thigh.idx_local])
        self.target_dict['RL_target'].set_pos(pos[self.RL_thigh.idx_local])
        self.target_dict['RR_target'].set_pos(pos[self.RR_thigh.idx_local])
        self.target_dict['FL_target'].set_quat(quat[self.FL_thigh.idx_local])
        self.target_dict['FR_target'].set_quat(quat[self.FR_thigh.idx_local])
        self.target_dict['RL_target'].set_quat(quat[self.RL_thigh.idx_local])
        self.target_dict['RR_target'].set_quat(quat[self.RR_thigh.idx_local])

        for point in self.point_list:
            if point["name"] == "b_RightAnkle":
                RLF_target = point['point'].get_pos()
            if point["name"] == "b_LeftAnkle":
                RRF_target = point['point'].get_pos()
            if point["name"] == "b_RightFinger":
                FLF_target = point['point'].get_pos()
            if point["name"] == "b__LeftFinger":
                FRF_target = point['point'].get_pos()
                
            if point["name"] == "b_RightLegUpper":
                RLS_target = point['point'].get_pos()
            if point["name"] == "b_LeftLegUpper":
                RRS_target = point['point'].get_pos()
            if point["name"] == "b_RightArm":
                FLS_target = point['point'].get_pos()
            if point["name"] == "b_LeftArm":
                FRS_target = point['point'].get_pos()
                
        
        delta_FR = FRF_target - FRS_target
        delta_FL = FLF_target - FLS_target
        delta_RR = RRF_target - RRS_target
        delta_RL = RLF_target - RLS_target
        
        FLF_target = delta_FL + pos[self.FL_sholder.idx_local]
        FRS_target = delta_FR + pos[self.FR_sholder.idx_local]
        RLF_target = delta_RL + pos[self.RL_sholder.idx_local]
        RRF_target = delta_RR + pos[self.RR_sholder.idx_local]
                 

        qpos = self.robot.inverse_kinematics_multilink(
            links=[self.FL_link, self.FR_link, self.RL_link, self.RR_link],  # 指定需要控制的链接
            poss=[FLF_target, FRF_target, RLF_target, RRF_target], 
            max_samples=25,
            max_solver_iters=10,# 指定目标位置
        ) 
        self.robot.set_qpos(qpos)
        

                
                
        
        
        
        global_translation = self.robot.get_links_pos()
        global_rotation = self.robot.get_links_quat()
        local_rotation = self.calculate_local_rotations(global_rotation, root_rot)

        global_translation = global_translation[self.motion_link_idxs]
        global_rotation = global_rotation[self.motion_link_idxs]
        local_rotation = local_rotation[self.motion_link_idxs]
        curr = {}
        curr['global_translation'] = global_translation
        curr['global_rotation'] = global_rotation
        curr['local_rotation'] = local_rotation
        curr['dof_pos'] = qpos
        return curr



    def calculate_local_rotations(self,global_rotation, root_rot):
        """
        计算每个链接的局部旋转。

        参数:
        - global_rotation (torch.Tensor): 所有链接的全局旋转四元数，形状为 (n_links, 4)。
        - root_rot (torch.Tensor): 根链接的全局旋转四元数，形状为 (4,)。

        返回:
        - local_rotation (torch.Tensor): 所有链接的局部旋转四元数，形状为 (n_links, 4)。
        """
        n_links = global_rotation.shape[0]
        local_rotation = torch.zeros_like(global_rotation)

        for i in range(n_links):
            # 获取当前链接的全局旋转
            current_rot = global_rotation[i]

            # 如果没有父链接，则局部旋转等于全局旋转
            if i == 0:  # 假设根链接的索引为 0
                local_rotation[i] = current_rot
            else:
                # 获取父链接的全局旋转
                parent_rot = global_rotation[self.robot._links[i].parent_idx_local]# 假设父链接的索引为 i - 1

                # 计算局部旋转：当前旋转减去父旋转
                # 四元数的减法需要特殊处理，这里使用四元数的乘法逆
                parent_rot_inv = quaternion_inverse(parent_rot)
                local_rotation[i] = quaternion_multiply(current_rot, parent_rot_inv)

        return local_rotation

def quaternion_inverse(q):
    """
    计算四元数的逆。

    参数:
    - q (torch.Tensor): 四元数，形状为 (4,)。

    返回:
    - q_inv (torch.Tensor): 四元数的逆，形状为 (4,)。
    """
    q_inv = torch.tensor([q[0], -q[1], -q[2], -q[3]], dtype=q.dtype, device=q.device)
    q_inv = q_inv / torch.norm(q)
    return q_inv

def quaternion_multiply(quaternion1, quaternion0):
    """
    返回两个四元数的乘积（PyTorch 版本）
    """
    w0, x0, y0, z0 = quaternion0.unbind(dim=-1)
    w1, x1, y1, z1 = quaternion1.unbind(dim=-1)
    
    return torch.stack((
        -x1*x0 - y1*y0 - z1*z0 + w1*w0,
         x1*w0 + y1*z0 - z1*y0 + w1*x0,
        -x1*z0 + y1*w0 + z1*x0 + w1*y0,
         x1*y0 - y1*x0 + z1*w0 + w1*z0
    ), dim=-1)
import plotly.graph_objs as go
import plotly.express as px
from dash import Dash, dcc, html
from dash.dependencies import Input, Output
import webbrowser

class TkinterUI:
    def __init__(self, master, motion_retarget: MotionRetarget,csv_header):
        self.master = master  # 设置主窗口
        self.motion_retarget = motion_retarget  # 设置 MotionRetarget 实例
        self.current_data_index = None  # 当前选中的数据集索引
        self.current_frame = 0  # 当前帧索引
        self.is_playing = False  # 播放状态标志
        self.csv_header = csv_header  # 存储CSV表头

        # 创建一个框架来存放数据按钮
        self.data_frame = ttk.Frame(master)
        self.data_frame.pack(pady=10, side=tk.LEFT, fill=tk.Y)  # 将框架添加到窗口

        self.data_buttons = []  # 存储数据按钮的列表
        for i, data_info in enumerate(self.motion_retarget.csv_data_list):
            button = ttk.Button(self.data_frame, text=data_info['filename'], command=lambda i=i: self.load_data(i))  # 创建按钮，点击时加载数据
            button.pack(side=tk.TOP, fill=tk.X)  # 将按钮添加到框架中
            self.data_buttons.append(button)  # 将按钮添加到列表中

        # 创建一个框架来存放球体按键
        self.points_frame = ttk.Frame(master)
        self.points_frame.pack(pady=10, side=tk.LEFT, fill=tk.Y)  # 将框架添加到窗口

        self.point_buttons = {}  # 存储球体按键的字典
        self.point_states = {}  # 存储球体按键的状态

        # 创建球体按键
        for point_info in self.motion_retarget.point_list:
            point_name = point_info['name']
            button = ttk.Button(self.points_frame, text=point_name, command=lambda name=point_name: self.toggle_point(name))
            button.pack(side=tk.TOP, fill=tk.X)
            self.point_buttons[point_name] = button
            self.point_states[point_name] = False

            # 根据 key_points 初始化按键状态
            if point_name in self.motion_retarget.key_points:
                self.toggle_point(point_name)

        self.progress_var = tk.IntVar()  # 进度条变量
        self.progress_bar = ttk.Progressbar(master, variable=self.progress_var, maximum=100)  # 创建进度条
        self.progress_bar.pack()  # 将进度条添加到窗口

        self.frame_label = ttk.Label(master, text="Frame: 0")  # 创建帧标签
        self.frame_label.pack()  # 将帧标签添加到窗口

        self.total_frames_label = ttk.Label(master, text="Total Frames: 0")  # 创建总帧数标签
        self.total_frames_label.pack()  # 将总帧数标签添加到窗口

        self.play_button = ttk.Button(master, text="Play", command=self.toggle_play)  # 创建播放/暂停按钮
        self.play_button.pack()  # 将播放/暂停按钮添加到窗口

        self.start_frame_entry = ttk.Entry(master)  # 创建起始帧输入框
        self.start_frame_entry.pack()  # 将起始帧输入框添加到窗口
        self.start_frame_entry.insert(0, "0")  # 设置起始帧输入框的默认值为 0

        self.set_start_frame_button = ttk.Button(master, text="Set Start Frame", command=self.set_start_frame)  # 创建设置起始帧按钮
        self.set_start_frame_button.pack()  # 将设置起始帧按钮添加到窗口

        self.end_frame_entry = ttk.Entry(master)  # 创建结束帧输入框
        self.end_frame_entry.pack()  # 将结束帧输入框添加到窗口
        self.end_frame_entry.insert(0, "100")  # 设置结束帧输入框的默认值为 100

        self.set_end_frame_button = ttk.Button(master, text="Set End Frame", command=self.set_end_frame)  # 创建设置结束帧按钮
        self.set_end_frame_button.pack()  # 将设置结束帧按钮添加到窗口

        self.frame_entry = ttk.Entry(master)  # 创建手动输入帧号的输入框
        self.frame_entry.pack()  # 将输入框添加到窗口

        self.jump_button = ttk.Button(master, text="Jump to Frame", command=self.jump_to_frame)  # 创建跳转按钮
        self.jump_button.pack()  # 将跳转按钮添加到窗口

        # 添加记录按钮
        self.record_button = ttk.Button(master, text="Record", command=self.toggle_record)  # 创建记录按钮
        self.record_button.pack()  # 将记录按钮添加到窗口
        # 添加平滑数据按钮
        self.smooth_button = ttk.Button(master, text="Smooth Data", command=self.smooth_data)
        self.smooth_button.pack()
        # 添加导出按钮
        self.export_button = ttk.Button(master, text="Export", command=self.export_data)  # 创建导出按钮
        self.export_button.pack()  # 将导出按钮添加到窗口


        self.update_progress()  # 启动进度条更新循环

    def load_data(self, index):
        self.current_data_index = index
        self.current_frame = 0
        data_info = self.motion_retarget.csv_data_list[index]
        self.progress_var.set(0)
        self.progress_bar['maximum'] = data_info['length']
        self.frame_label.config(text=f"Frame: {self.current_frame}")
        self.total_frames_label.config(text=f"Total Frames: {data_info['length']}")  # 更新总帧数标签
        self.motion_retarget.data_in_play = index
        self.motion_retarget.frame_in_play = 0
        self.motion_retarget.start_frame = 0
        self.motion_retarget.end_frame = data_info['length']

    def toggle_play(self):
        self.is_playing = not self.is_playing
        if self.is_playing:
            self.play_button.config(text="Pause")
            self.motion_retarget.play = False
        else:
            self.play_button.config(text="Play")
            self.motion_retarget.play = True

    def update_progress(self):
        if self.current_data_index is not None:
            data_info = self.motion_retarget.csv_data_list[self.current_data_index]
            self.current_frame = self.motion_retarget.frame_in_play
            self.progress_var.set(self.current_frame)
            self.frame_label.config(text=f"Frame: {self.current_frame}")
            self.progress_bar['maximum'] = data_info['length']
        self.master.after(100, self.update_progress)  # 每100毫秒更新一次

    def jump_to_frame(self):
        try:
            frame_number = int(self.frame_entry.get())
            if 0 <= frame_number < self.motion_retarget.csv_data_list[self.current_data_index]['length']:
                self.motion_retarget.frame_in_play = frame_number
                self.update_progress()
            else:
                tk.messagebox.showerror("Invalid Input", "Frame number out of range.")
        except ValueError:
            tk.messagebox.showerror("Invalid Input", "Please enter a valid frame number.")

    def toggle_point(self, point_name):
        if self.point_states[point_name]:
            color = (0.5, 0.5, 0.5, 1.0)  # 未选中时变为灰色
            if point_name in self.motion_retarget.key_points:
                self.motion_retarget.key_points.remove(point_name)
        else:
            color = (1.0, 0.0, 0.0, 1.0)  # 选中时变为红色
            if point_name not in self.motion_retarget.key_points:
                self.motion_retarget.key_points.append(point_name)
        
        self.point_states[point_name] = not self.point_states[point_name]
        self.motion_retarget.set_color(point_name, color)

    def set_start_frame(self):
        try:
            start_frame = int(self.start_frame_entry.get())
            if 0 <= start_frame < self.motion_retarget.csv_data_list[self.current_data_index]['length']:
                self.motion_retarget.start_frame = start_frame
            else:
                tk.messagebox.showerror("Invalid Input", "Start frame number out of range.")
        except ValueError:
            tk.messagebox.showerror("Invalid Input", "Please enter a valid start frame number.")

    def toggle_record(self):
        self.motion_retarget.record = not self.motion_retarget.record
        if self.motion_retarget.record:
            self.record_button.config(text="Stop Recording")
            self.motion_retarget.play = True  # 开始播放数据
            self.is_playing = True
            self.play_button.config(text="Pause")
        else:
            self.record_button.config(text="Record")
            self.motion_retarget.play = False  # 停止播放数据
            self.is_playing = False
            self.play_button.config(text="Play")

    def set_end_frame(self):
        try:
            end_frame = int(self.end_frame_entry.get())
            if 0 <= end_frame < self.motion_retarget.csv_data_list[self.current_data_index]['length']:
                self.motion_retarget.end_frame = end_frame
            else:
                tk.messagebox.showerror("Invalid Input", "End frame number out of range.")
        except ValueError:
            tk.messagebox.showerror("Invalid Input", "Please enter a valid end frame number.")


    def butter_lowpass(self, cutoff, fs, order=5):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        if not 0 < normal_cutoff < 1:
            raise ValueError("Normal cutoff frequency must be between 0 and 1")
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return b, a

    def lowpass_filter(self, data, cutoff, fs, order=5):
        b, a = self.butter_lowpass(cutoff, fs, order=order)
        y = lfilter(b, a, data)
        return y
    def plot_smoothed_data(self, original, smoothed, joint_indices):
        """
        可视化原始数据与平滑后数据的对比
        
        参数:
        - original: numpy数组，原始关节角速度数据 [n_frames, n_joints]
        - smoothed: numpy数组，平滑后的关节角速度数据 [n_frames, n_joints]
        - joint_indices: 需要可视化的关节索引列表
        """
        # 转换数据为适合Plotly的格式
        original = np.asarray(original)
        smoothed = np.asarray(smoothed)
        
        # 创建Dash应用
        app = Dash(__name__)
        
        # 生成图表布局
        app.layout = html.Div([
            html.H1("关节速度平滑效果对比", style={'textAlign': 'center'}),
            html.Div([
                dcc.Dropdown(
                    id='joint-dropdown',
                    options=[{'label': f'关节 {i}', 'value': i} for i in joint_indices],
                    value=0,
                    style={'width': '50%', 'margin': 'auto'}
                )
            ], style={'padding': 20}),
            dcc.Graph(id='velocity-plot', style={'height': '80vh'})
        ])

        @app.callback(
            Output('velocity-plot', 'figure'),
            [Input('joint-dropdown', 'value')]
        )
        def update_plot(selected_joint):
            # 确保索引在合法范围内
            selected_joint = min(max(selected_joint, 0), original.shape[1]-1)
            
            # 创建轨迹数据
            frames = np.arange(original.shape[0])
            
            fig = go.Figure()
            
            # 添加原始数据轨迹
            fig.add_trace(go.Scatter(
                x=frames,
                y=original[:, selected_joint],
                name='原始数据',
                line=dict(color='#1f77b4', width=1.5),
                opacity=0.7
            ))
            
            # 添加平滑数据轨迹
            fig.add_trace(go.Scatter(
                x=frames,
                y=smoothed[:, selected_joint],
                name='平滑数据',
                line=dict(color='#ff7f0e', width=2, dash='dash'),
                opacity=0.9
            ))
            
            # 更新布局
            fig.update_layout(
                title=f'关节 {selected_joint} 角速度对比',
                xaxis_title='帧序号',
                yaxis_title='角速度 (rad/s)',
                hovermode='x unified',
                template='plotly_white',
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                margin=dict(l=50, r=50, b=50, t=80),
                height=600
            )
            
            # 添加辅助线
            fig.add_shape(
                type="line",
                x0=0, x1=frames[-1],
                y0=0, y1=0,
                line=dict(color="grey", width=1, dash="dot")
            )
            
            return fig

        # 异步启动可视化
        def run_server():
            app.run_server(debug=False, use_reloader=False)
        
        threading.Thread(target=run_server).start()
        webbrowser.open('http://127.0.0.1:8050/')

    def smooth_data(self):
        if not self.motion_retarget.record_stack:
            tk.messagebox.showinfo("No Data", "No data to smooth.")
            return

        # 转换为numpy数组
        dof_vels = np.array(self.motion_retarget.record_stack['dof_vels'])
        
        # 定义要平滑的关节索引
        JOINT_INDICES = list(range(dof_vels.shape[1]))  # 自动获取关节数量
        
        # 参数设置
        fs = 120
        cutoff = 50
        order = 5

        # 对每个关节进行滤波
        smoothed = np.zeros_like(dof_vels)
        for joint_idx in JOINT_INDICES:
            smoothed[:, joint_idx] = self.lowpass_filter(
                dof_vels[:, joint_idx], 
                cutoff, 
                fs, 
                order
            )

        # 转换回列表格式并更新
        self.motion_retarget.record_stack['dof_vels'] = smoothed.tolist()

        # 可视化（保持原有代码不变）
        self.plot_smoothed_data(
            original=dof_vels,
            smoothed=smoothed,
            joint_indices=JOINT_INDICES
        )

        tk.messagebox.showinfo("Smoothing Complete", "Joint velocities have been smoothed.")
    def export_data(self):
        if not self.motion_retarget.record_stack:
            tk.messagebox.showinfo("No Data", "No data to export.")
            return

        # 获取正在播放的数据文件名
        if self.current_data_index is None:
            tk.messagebox.showinfo("No Data", "No data file is currently playing.")
            return

        current_data_info = self.motion_retarget.csv_data_list[self.current_data_index]
        filename = os.path.splitext(current_data_info['filename'])[0] + ".npz"  # 修改扩展名为pth

        # 确保datasets文件夹存在
        datasets_folder = os.path.join(os.path.dirname(__file__), '..', 'datasets')
        if not os.path.exists(datasets_folder):
            os.makedirs(datasets_folder)

        # 构建输出路径
        output_path = os.path.join(datasets_folder, filename)

        # 保存数据
        torch.save(self.motion_retarget.record_stack, output_path)
        
        # 清空record_stack
        self.motion_retarget.record_stack = {
            'global_root_velocity': [],
            'global_root_angular_velocity': [],
            'global_translation': [],
            'global_rotation': [],
            'local_rotation': [],
            'global_velocity': [],
            'global_angular_velocity': [],
            'dof_pos': [],
            'dof_vels': [],
        }



if __name__ == "__main__":
    motion_retarget = MotionRetarget(
        mocap_file="datasets/lssp_keypoints",
        urdf_file="datasets/go2_description/urdf/go2_description.urdf",
        name_list=[
            "Base",
            "FR",
            "FL",
            "RR",
            "RL",
        ],
        point_names=[
            "Bip01",
            "b_Hips",
            "b_RightLegUpper",
            "Dog_RightArmor001",
            "Dog_RightArmor001End",
            "b_RightLeg",
            "b_RightLeg1",
            "b_RightAnkle",
            "b_RightToe002",
            "b_RightToe002End",
            "b_RightToe",
            "Bip01_R_Toe0Nub",
            "Bip01_R_Toe0NubEnd",
            "b_LeftLegUpper",
            "Dog_LeftArmor001",
            "Dog_LeftArmor001End",
            "b_LeftLeg",
            "b_LeftLeg1",
            "b_LeftAnkle",
            "b_LeftToe002",
            "b_LeftToe002End",
            "b_LeftToe",
            "Bip01_L_Toe0Nub",
            "Bip01_L_Toe0NubEnd",
            "b_Tail009End",
            "b_Spine",
            "b_Spine1",
            "b_Spine2",
            "b_Spine3",
            "b_RightClav",
            "b_RightArm",
            "Dog_RightArmor002",
            "Dog_RightArmor002End",
            "b_RightForeArm",
            "b_RightHand",
            "b_RightFinger",
            "Bip01_R_Finger0Nub",
            "Bip01_R_Finger0NubEnd",
            "b_LeftClav",
            "b_LeftArm",
            "Dog_LeftArmor002",
            "Dog_LeftArmor002End",
            "b_LeftForeArm",
            "b_LeftHand",
            "b__LeftFinger",
            "Bip01_L_Finger0Nub",
            "Bip01_L_Finger0NubEnd",
            "b__Neck",
            "b__Neck1",
            "b__Neck2",
            "b_Head",
            "Dog_Jaw",
            "Dog_JawEnd",
            #"Bip01_HeadNub",
            #"Bip01_HeadNubEnd",
            #"Bip01_Footsteps",
            #"Bip01_FootstepsEnd"
        ],
        joint_names=[
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
        ],
        key_points=[
            "Bip01",
            "b__Neck",
            "b_LeftArm",
            "b_RightArm",
            "b_LeftLegUpper",
            "b_RightLegUpper",
            
            "b_RightFinger",
            "b__LeftFinger",
            
            "b_LeftAnkle",
            "b_RightAnkle",                        
                                    
        ],
        scale=[0.007,0.007,0.007]
    )

    # 定义CSV表头
    csv_header = [
        'root_pos_x', 'root_pos_y', 'root_pos_z',
        #quaternion (w-x-y-z convention)
        'root_rot_w', 'root_rot_x', 'root_rot_y', 'root_rot_z',
        'linear_velocity_x', 'linear_velocity_y', 'linear_velocity_z',
        'angular_velocity_x', 'angular_velocity_y', 'angular_velocity_z',
        'qpos_0', 'qpos_1', 'qpos_2', 'qpos_3', 'qpos_4', 'qpos_5', 'qpos_6', 'qpos_7', 'qpos_8', 'qpos_9', 'qpos_10', 'qpos_11', 
        'joint_angular_velocity_0', 'joint_angular_velocity_1', 'joint_angular_velocity_2', 'joint_angular_velocity_3', 'joint_angular_velocity_4', 'joint_angular_velocity_5', 'joint_angular_velocity_6', 'joint_angular_velocity_7', 'joint_angular_velocity_8', 'joint_angular_velocity_9', 'joint_angular_velocity_10', 'joint_angular_velocity_11',
        'FL_foot_x','FL_foot_y','FL_foot_z', 'FR_foot_x','FR_foot_y','FR_foot_z', 'RL_foot_x','RL_foot_y','RL_foot_z', 'RR_foot_x','RR_foot_y','RR_foot_z'
    ]

    root = tk.Tk()
    ui = TkinterUI(root, motion_retarget, csv_header)

    # 启动数据播放线程
    data_thread = threading.Thread(target=motion_retarget.play_data, args=())
    data_thread.daemon = True  # 设置为守护线程，确保主线程结束时自动结束
    data_thread.start()

    root.mainloop()