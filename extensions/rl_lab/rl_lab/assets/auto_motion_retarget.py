import os
import pandas as pd
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from scipy.signal import butter, lfilter
from rl_lab.assets.transformations import quaternion_from_matrix
import genesis as gs

INIT_ROT = np.array([0, 0, 0, 1.0])

class AutoMotionRetarget:
    def __init__(self, mocap_dir, urdf_file, output_dir, point_names, joint_names, key_points, scale, mirror=False):
        self.mocap_dir = mocap_dir
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.urdf_file = urdf_file
        self.output_dir = output_dir
        self.point_names = point_names
        self.joint_names = joint_names
        self.key_points = key_points
        self.scale = scale
        self.mirror = mirror  # 新增镜像参数
        
        # 初始化核心数据结构
        self.csv_data_list = []
        self.record_stack = {
            'fps': 120,
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
        
        # 初始化物理场景
        self._init_scene()

    def _init_scene(self):
        """初始化无头模式物理场景"""
        gs.init(seed=0, precision="32", logging_level="error")
        self.scene = gs.Scene(
            show_viewer=False,  # 禁用可视化
            rigid_options=gs.options.RigidOptions(
                gravity=(0, 0, 0),
                enable_collision=False,
                enable_joint_limit=False,
            ),
        )
        self.robot = self.scene.add_entity(
            gs.morphs.URDF(
                file=self.urdf_file,
                pos=(0, 0, 0.4),
                merge_fixed_links=True,
                links_to_keep=['FL_calf_rotor', 'FR_calf_rotor', 'RL_calf_rotor', 'RR_calf_rotor',
                              'FL_foot', 'FR_foot', 'RL_foot', 'RR_foot'],
                fixed=True
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
        self.motion_link_idxs = []  
        
        self.scene.build()
        self.scene.reset()
        self.motion_link_idxs = []
        for link in self.robot._links:
            if link.name not in [
                #'FL_foot', 'FR_foot', 'RL_foot', 'RR_foot', 
                'FL_calf_rotor', 'FR_calf_rotor', 'RL_calf_rotor', 'RR_calf_rotor'
                                 ]:
                self.motion_link_idxs.append(link.idx_local)  
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
        self.motor_dofs = [self.robot.get_joint(name).dof_idx_local for name in self.dof_names]
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


    def batch_process(self):
        """批量处理所有运动文件"""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # 获取所有CSV文件
        csv_files = [f for f in os.listdir(self.mocap_dir) if f.endswith('.csv')]
        
        for filename in csv_files:
            print(f"Processing {filename}...")

            # 加载并处理单个文件
            self._process_single_file(filename)
            # 应用数据平滑
            self._apply_smoothing()
            # 导出数据
            self._export_data(filename)

            self._reset_recorder()

    def _process_single_file(self, filename):
        """处理单个运动文件"""
        file_path = os.path.join(self.mocap_dir, filename)
        data = self._load_and_preprocess(file_path)
        
        # 初始化记录器
        self.record_stack['fps'] = 120
        
        # 处理所有帧
        for frame_idx in range(len(data)):
            self._process_frame(data, frame_idx)

    def _load_and_preprocess(self, file_path):
        """加载并预处理CSV数据"""
        data = pd.read_csv(file_path)
        
        # 应用缩放
        for point_name in self.point_names:
            x_col = f"{point_name}.X"
            y_col = f"{point_name}.Y"
            z_col = f"{point_name}.Z"
            if x_col in data.columns:
                data[x_col] *= self.scale[0]
                if self.mirror:  # 根据镜像参数调整x轴缩放
                    data[x_col] *= -1
                data[y_col] *= self.scale[1]
                data[z_col] *= self.scale[2]
        
        # 数据平滑处理
        window_size = 5
        data_smoothed = data.rolling(window=window_size, center=True).mean()
        return data_smoothed.fillna(method='bfill').fillna(method='ffill')


    def _process_frame(self, data, frame_idx):
        """处理单帧数据"""
        curr = self.IK(data.iloc[frame_idx])
        
        # 记录数据
        if frame_idx > 0:
            prev_curr = self.IK(data.iloc[frame_idx-1])
            self._record_frame(prev_curr, curr, 1/120)

        # 物理模拟步进
        self.scene.step()

    def _record_frame(self, prev_curr, curr, dt):
        """记录帧数据"""
        data_to_record = self.collect_data(prev_curr, curr, dt)
        for key in self.record_stack:
            if isinstance(self.record_stack[key], list):
                self.record_stack[key].append(data_to_record[key])

    def _apply_smoothing(self):
        """应用数据平滑"""
        dof_vels = np.array(self.record_stack['dof_vels'])
        smoothed = np.zeros_like(dof_vels)
        
        # 低通滤波参数
        fs = 120
        cutoff = 50
        order = 5
        
        for joint_idx in range(dof_vels.shape[1]):
            smoothed[:, joint_idx] = butter_lowpass_filter(
                dof_vels[:, joint_idx], cutoff, fs, order)
        
        self.record_stack['dof_vels'] = smoothed.tolist()

    def _export_data(self, filename):
        """导出处理后的数据"""
        base_name = os.path.splitext(filename)[0]
        if self.mirror:  # 根据镜像参数调整导出文件名
            base_name += "_mirror"
        output_path = os.path.join(self.output_dir, f"{base_name}_processed.npy")
        
        export_data = {}
        for k, v in self.record_stack.items():
            if isinstance(v, list):
                export_data[k] = torch.tensor(v) if v else torch.empty(0)
            else:
                export_data[k] = v
        
        torch.save(export_data, output_path)

    def _reset_recorder(self):
        """重置记录器"""
        for key in self.record_stack:
            if isinstance(self.record_stack[key], list):
                self.record_stack[key] = []

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

        root_vel = linear_velocity[0]
        root_angular_vel = angular_velocity[0]
        
        
        data = {
            'global_root_velocity': torch.tensor(root_vel).cpu().tolist(),
            'global_root_angular_velocity': torch.tensor(root_angular_vel).cpu().tolist(),

            'global_translation': curr['global_translation'].tolist(),
            'global_rotation': curr['global_rotation'].tolist(),
            'local_rotation': curr['local_rotation'].tolist(),

            'global_velocity': _linear_velocity.tolist(),
            'global_angular_velocity': _angular_velocity.tolist(),
                        
            'dof_pos': curr['dof_pos'].tolist(),
            'dof_vels': _joint_angular_velocities.tolist(),
        }

        return data


    def calculate_root_state(self,frame):
        """
        根据参考关节位置计算当前帧的根节点位置和旋转。
        
        返回:
        - root_pos (np.ndarray): 计算得到的当前帧的根节点位置。
        - root_rot (np.ndarray): 当前帧的根节点的四元数旋转表示。
        """

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

        if self.mirror:
            # 镜像处理
            left_shoulder_pos = key_point_positions["b_RightArm"]
            right_shoulder_pos = key_point_positions["b_LeftArm"]
            left_hip_pos = key_point_positions["b_RightLegUpper"]
            right_hip_pos = key_point_positions["b_LeftLegUpper"]        


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
                    
                    self.point_list.append({'name': point_name})

    def play_frame(self,csv_data,row):
        frame = csv_data['data'].iloc[row]
        for point_info in self.point_list:
            point_name = point_info['name']
            x_col = f"{point_name}.X"
            y_col = f"{point_name}.Y"
            z_col = f"{point_name}.Z"
            
            if x_col in frame and y_col in frame and z_col in frame:
                position = np.array([frame[x_col], frame[z_col], frame[y_col]])

    def get_keypoints(self,frame,key_point_names):
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
        return key_point_positions

    def IK(self,frame):
        root_pos, root_rot = self.calculate_root_state(frame)

        self.robot.set_pos(root_pos)
        self.robot.set_quat(root_rot)
        #self.scene.step()   
        
        pos = self.robot.get_links_pos()
        quat = self.robot.get_links_quat()
        
        key_points = [
            "b_RightAnkle",
            "b_LeftAnkle",
            "b_RightFinger",
            "b__LeftFinger",
            
            "b_RightLegUpper",
            "b_LeftLegUpper",
            "b_RightArm",
            "b_LeftArm",
        ]
        
        
        
        key_point_positions = self.get_keypoints(frame,key_points )
        
        RLF_target = key_point_positions["b_RightAnkle"]
        RRF_target = key_point_positions["b_LeftAnkle"]
        FLF_target = key_point_positions["b_RightFinger"]
        FRF_target = key_point_positions["b__LeftFinger"]
        
        RLS_target = key_point_positions["b_RightLegUpper"]
        RRS_target = key_point_positions["b_LeftLegUpper"]
        FLS_target = key_point_positions["b_RightArm"]
        FRS_target = key_point_positions["b_LeftArm"]
        
        if self.mirror:
            RLF_target = key_point_positions["b_LeftAnkle"]
            RRF_target = key_point_positions["b_RightAnkle"]
            FLF_target = key_point_positions["b__LeftFinger"]
            FRF_target = key_point_positions["b_RightFinger"]
            
            RLS_target = key_point_positions["b_LeftLegUpper"]
            RRS_target = key_point_positions["b_RightLegUpper"]
            FLS_target = key_point_positions["b_LeftArm"]
            FRS_target = key_point_positions["b_RightArm"]            
        
        
        
        
        # 假设FRF_target, FRS_target, FLF_target, FLS_target, RRF_target, RRS_target, RLF_target, RLS_target都是numpy数组
        delta_FR = torch.tensor(FRF_target - FRS_target).to(self.device)
        delta_FL = torch.tensor(FLF_target - FLS_target).to(self.device)
        delta_RR = torch.tensor(RRF_target - RRS_target).to(self.device)
        delta_RL = torch.tensor(RLF_target - RLS_target).to(self.device)      
                
        
        FLF_target_t = delta_FL + pos[self.FL_sholder.idx_local]
        FRS_target_t = delta_FR + pos[self.FR_sholder.idx_local]
        RLF_target_t = delta_RL + pos[self.RL_sholder.idx_local]
        RRS_target_t = delta_RR + pos[self.RR_sholder.idx_local]
                 

        qpos = self.robot.inverse_kinematics_multilink(
            links=[self.FL_link, self.FR_link, self.RL_link, self.RR_link],  # 指定需要控制的链接
            poss=[FLF_target_t, FRS_target_t, RLF_target_t, RRS_target_t], 
            max_samples=25,
            max_solver_iters=10,# 指定目标位置
        ) 
        self.robot.set_qpos(qpos)
        
        global_translation = self.robot.get_links_pos()
        global_rotation = self.robot.get_links_quat()
        dof_pos = self.robot.get_dofs_position()
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

def butter_lowpass_filter(data, cutoff, fs, order=5):
    """低通滤波器"""
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return lfilter(b, a, data)

    
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


if __name__ == "__main__":
    # 配置参数
    CONFIG = {
        "mocap_dir": "datasets/lssp_keypoints",
        "urdf_file": "datasets/go2_description/urdf/go2_description.urdf",
        "output_dir": "processed_motions",
        "point_names": [
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
            ],  # 保留原有关键点配置
        "joint_names": [
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
            ],  # 保留原有关节配置
        "key_points": [
            
            
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
            ],   # 保留原有关键点
        "scale": [0.007, 0.007, 0.007],
        "mirror": True,  # 新增镜像参数
    }

    processor = AutoMotionRetarget(**CONFIG)
    processor.batch_process()
    print("Batch processing completed!")
