import argparse
import os
import pandas as pd
import numpy as np

import genesis as gs
import tkinter as tk
from tkinter import ttk
import threading
import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as R


class MotionRetarget:
    def __init__(self, mocap_file, urdf_file, name_list, point_names, joint_names, key_points, scale=1):
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
                
                data = data * self.scale
                
                self.csv_data_list.append({
                    'data': data,
                    'length': data.shape[0],  # 添加数据长度
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
                    scale=0.05,
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
            ),
        )
        self.FL_link = self.robot.get_link("FL_calf")
        self.FR_link = self.robot.get_link("FR_calf")
        self.RL_link = self.robot.get_link('RL_calf')
        self.RR_link = self.robot.get_link('RR_calf')
        
        
        self.add_axis()
        self.add_points(self.csv_data_list[0])
        self.scene.build()
        self.scene.reset()

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
        self.scene.step()

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
        while True:
            if self.frame_in_play >= self.csv_data_list[self.data_in_play]['length'] or self.frame_in_play >= self.end_frame :
                #循环播放一个文件
                self.frame_in_play = self.start_frame
            self.play_frame(self.csv_data_list[self.data_in_play],self.frame_in_play)
            if self.play ==  True:
                self.frame_in_play += 1
            elif self.back == True:
                self.frame_in_play -= 1 


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

        # 获取关键点位置
        pelvis_pos = np.array([frame[f"Bip01.X"], frame[f"Bip01.Y"], frame[f"Bip01.Z"]], dtype=np.float64)
        neck_pos = np.array([frame[f"b__Neck.X"], frame[f"b__Neck.Y"], frame[f"b__Neck.Z"]], dtype=np.float64)
        left_shoulder_pos = np.array([frame[f"b_LeftArm.X"], frame[f"b_LeftArm.Y"], frame[f"b_LeftArm.Z"]], dtype=np.float64)
        right_shoulder_pos = np.array([frame[f"b_RightArm.X"], frame[f"b_RightArm.Y"], frame[f"b_RightArm.Z"]], dtype=np.float64)
        left_hip_pos = np.array([frame[f"b_LeftLegUpper.X"], frame[f"b_LeftLegUpper.Y"], frame[f"b_LeftLegUpper.Z"]], dtype=np.float64)
        right_hip_pos = np.array([frame[f"b_RightLegUpper.X"], frame[f"b_RightLegUpper.Y"], frame[f"b_RightLegUpper.Z"]], dtype=np.float64)

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
        up_dir = np.cross(forward_dir, left_dir)
        up_dir = up_dir / np.linalg.norm(up_dir)
        left_dir = np.cross(up_dir, forward_dir)
        left_dir = left_dir / np.linalg.norm(left_dir)

        # 构造旋转矩阵并转换为四元数
        rot_mat = np.column_stack([up_dir, forward_dir, left_dir])
        rot = R.from_matrix(rot_mat)
        quat1 = rot.as_quat()
        y_quat = R.from_euler('y', -90, degrees=True).as_quat()
        quat2 = (R.from_quat(y_quat) * R.from_quat(quat1)).as_quat()
        x_quat = R.from_euler('x', 90, degrees=True).as_quat()
        root_rot = (R.from_quat(x_quat) * R.from_quat(quat2)).as_quat()

        # 归一化四元数
        root_rot = root_rot / np.linalg.norm(root_rot)

        # 计算根节点位置
        root_pos = 0.25 * (left_shoulder_pos + right_shoulder_pos + left_hip_pos + right_hip_pos)

        return root_pos, root_rot

    def IK(self):
        
        self.motion_retarget.IK()


class TkinterUI:
    def __init__(self, master, motion_retarget: MotionRetarget):
        self.master = master  # 设置主窗口
        self.motion_retarget = motion_retarget  # 设置 MotionRetarget 实例
        self.current_data_index = None  # 当前选中的数据集索引
        self.current_frame = 0  # 当前帧索引
        self.is_playing = False  # 播放状态标志

        self.master.title("Motion Retarget UI")  # 设置窗口标题

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

        self.end_frame_entry = ttk.Entry(master)  # 创建结束帧输入框
        self.end_frame_entry.pack()  # 将结束帧输入框添加到窗口
        self.end_frame_entry.insert(0, "100")  # 设置结束帧输入框的默认值为 100

        self.frame_entry = ttk.Entry(master)  # 创建手动输入帧号的输入框
        self.frame_entry.pack()  # 将输入框添加到窗口

        self.jump_button = ttk.Button(master, text="Jump to Frame", command=self.jump_to_frame)  # 创建跳转按钮
        self.jump_button.pack()  # 将跳转按钮添加到窗口

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
            
            #"b_LeftToe002",
            #"b_RightToe002",        
            #"b_LeftToe",
            #"b_RightToe",
            "b_LeftAnkle",
            "b_RightAnkle",                        
                                    
        ],
        scale=0.008
    )

    root = tk.Tk()
    ui = TkinterUI(root, motion_retarget)

    # 启动数据播放线程
    data_thread = threading.Thread(target=motion_retarget.play_data, args=())
    data_thread.daemon = True  # 设置为守护线程，确保主线程结束时自动结束
    data_thread.start()

    root.mainloop()