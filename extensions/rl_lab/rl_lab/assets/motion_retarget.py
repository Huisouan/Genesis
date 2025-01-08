import argparse
import os
import pandas as pd
import numpy as np

import genesis as gs
import tkinter as tk
from tkinter import filedialog, messagebox
import threading

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
                    'header': header,
                    'filename':filename,
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
        self.add_axis()
        self.add_points(self.csv_data_list[0])
        self.scene.build()
        self.scene.reset()
    def run(self):
        self.load_csv_files_from_folder()
        parser = argparse.ArgumentParser()
        parser.add_argument("-v", "--vis", action="store_true", default=True)
        args = parser.parse_args()

        self.initialize_scene(args.vis)

        target_quat = np.array([1, 0, 0, 0])

        for csv_data in self.csv_data_list:
            for row in range(0, csv_data['data'].shape[0]):
                frame = csv_data['data'].iloc[row]
                Base_qpos = self.calculate_baseqps()

                for point_info in self.point_list:
                    point_name = point_info['name']
                    x_col = f"{point_name}.X"
                    y_col = f"{point_name}.Y"
                    z_col = f"{point_name}.Z"
                    
                    if x_col in frame and y_col in frame and z_col in frame:
                        position = np.array([frame[x_col], frame[z_col], frame[y_col]])
                        point_info['point'].set_pos(position)

                self.scene.step()

class TkinterUI:
    def __init__(self, motion_retarget: MotionRetarget):
        self.motion_retarget = motion_retarget
        self.root = tk.Tk()
        self.root.title("Motion Retargeting UI")
        self.root.geometry("400x200")

        self.dataset_var = tk.StringVar()
        self.frame_var = tk.IntVar(value=0)

        self.create_widgets()
        self.root.mainloop()

    def create_widgets(self):
        # Dataset selection
        tk.Label(self.root, text="Select Dataset:").pack(pady=5)
        tk.Button(self.root, text="Browse", command=self.browse_dataset).pack(pady=5)

        # Progress bar
        tk.Label(self.root, text="Frame:").pack(pady=5)
        self.progress = tk.Scale(self.root, from_=0, to=0, orient=tk.HORIZONTAL, variable=self.frame_var, command=self.update_frame)
        self.progress.pack(pady=5)

        # Play button
        self.play_button = tk.Button(self.root, text="Play", command=self.play_animation)
        self.play_button.pack(pady=5)

    def browse_dataset(self):
        folder_path = filedialog.askdirectory()
        if folder_path:
            self.motion_retarget.mocap_file = folder_path
            self.motion_retarget.load_csv_files_from_folder()
            if self.motion_retarget.csv_data_list:
                self.progress.config(to=self.motion_retarget.csv_data_list[0]['data'].shape[0] - 1)
                self.frame_var.set(0)
                messagebox.showinfo("Success", "Dataset loaded successfully.")
            else:
                messagebox.showerror("Error", "No CSV files found in the selected folder.")

    def update_frame(self, value):
        self.frame_var.set(int(value))
        self.update_scene(int(value))

    def update_scene(self, frame):
        if self.motion_retarget.csv_data_list:
            csv_data = self.motion_retarget.csv_data_list[0]
            frame_data = csv_data['data'].iloc[frame]
            Base_qpos = self.motion_retarget.calculate_baseqps()

            for point_info in self.motion_retarget.point_list:
                point_name = point_info['name']
                x_col = f"{point_name}.X"
                y_col = f"{point_name}.Y"
                z_col = f"{point_name}.Z"
                
                if x_col in frame_data and y_col in frame_data and z_col in frame_data:
                    position = np.array([frame_data[x_col], frame_data[z_col], frame_data[y_col]])
                    point_info['point'].set_pos(position)

            self.motion_retarget.scene.step()

    def play_animation(self):
        def play():
            for frame in range(self.frame_var.get(), self.progress.cget('to') + 1):
                self.update_frame(frame)
                self.root.update()
                if not self.play_button.cget('text') == "Stop":
                    break

        if self.play_button.cget('text') == "Play":
            self.play_button.config(text="Stop")
            threading.Thread(target=play).start()
        else:
            self.play_button.config(text="Play")

if __name__ == "__main__":
    motion_retarget = MotionRetarget(
        mocap_file="datasets/lssp_keypoints",
        urdf_file="urdf/shadow_hand/shadow_hand.urdf",
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
            "Bip01_HeadNub",
            "Bip01_HeadNubEnd",
            "Bip01_Footsteps",
            "Bip01_FootstepsEnd"
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
        ],
        scale=0.008
    )
    
    ui = TkinterUI(motion_retarget)