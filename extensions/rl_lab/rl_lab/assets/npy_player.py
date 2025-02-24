import argparse
import os
import numpy as np
import genesis as gs
import tkinter as tk
from tkinter import ttk
import threading
import torch
from easydict import EasyDict


class NPYPlayer:
    def __init__(self, urdf_file, npy_files, fps=120):
        self.urdf_file = urdf_file
        self.npy_files = npy_files
        self.current_file_index = 0
        self.fps = fps
        self.data = None
        self.scene = None
        self.robot = None
        
        # 播放控制参数
        self.frame_in_play = 0
        self.start_frame = 0
        self.end_frame = 0
        self.play = True
        self.record = False
        
        # 数据存储
        self.record_stack = {
            'motion_data': [],
            'fps': self.fps
        }
        
        self.load_npy_data()
        self.initialize_scene()

    def load_npy_data(self):
        """加载包含机器人状态的npy数据"""
        self.data = EasyDict(torch.load(self.npy_files[self.current_file_index]))
        self.end_frame = len(self.data)

    def switch_file(self, index):
        """切换到指定的npy文件"""
        if 0 <= index < len(self.npy_files):
            self.current_file_index = index
            self.load_npy_data()
            self.frame_in_play = 0
            self.start_frame = 0
            self.end_frame = 0
            self.record_stack['motion_data'] = []
            self.record = False
            self.initialize_scene()

    def initialize_scene(self):
        """初始化3D场景并加载机器人模型"""
        gs.init(seed=0, precision="32")
        self.scene = gs.Scene(
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(2.5, 0.0, 1.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=40,
            ),
            show_viewer=True,
            rigid_options=gs.options.RigidOptions(
                gravity=(0, 0, 0),
                enable_collision=False,
            ),
        )
        
        # 加载机器人URDF模型
        self.robot = self.scene.add_entity(
            gs.morphs.URDF(
                file=self.urdf_file,
                pos=(0, 0, 0.4),
                merge_fixed_links=True
            ),
        )
        
        self.scene.build()
        self.scene.reset()

    def play_frame(self, frame_idx):
        """应用机器人状态到仿真环境"""

        
        # 设置根关节状态
        self.robot.set_pos(self.data['global_translation'][frame_idx,0,:])
        self.robot.set_quat(self.data['global_rotation'][frame_idx,0,:])
        qp = self.robot.get_qpos()
        # 设置关节位置
        self.robot.set_dofs_position(self.data['dof_pos'][frame_idx,:])
        
    def play_data(self):
        """数据播放主循环"""
        while True:
            if self.frame_in_play >= self.end_frame:
                self.frame_in_play = self.start_frame

            self.play_frame(self.frame_in_play)
            self.scene.step()
            if self.play:
                self.frame_in_play += 1

class TkinterUI:
    def __init__(self, master, player: NPYPlayer):
        self.master = master
        self.player = player
        
        # 文件选择组件
        self.file_var = tk.StringVar()
        self.file_var.set(self.player.npy_files[0])  # 默认选择第一个文件
        self.file_menu = ttk.OptionMenu(master, self.file_var, *self.player.npy_files, command=self.select_file)
        self.file_menu.pack()
        
        # 播放控制组件
        self.progress_var = tk.IntVar()
        self.progress_bar = ttk.Progressbar(master, variable=self.progress_var, maximum=100)
        self.progress_bar.pack()
        
        self.frame_label = ttk.Label(master, text="Frame: 0")
        self.frame_label.pack()
        
        # 帧范围设置
        ttk.Label(master, text="起始帧").pack()
        self.start_frame_entry = ttk.Entry(master)
        self.start_frame_entry.insert(0, "0")
        self.start_frame_entry.pack()
        
        ttk.Label(master, text="结束帧").pack()
        self.end_frame_entry = ttk.Entry(master) 
        self.end_frame_entry.insert(0, str(len(self.player.data)))
        self.end_frame_entry.pack()
        
        # 控制按钮
        self.play_button = ttk.Button(master, text="播放", command=self.toggle_play)
        self.play_button.pack()
        
        ttk.Button(master, text="设置范围", command=self.set_range).pack()
        ttk.Button(master, text="导出数据", command=self.export_data).pack()
        
        self.update_progress()

    def select_file(self, value):
        """选择要播放的文件"""
        index = self.player.npy_files.index(value)
        self.player.switch_file(index)

    def toggle_play(self):
        self.player.play = not self.player.play
        self.play_button.config(text="暂停" if self.player.play else "播放")

    def set_range(self):
        try:
            start = int(self.start_frame_entry.get())
            end = int(self.end_frame_entry.get())
            if 0 <= start < end <= len(self.player.data):
                self.player.start_frame = start
                self.player.end_frame = end
                self.player.frame_in_play = start
        except ValueError:
            pass

    def export_data(self):
        """导出剪辑后的机器人数据"""
        self.player.record_stack['motion_data'] = self.player.data[self.player.start_frame:self.player.end_frame]
        torch.save(self.player.record_stack, "robot_motion_data_clip.npy")
        print("数据已导出")

    def update_progress(self):
        """更新进度显示"""
        total = self.player.end_frame - self.player.start_frame
        curr = self.player.frame_in_play - self.player.start_frame
        self.progress_var.set(int(curr/total*100))
        self.frame_label.config(text=f"Frame: {self.player.frame_in_play}")
        self.master.after(100, self.update_progress)

if __name__ == "__main__":
    import glob

    urdf_file = "datasets/go2_description/urdf/go2_description.urdf"
    npy_folder = "processed_motions"  # 替换为你的npy文件夹路径
    npy_files = glob.glob(os.path.join(npy_folder, "*.npy"))

    if not npy_files:
        print("没有找到npy文件")
        exit(1)

    player = NPYPlayer(
        urdf_file=urdf_file,
        npy_files=npy_files
    )
    
    root = tk.Tk()
    ui = TkinterUI(root, player)
    
    data_thread = threading.Thread(target=player.play_data)
    data_thread.daemon = True
    data_thread.start()
    
    root.mainloop()