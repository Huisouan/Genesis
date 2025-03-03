import argparse
import os
import numpy as np
import genesis as gs
import tkinter as tk
from tkinter import ttk
import threading
import torch
from easydict import EasyDict
import glob

class NPYPlayer:
    def __init__(self, urdf_file, npy_files, fps=120):
        self.urdf_file = urdf_file
        self.npy_files = npy_files
        self.current_file_index = 0
        self.fps = fps
        self.data_list = []
        self.scene = None
        self.robot = None
        # 按文件名排序存储完整路径
        self.npy_files = sorted(npy_files, key=lambda x: os.path.basename(x))
        # 存储文件名列表用于显示
        self.file_names = [os.path.basename(f) for f in self.npy_files]        
        
        # 确保data_list按排序后的顺序加载
        self.data_list = []
        for file_path in self.npy_files:  # 遍历已排序的路径
            data = EasyDict(torch.load(file_path))
            self.data_list.append(data)
        # 删除原有的load_all_npy_data调用        
        
        
        
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
        
        self.load_all_npy_data()
        self.initialize_scene()

    def load_all_npy_data(self):
        """加载所有npy文件的数据，如果输出文件夹有同名文件则跳过"""
        for file_path in self.npy_files:
            # 获取文件名和基础名称
            file_name = os.path.basename(file_path)
            base_name = os.path.splitext(file_name)[0]
            
            # 检查目标文件夹中的clipped目录
            clipped_folder = os.path.join(os.path.dirname(file_path), "clipped")
            if os.path.exists(clipped_folder):
                # 检查是否存在同名文件或带_clip后缀的文件
                clipped_files = glob.glob(os.path.join(clipped_folder, f"{base_name}_clip*.npy"))
                if clipped_files:
                    print(f"跳过已处理文件: {file_name}")
                    continue  # 如果存在同名文件，跳过加载
            
            # 加载数据
            try:
                data = EasyDict(torch.load(file_path))
                self.data_list.append(data)
                print(f"成功加载文件: {file_name}")
            except Exception as e:
                print(f"加载文件失败: {file_name}, 错误: {e}")

    def switch_file(self, index):
        """切换到指定的npy文件"""
        if 0 <= index < len(self.npy_files):
            # 确保索引有效性
            self.current_file_index = index
            self.data = self.data_list[index]
            self.end_frame = self.data['dof_pos'].shape[0]  # 必须立即更新关键值
            
            # 强制重置播放状态
            self.frame_in_play = 0
            self.start_frame = 0
            self.record_stack['motion_data'] = []
            

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
        self.scene.add_entity(gs.morphs.Plane())        
        # 加载机器人URDF模型
        self.robot = self.scene.add_entity(
            gs.morphs.URDF(
                file=self.urdf_file,
                pos=(0, 0, 0.4),
                merge_fixed_links=True,
                fixed=True
            ),
        )
        
        self.scene.build()
        self.scene.reset()
        self.data = self.data_list[self.current_file_index]
        self.end_frame = self.data['dof_pos'].shape[0]

    def play_frame(self, frame_idx):
        """应用机器人状态到仿真环境"""

        # 设置根关节状态
        self.robot.set_pos(self.data['global_translation'][frame_idx,0,:])
        self.robot.set_quat(self.data['global_rotation'][frame_idx,0,:])
        qp = self.robot.get_qpos()
        # 设置关节位置
        self.robot.set_qpos(self.data['dof_pos'][frame_idx,:])
        
    def play_data(self):
        """数据播放主循环"""
        while True:
            if self.frame_in_play >= self.end_frame:
                self.frame_in_play = self.start_frame

            self.play_frame(self.frame_in_play)
            self.scene.step()
            if self.play:
                self.frame_in_play += 2

    def get_total_frames(self):
        """获取当前文件的总帧数"""
        return self.end_frame

class TkinterUI:
    def __init__(self, master, player: NPYPlayer):
        self.master = master
        self.player = player
        
        # 创建带滚动条的框架容器
        file_frame = ttk.Frame(master)
        file_frame.pack(fill='x', padx=5, pady=5)

        # 按文件名排序（不包含路径）
        sorted_files = sorted(player.npy_files, key=lambda x: os.path.basename(x))
        
        # 使用Combobox代替OptionMenu
        self.file_var = tk.StringVar()
        # 使用已排序的文件名列表
        self.file_combobox = ttk.Combobox(
            file_frame,
            textvariable=self.file_var,
            values=player.file_names,  # 直接使用预先生成的文件名列表
            state="readonly"
        )
        # 其他初始化代码保持不变...

        self.file_combobox.pack(side='left', fill='x', expand=True)

        # 添加滚动条
        scrollbar = ttk.Scrollbar(file_frame, orient='horizontal', command=self.file_combobox.xview)
        scrollbar.pack(side='right', fill='x')
        self.file_combobox.configure(xscrollcommand=scrollbar.set)

        # 设置默认选择并绑定事件
        self.file_combobox.current(0)
        self.file_combobox.bind("<<ComboboxSelected>>", self.select_file_combobox)


        # 显示当前文件的总帧数
        self.total_frames_label = ttk.Label(master, text=f"总帧数: {self.player.get_total_frames()}")
        self.total_frames_label.pack()
        
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
        self.end_frame_entry.insert(0, str(self.player.get_total_frames()))  # 设置结束帧的默认值为总帧数
        self.end_frame_entry.pack()
        
        # 控制按钮
        self.play_button = ttk.Button(master, text="播放", command=self.toggle_play)
        self.play_button.pack()
        
        ttk.Button(master, text="设置范围", command=self.set_range).pack()
        ttk.Button(master, text="导出数据", command=self.export_data).pack()
        
        self.update_progress()

    def select_file_combobox(self, event):
        selected_index = self.file_combobox.current()
        self.player.switch_file(selected_index)
        
        # 使用after方法确保在主线程执行UI更新
        self.master.after(0, self._update_ui_after_switch)

        
    def _update_ui_after_switch(self):
        self.total_frames_label.config(text=f"总帧数: {self.player.get_total_frames()}")
        self.end_frame_entry.delete(0, tk.END)
        self.end_frame_entry.insert(0, str(self.player.get_total_frames()))
        self.progress_var.set(0)
        self.frame_label.config(text="Frame: 0")
    def toggle_play(self):
        self.player.play = not self.player.play
        self.play_button.config(text="暂停" if self.player.play else "播放")

    def set_range(self):
        try:
            start = int(self.start_frame_entry.get())
            end = int(self.end_frame_entry.get())
            if 0 <= start < end <= self.player.data['dof_pos'].shape[0]:
                self.player.start_frame = start
                self.player.end_frame = end
                self.player.frame_in_play = start
        except ValueError:
            pass

    def export_data(self):
        """导出剪辑后的机器人数据"""
        start = self.player.start_frame
        end = self.player.end_frame
        
        # 创建一个新的记录栈
        self.player.record_stack = {
            'fps': self.player.fps
        }
        
        for key, value in self.player.data.items():
            if isinstance(value, torch.Tensor):
                # 根据第0维度进行切片
                self.player.record_stack[key] = value[start:end]
            else:
                # 保留原有状态
                self.player.record_stack[key] = value
        
        # 获取源文件名并生成目标文件名
        source_file = self.player.npy_files[self.player.current_file_index]
        base_name = os.path.splitext(os.path.basename(source_file))[0]
        target_folder = os.path.dirname(source_file)
        
        # 创建clipped目录
        clipped_folder = os.path.join(target_folder, "clipped")
        if not os.path.exists(clipped_folder):
            os.makedirs(clipped_folder)
        
        target_file_name = f"{base_name}_clip.npy"
        target_file_path = os.path.join(clipped_folder, target_file_name)
        
        # 检查目标文件夹中的文件，如果有相同名称的文件则增加序号
        if os.path.exists(target_file_path):
            existing_files = glob.glob(os.path.join(clipped_folder, f"{base_name}_clip_*.npy"))
            if existing_files:
                max_index = max(int(os.path.splitext(os.path.basename(f))[0].split('_')[-1]) for f in existing_files)
                target_file_name = f"{base_name}_clip_{max_index + 1}.npy"
            else:
                target_file_name = f"{base_name}_clip_1.npy"
            target_file_path = os.path.join(clipped_folder, target_file_name)
        
        torch.save(self.player.record_stack, target_file_path)
        print(f"数据已导出到 {target_file_path}")

    def update_progress(self):
        """更新进度显示"""
        total = self.player.end_frame - self.player.start_frame
        curr = self.player.frame_in_play - self.player.start_frame
        self.progress_var.set(int(curr/total*100))
        self.frame_label.config(text=f"Frame: {self.player.frame_in_play}")
        self.master.update_idletasks()  # 强制立即更新UI
        self.master.after(200, self.update_progress)

if __name__ == "__main__":
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