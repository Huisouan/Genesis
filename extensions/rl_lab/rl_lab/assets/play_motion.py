import argparse
import os
import pandas as pd
import numpy as np
import torch
import genesis as gs
from rl_lab.assets.transformations import quaternion_from_matrix
import threading

INIT_ROT = np.array([0, 0, 0, 1.0])

class DataPlayer:
    def __init__(self, csv_file, urdf_file, scale=[1, 1, 1]):
        self.csv_file = csv_file
        self.urdf_file = urdf_file
        self.scale = scale
        self.data = self.load_csv_file()
        self.scene = None
        self.robot = None
        self.frame_in_play = 0
        self.init()

    def load_csv_file(self):
        data = pd.read_csv(self.csv_file)
        return torch.tensor(data.values, dtype=torch.float32)

    def initialize_scene(self, vis):
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
                merge_fixed_links=True,
                links_to_keep=['FL_calf_rotor', 'FR_calf_rotor', 'RL_calf_rotor', 'RR_calf_rotor',
                               'FL_foot', 'FR_foot', 'RL_foot', 'RR_foot'],
                fixed=True
            ),
        )
        self.scene.build()
        self.scene.reset()

    def play_frame(self, row):
        root_pos = row[[0, 1, 2]]
        root_rot = row[[3, 4, 5, 6]]
        qpos = row[[ 13, 14, 15, 16, 17, 18,19,20,21,22,23,24]]

        self.robot.set_pos(root_pos)
        self.robot.set_quat(root_rot)
        self.robot.set_qpos(qpos)

        self.scene.step()

    def play_data(self):
        while True:
            if self.frame_in_play >= self.data.shape[0]:
                self.frame_in_play = 0  # 重新播放

            row = self.data[self.frame_in_play]
            self.play_frame(row)
            self.frame_in_play += 1

    def init(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("-v", "--vis", action="store_true", default=True)
        args = parser.parse_args()
        self.initialize_scene(args.vis)

if __name__ == "__main__":
    data_player = DataPlayer(
        csv_file="extensions/rl_lab/rl_lab/datasets/exported_data.csv",
        urdf_file="datasets/go2_description/urdf/go2_description.urdf",
        scale=[0.007, 0.007, 0.007]
    )
    data_player.play_data()