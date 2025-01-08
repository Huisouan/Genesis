import argparse
import os
import pandas as pd
import numpy as np

import genesis as gs
mocap_file = "datasets/lssp_keypoints"
urdf_file = "urdf/shadow_hand/shadow_hand.urdf"
name_list = [
    "Base",
    "FR",
    "FL",
    "RR",
    "RL",
]
point_names = [
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
]



joint_names = [
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

key_points = [
    "Bip01",
    "b__Neck",
    "b_LeftArm",
    "b_RightArm",
    "b_LeftLegUpper",
    "b_RightLegUpper",
    
]
def load_csv_files_from_folder(folder_path, name_list=None, scale=1):
    """
    从指定文件夹中读取所有的 CSV 文件，并提取表头信息。
    如果提供了 name_list，则只保留表头中包含 name_list 中名称的部分，其他部分丢弃。
    所有数据将乘以 scale 参数。

    参数
    ----------
    folder_path : str
        包含 CSV 文件的文件夹路径。
    name_list : list of str, optional
        要保留的表头名称列表。如果为 None，则保留所有表头信息。默认为 None。
    scale : float, optional
        缩放因子。默认为 1。

    返回值
    -------
    csv_data_list : list of dict
        包含每个 CSV 文件数据和表头信息的列表。每个字典包含两个键：
        - 'filename': CSV 文件的文件名。
        - 'data': pandas DataFrame 包含 CSV 文件的数据。
        - 'header': list 包含 CSV 文件的表头信息。
    """
    csv_data_list = []

    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)
            # 读取 CSV 文件
            data = pd.read_csv(file_path)
            # 提取表头信息
            header = data.columns.tolist()
            
            if name_list is not None:
                # 过滤表头，只保留包含 name_list 中名称的部分
                filtered_header = [col for col in header if any(name in col for name in name_list)]
                # 过滤数据，只保留包含 name_list 中名称的部分
                data = data[filtered_header]
                header = filtered_header
            
            # 将数据乘以 scale
            data = data * scale
            
            # 将数据和表头信息存储在字典中
            csv_data_list.append({
                'data': data,
                'header': header
            })

    return csv_data_list

def claculate_baseqps():
    
    pass

def add_axis(scene, name_list):
    """
    在场景中添加多个轴对象，并将它们存储在字典中。

    参数
    ----------
    scene : gs.Scene
        要添加实体的场景对象。
    name_list : list of str
        轴对象的名称列表。
    target_dict : dict
        用于存储添加的轴对象的字典，键为名称，值为实体对象。
    """
    target_dict = {}
    for name in name_list:
        target_name = name + "_target"
        target = scene.add_entity(
            gs.morphs.Mesh(
                file="meshes/axis.obj",
                scale=0.05,
            ),
            surface=gs.surfaces.Default(color=(1, 0.5, 0.5, 1)),
        )
        target_dict[target_name] = target
    return target_dict

def add_points(scene: gs.Scene, csv_info, key_point_list=None):
    """
    根据 CSV 数据在场景中添加点对象，并返回一个包含点名称和对应点对象的列表。

    参数
    ----------
    scene : gs.Scene
        要添加实体的场景对象。
    csv_info : dict
        包含 CSV 文件数据和表头信息的字典。包含两个键：
        - 'data': pandas DataFrame 包含 CSV 文件的数据。
        - 'header': list 包含 CSV 文件的表头信息。
    key_point_list : list of str, optional
        要设置为红色的点名称列表。默认为 None。

    返回值
    -------
    point_list : list of dict
        包含点名称和对应点对象的列表。每个字典包含两个键：
        - 'name': 点的名称。
        - 'point': 点对象。
    """
    data = csv_info['data']
    header = csv_info['header']
    
    # 提取 X, Y, Z 列名
    columns = [col for col in header if col.endswith('.X') or col.endswith('.Y') or col.endswith('.Z')]
    unique_points = set()
    point_list = []
    
    for index, row in data.iterrows():
        for i in range(0, len(columns), 3):
            x_col = columns[i]
            
            # 提取点的名称，假设名称是列名去掉 .X, .Y, .Z 后的部分
            point_name = x_col[:-2]  # 去掉 .X
            
            if point_name not in unique_points:
                unique_points.add(point_name)
                
                # 判断点的颜色
                if key_point_list is not None and point_name in key_point_list:
                    color = (0.8, 0.8, 0.8, 1)  #
                else:
                    color = (0.5, 0.5, 0.5, 1)  # 默认颜色
                
                point = scene.add_entity(
                    gs.morphs.Sphere(
                        radius=0.01,  # 点的半径
                    ),
                    surface=gs.surfaces.Default(color=color),  # 点的颜色
                )
                point_list.append({'name': point_name, 'point': point})
    
    return point_list

def main():
    csv_data_list = load_csv_files_from_folder(mocap_file,point_names,0.008)

    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=True)
    args = parser.parse_args()

    ########################## init ##########################
    gs.init(seed=0, precision="32", logging_level="debug")

    ########################## create a scene ##########################
    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(2.5, 0.0, 1.5),
            camera_lookat=(0.0, 0.0, 0.5),
            camera_fov=40,
        ),
        show_viewer=args.vis,
        rigid_options=gs.options.RigidOptions(
            gravity=(0, 0, 0),
            enable_collision=False,
            enable_joint_limit=False,
        ),
    )
    ########################## targets ##########################
    target_dict = add_axis(scene, name_list)
    point_list = add_points(scene, csv_data_list[0],key_points)
    ########################## entities ##########################
    scene.add_entity(
        gs.morphs.Plane(),
    )
    robot = scene.add_entity(
        gs.morphs.URDF(
            file="urdf/go2/urdf/go2.urdf",
            pos=(0, 0, 0.4),
        ),
    )

    ########################## build ##########################
    scene.build()
    scene.reset()

    target_quat = np.array([1, 0, 0, 0])
    '''
    index_finger_distal = robot.get_link("index_finger_distal")

    dofs_idx_local = []
    for v in robot.joints:
        if v.name in [
            "wrist_joint",
            "index_finger_joint1",
            "index_finger_join2",
            "index_finger_joint3",
        ]:
            dof_idx_local_v = v.dof_idx_local
            if isinstance(dof_idx_local_v, list):
                dofs_idx_local.extend(dof_idx_local_v)
            else:
                assert isinstance(dof_idx_local_v, int)
                dofs_idx_local.append(dof_idx_local_v)

    center = np.array([0.033, -0.01, 0.42])
    r1 = 0.05
    '''

    for csv_data in csv_data_list:
        for row in range(0, csv_data['data'].shape[0]):
            frame = csv_data['data'].iloc[row]
            Base_qpos = claculate_baseqps()

            # 更新点的位置
            for point_info in point_list:
                point_name = point_info['name']
                x_col = f"{point_name}.X"
                y_col = f"{point_name}.Y"
                z_col = f"{point_name}.Z"
                
                if x_col in frame and y_col in frame and z_col in frame:
                    position = np.array([frame[x_col], frame[z_col],frame[y_col]])
                    point_info['point'].set_pos(position)
                '''
                    target_Base.set_qpos(Base_qpos)

                qpos = robot.inverse_kinematics_multilink(
                    links=[index_finger_distal],  # IK targets
                    poss=[index_finger_pos],
                    dofs_idx_local=dofs_idx_local,  # IK wrt these dofs
                )

                robot.set_qpos(qpos)
                '''
            scene.step()

if __name__ == "__main__":
    main()