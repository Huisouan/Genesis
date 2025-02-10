import argparse
import time

import numpy as np
import torch

import genesis as gs


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=True, help="是否启用可视化")
    parser.add_argument("-c", "--cpu", action="store_true", default=False, help="是否使用CPU进行计算，默认使用GPU")
    args = parser.parse_args()

    ########################## 初始化 ##########################
    # 初始化Genesis库，设置随机种子和后端（CPU或GPU）
    gs.init(seed=0, backend=gs.cpu if args.cpu else gs.gpu)

    ########################## 创建场景 ##########################
    # 创建一个场景对象，并配置视图选项、是否显示查看器、刚体选项和可视化选项
    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(-5.0, -5.0, 10.0),  # 相机位置
            camera_lookat=(5.0, 5.0, 0.0),  # 相机朝向
            camera_fov=40,  # 相机视野角度
        ),
        show_viewer=args.vis,  # 是否显示查看器
        rigid_options=gs.options.RigidOptions(
            dt=0.01,  # 时间步长
            constraint_solver=gs.constraint_solver.Newton,  # 约束求解器类型
        ),
        vis_options=gs.options.VisOptions(
            # geom_type='sdf',  # 可选的几何类型
        ),
    )

    # 定义地形的水平和垂直缩放比例
    horizontal_scale = 0.25
    vertical_scale = 0.005

    # 创建高度场矩阵，初始值为零
    height_field = np.zeros([40, 40])

    # 定义高度范围，并在特定区域内随机生成高度值
    heights_range = np.arange(-10, 20, 10)
    height_field[5:35, 5:35] = 200 + np.random.choice(heights_range, (30, 30))

    ########################## 添加实体 ##########################
    # 向场景中添加地形实体，指定地形的水平和垂直缩放比例以及高度场数据
    terrain = scene.add_entity(
        morph=gs.morphs.Terrain(
            horizontal_scale=horizontal_scale,
            vertical_scale=vertical_scale,
            height_field=height_field,
            name="example",
            # from_stored=True,
        ),
    )

    ########################## 构建场景 ##########################
    # 构建场景，指定环境数量
    scene.build(n_envs=1)

    # 获取地形的高度场数据，并转换为张量
    height_field = terrain.geoms[0].metadata["height_field"]
    rows = horizontal_scale * torch.arange(0, height_field.shape[0], 1, device="cuda").unsqueeze(1).repeat(
        1, height_field.shape[1]
    ).unsqueeze(-1)
    cols = horizontal_scale * torch.arange(0, height_field.shape[1], 1, device="cuda").unsqueeze(0).repeat(
        height_field.shape[0], 1
    ).unsqueeze(-1)
    heights = vertical_scale * torch.tensor(height_field, device="cuda").unsqueeze(-1)

    # 将行列坐标和高度组合成三维坐标点
    poss = torch.cat([rows, cols, heights], dim=-1).reshape(-1, 3)

    # 在场景中绘制调试球体，用于可视化地形点
    scene.draw_debug_spheres(poss=poss, radius=0.05, color=(0, 0, 1, 0.7))

    # 进入主循环，模拟1000个时间步
    for _ in range(1000):
        time.sleep(0.5)  # 每次迭代暂停0.5秒
        scene.step()  # 更新场景状态


if __name__ == "__main__":
    main()