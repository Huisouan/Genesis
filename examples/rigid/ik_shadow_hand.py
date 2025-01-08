# 导入必要的库和模块
import argparse  # 用于解析命令行参数
import numpy as np  # 用于数值计算
import genesis as gs  # 引入Genesis库，假设这是一个用于物理仿真和图形渲染的库

def main():
    # 创建一个命令行参数解析器，允许用户通过命令行传递参数
    parser = argparse.ArgumentParser()
    # 添加一个可选参数 --vis，默认值为True，表示是否显示查看器
    parser.add_argument("-v", "--vis", action="store_true", default=True)
    # 解析命令行参数
    args = parser.parse_args()

    ########################## 初始化 Genesis 环境 ##########################
    # 初始化Genesis环境，设置随机种子、精度和日志级别
    gs.init(seed=0, precision="32", logging_level="debug")

    ########################## 创建场景 ##########################
    # 创建一个新的场景对象，并配置查看器选项（相机位置、视角等）
    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(2.5, 0.0, 1.5),  # 相机初始位置
            camera_lookat=(0.0, 0.0, 0.5),  # 相机观察点
            camera_fov=40,  # 相机视场角
        ),
        show_viewer=args.vis,  # 是否显示查看器
        rigid_options=gs.options.RigidOptions(
            gravity=(0, 0, 0),  # 重力设置为零
            enable_collision=False,  # 禁用碰撞检测
            enable_joint_limit=False,  # 禁用关节限制
        ),
    )

    # 在场景中添加第一个目标实体（轴对象），并设置其外观属性
    target_1 = scene.add_entity(
        gs.morphs.Mesh(
            file="meshes/axis.obj",  # 轴对象的模型文件路径
            scale=0.05,  # 缩放比例
        ),
        surface=gs.surfaces.Default(color=(1, 0.5, 0.5, 1)),  # 设置颜色为浅红色
    )

    # 在场景中添加第二个目标实体（轴对象），并设置其外观属性
    target_2 = scene.add_entity(
        gs.morphs.Mesh(
            file="meshes/axis.obj",  # 轴对象的模型文件路径
            scale=0.05,  # 缩放比例
        ),
        surface=gs.surfaces.Default(color=(0.5, 1.0, 0.5, 1)),  # 设置颜色为浅绿色
    )

    # 在场景中添加第三个目标实体（轴对象），并设置其外观属性
    target_3 = scene.add_entity(
        gs.morphs.Mesh(
            file="meshes/axis.obj",  # 轴对象的模型文件路径
            scale=0.05,  # 缩放比例
        ),
        surface=gs.surfaces.Default(color=(0.5, 0.5, 1.0, 1)),  # 设置颜色为浅蓝色
    )

    ########################## 添加机器人实体 ##########################
    # 在场景中添加一个机器人实体，使用URDF文件定义其形态，并设置其外观属性
    robot = scene.add_entity(
        morph=gs.morphs.URDF(
            scale=1.0,  # 缩放比例
            file="urdf/shadow_hand/shadow_hand.urdf",  # 机器人的URDF文件路径
        ),
        surface=gs.surfaces.Reflective(color=(0.4, 0.4, 0.4)),  # 设置颜色为灰色且具有反射效果
    )

    ########################## 构建场景并重置状态 ##########################
    # 构建场景，准备进行仿真
    scene.build()
    # 重置场景状态
    scene.reset()

    # 定义目标四元数，表示目标物体的方向
    target_quat = np.array([1, 0, 0, 0])

    # 获取机器人末端执行器的链接对象（手指远端和前臂）
    index_finger_distal = robot.get_link("index_finger_distal")  # 食指远端
    middle_finger_distal = robot.get_link("middle_finger_distal")  # 中指远端
    forearm = robot.get_link("forearm")  # 前臂

    # 定义中心位置和两个半径
    center = np.array([0.5, 0.5, 0.2])  # 中心位置坐标
    r1 = 0.1  # 第一个半径
    r2 = 0.13  # 第二个半径

    # 进行2000次仿真循环
    for i in range(2000):
        # 计算食指远端的位置，基于当前循环次数和圆周运动公式
        index_finger_pos = center + np.array([np.cos(i / 90 * np.pi), np.sin(i / 90 * np.pi), 0]) * r1
        # 计算中指远端的位置，基于当前循环次数和圆周运动公式
        middle_finger_pos = center + np.array([np.cos(i / 90 * np.pi), np.sin(i / 90 * np.pi), 0]) * r2
        # 计算前臂的位置，基于食指远端的位置向下移动一段距离
        forearm_pos = index_finger_pos - np.array([0, 0, 0.40])

        # 更新目标实体的位置和方向
        target_1.set_qpos(np.concatenate([index_finger_pos, target_quat]))  # 更新目标1的位置和方向
        target_2.set_qpos(np.concatenate([middle_finger_pos, target_quat]))  # 更新目标2的位置和方向
        target_3.set_qpos(np.concatenate([forearm_pos, target_quat]))  # 更新目标3的位置和方向

        # 使用逆运动学计算机器人关节角度，使末端执行器达到指定位置
        qpos = robot.inverse_kinematics_multilink(
            links=[index_finger_distal, middle_finger_distal, forearm],  # 指定需要控制的链接
            poss=[index_finger_pos, middle_finger_pos, forearm_pos],  # 指定目标位置
        )

        # 设置机器人关节的角度
        robot.set_qpos(qpos)

        # 执行一步仿真
        scene.step()


if __name__ == "__main__":
    # 如果脚本作为主程序运行，则调用main函数
    main()