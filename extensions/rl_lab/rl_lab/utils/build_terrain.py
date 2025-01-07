import os
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import time
import genesis as gs
import genesis.utils.misc as mu
from genesis.ext import trimesh
from genesis.ext.isaacgym import terrain_utils as isaacgym_terrain_utils
class MultiScaleTerrain(gs.morphs.Morph):
    """
    用于创建刚性地形的形态。可以通过两种方式实例化：1) 使用给定配置生成的子地形网格，2) 使用给定高度场生成的地形。

    如果 `randomize` 为 True，则涉及随机性的子地形类型将具有随机参数。否则，它们将使用固定的随机种子 0。

    用户可以通过指定 `subterrain_types` 参数轻松配置子地形类型。如果使用单个字符串，则将其重复用于所有子地形。如果是 2D 列表，则应具有与 `n_subterrains` 相同的形状。支持的子地形类型包括：

    - 'flat_terrain': 平坦地形
    - 'random_uniform_terrain': 随机均匀地形
    - 'sloped_terrain': 倾斜地形
    - 'pyramid_sloped_terrain': 金字塔倾斜地形
    - 'discrete_obstacles_terrain': 离散障碍物地形
    - 'wave_terrain': 波浪地形
    - 'stairs_terrain': 台阶地形
    - 'pyramid_stairs_terrain': 金字塔台阶地形
    - 'stepping_stones_terrain': 跳石地形

    注意
    ----
    刚性地形也将以 SDF 形式表示以进行碰撞检查，但其分辨率是自动计算的，并忽略 `gs.materials.Rigid()` 中指定的值。

    参数
    ----------
    file : str
        文件路径。
    scale : float 或 tuple, 可选
        实体大小的缩放因子。如果为浮点数，则均匀缩放。如果为 3 元组，则沿每个轴缩放。默认为 1.0。注意，3 元组缩放仅支持 `gs.morphs.Mesh`。
    pos : tuple, 形状 (3,), 可选
        实体在米中的位置。默认为 (0.0, 0.0, 0.0)。
    visualization : bool, 可选
        实体是否需要可视化。如果需要仅用于碰撞检测的不可见对象，请将其设置为 False。默认为 True。`visualization` 和 `collision` 不能同时为 False。
    collision : bool, 可选
        实体是否需要考虑碰撞检测。默认为 True。`visualization` 和 `collision` 不能同时为 False。
    randomize : bool, 可选
        是否随机化涉及随机性的子地形。默认为 False。
    n_subterrains : tuple of int, 可选
        x 和 y 方向的子地形数量。默认为 (3, 3)。
    subterrain_size : tuple of float, 可选
        每个子地形的大小（米）。默认为 (12.0, 12.0)。
    horizontal_scale : float, 可选
        子地形中每个单元格的大小（米）。默认为 0.25。
    vertical_scale : float, 可选
        子地形中每个台阶的高度（米）。默认为 0.005。
    subterrain_types : str 或 2D list of str, 可选
        要生成的子地形类型。如果为字符串，则将其重复用于所有子地形。如果是 2D 列表，则应具有与 `n_subterrains` 相同的形状。
    height_field : array-like, 可选
        用于生成地形的高度场。如果指定，则忽略所有其他配置。默认为 None。
    """

    randomize: bool = False  # whether to randomize the terrain
    n_subterrains: Tuple[int, int] = (3, 3)  # number of subterrains in x and y directions
    subterrain_size: Tuple[float, float] = (8,8)  # meter
    horizontal_scale:float  = 0.1  # meter size of each cell in the subterrain
    vertical_scale: float = 0.005   # meter height of each step in the subterrain
    subterrain_types: Any = [
        ["flat_terrain", "random_uniform_terrain", "stepping_stones_terrain"],
        ["pyramid_sloped_terrain", "discrete_obstacles_terrain", "wave_terrain"],
        ["random_uniform_terrain", "pyramid_stairs_terrain", "sloped_terrain"],
    ]
    height_field: Any = None
    curriculum: bool = True
    def __init__(self, **data):
        super().__init__(**data)

        supported_subterrain_types = [
            "flat_terrain",
            "fractal_terrain",
            "random_uniform_terrain",
            "sloped_terrain",
            "pyramid_sloped_terrain",
            "pyramid_sloped_neg_terrain",
            "discrete_obstacles_terrain",
            "wave_terrain",
            "stairs_terrain",
            "pyramid_stairs_terrain",
            "pyramid_stairs_neg_terrain",
            "stepping_stones_terrain",
        ]

        if self.height_field is not None:
            try:
                if np.array(self.height_field).ndim != 2:
                    gs.raise_exception("`height_field` should be a 2D array.")
            except:
                gs.raise_exception("`height_field` should be array-like to be converted to np.ndarray.")

            return
        
        if self.curriculum:
            if isinstance(self.subterrain_types, str):
                subterrain_types = []
                for i in range(self.n_subterrains[0]):
                    row = []
                    for j in range(self.n_subterrains[1]):
                        row.append(self.subterrain_types)
                    subterrain_types.append(row)
                self.subterrain_types = subterrain_types
            elif isinstance(self.subterrain_types, list):
                # 将一维列表转换为二维列表
                subterrain_types = []
                for i in range(self.n_subterrains[0]):
                    row = []
                    for j in range(self.n_subterrains[1]):
                        row.append(self.subterrain_types[i])
                    subterrain_types.append(row)
                self.subterrain_types = subterrain_types
            else:
                if np.array(self.subterrain_types).shape != (self.n_subterrains[0], self.n_subterrains[1]):
                    gs.raise_exception(
                        "`subterrain_types` should be either a string or a 2D list of strings with the same shape as `n_subterrains`."
                    )
        else:
            # curriculum == False 的逻辑保持不变
            if isinstance(self.subterrain_types, str):
                subterrain_types = []
                for i in range(self.n_subterrains[0]):
                    row = []
                    for j in range(self.n_subterrains[1]):
                        row.append(self.subterrain_types)
                    subterrain_types.append(row)
                self.subterrain_types = subterrain_types
            else:
                if np.array(self.subterrain_types).shape != (self.n_subterrains[0], self.n_subterrains[1]):
                    gs.raise_exception(
                        "`subterrain_types` should be either a string or a 2D list of strings with the same shape as `n_subterrains`."
                    )
                
        for row in self.subterrain_types:
            for subterrain_type in row:
                if subterrain_type not in supported_subterrain_types:
                    gs.raise_exception(
                        f"Unsupported subterrain type: {subterrain_type}, should be one of {supported_subterrain_types}"
                    )

        # 修改这部分逻辑以适应 horizontal_scale 和 vertical_scale 为列表
        if not mu.is_approx_multiple(self.subterrain_size[0], self.horizontal_scale) or not mu.is_approx_multiple(
            self.subterrain_size[1], self.horizontal_scale
        ):
            gs.raise_exception("`subterrain_size` should be divisible by `horizontal_scale`.")
            




def parse_terrain(morph: MultiScaleTerrain, surface):
# 在 genesis/utils/terrain.py 文件中
    """
    根据 morph 传递的配置生成网格（和高度场）。

    ------------------------------------------------------------------------------------------------------
    如果 morph.height_field 没有传递，则根据以下配置生成高度场：
        n_subterrains    : Tuple[int, int]     = (3, 3)     # x 和 y 方向上的子地形数量
        subterrain_types : Any                 = [
                ['flat_terrain', 'random_uniform_terrain', 'pyramid_sloped_terrain'],
                ['pyramid_sloped_terrain', 'discrete_obstacles_terrain', 'wave_terrain'],
                ['random_uniform_terrain', 'pyramid_stairs_terrain', 'pyramid_sloped_terrain'],
        ]                                                   # x 和 y 方向上的子地形类型
    如果传递了 morph.height_field，则忽略 (n_subterrains, subterrain_size, subterrain_types)。
    ------------------------------------------------------------------------------------------------------

    返回值
    --------------------------
    vmesh        : Mesh
    mesh         : Mesh
    height_field : np.ndarray
    """

    if morph.height_field is None:

        subterrain_rows = int(morph.subterrain_size[0] / morph.horizontal_scale)
        subterrain_cols = int(morph.subterrain_size[1] / morph.horizontal_scale)
        heightfield = np.zeros(
            np.array(morph.n_subterrains) * np.array([subterrain_rows, subterrain_cols]), dtype=np.int16
        )

        fractal_terrain_levels = 8
        fractal_terrain_scale = 5.0

        random_uniform_terrain_height = 0.1

        random_uniform_terrain_step = 0.1
        random_uniform_terrain_downsampled_scale = 0.5

        sloped_terrain_slope = -0.5

        pyramid_sloped_terrain_slope = -0.1

        discrete_obstacles_terrain_max_height = 0.05
        discrete_obstacles_terrain_min_size = 1.0
        discrete_obstacles_terrain_max_size = 5.0
        discrete_obstacles_terrain_num_rects = 20

        wave_terrain_num_waves = 2.0
        wave_terrain_amplitude = 0.1

        stairs_terrain_step_width = 0.75
        stairs_terrain_step_height = -0.1

        stepping_stones_terrain_stone_size = 1.0
        stepping_stones_terrain_stone_distance = 0.25
        stepping_stones_terrain_max_height = 0.2
        stepping_stones_terrain_platform_size = 0.0
        for i in range(morph.n_subterrains[0]):
            for j in range(morph.n_subterrains[1]):
                if morph.curriculum:
                    difficulty = (j)/morph.n_subterrains[1]                    
                    pyramid_sloped_terrain_slope = difficulty * 0.4
                    random_uniform_terrain_height = 0.01 + 0.07 * difficulty
                    stairs_terrain_step_height = 0.05 + 0.18 * difficulty
                    discrete_obstacles_terrain_max_height = 0.05 + difficulty * 0.1
                    stepping_stones_terrain_stone_size = 1.5 * (1.05 - difficulty)
                    stepping_stones_terrain_stone_distance = 0.05 if difficulty == 0 else 0.1
                    gap_size = 1. * difficulty
                    pit_depth = 1. * difficulty
                subterrain_type = morph.subterrain_types[i][j]

                new_subterrain = isaacgym_terrain_utils.SubTerrain(
                    width=subterrain_rows,
                    length=subterrain_cols,
                    vertical_scale=morph.vertical_scale,
                    horizontal_scale=morph.horizontal_scale,
                )
                if not morph.randomize:
                    saved_state = np.random.get_state()
                    np.random.seed(0)

                if subterrain_type == "flat_terrain":
                    subterrain_height_field = np.zeros((subterrain_rows, subterrain_cols), dtype=np.int16)

                elif subterrain_type == "fractal_terrain":
                    subterrain_height_field = fractal_terrain(new_subterrain, levels=fractal_terrain_levels, scale=fractal_terrain_scale).height_field_raw

                elif subterrain_type == "random_uniform_terrain":
                    subterrain_height_field = isaacgym_terrain_utils.random_uniform_terrain(
                        new_subterrain,
                        min_height=-random_uniform_terrain_height,
                        max_height=random_uniform_terrain_height,
                        step=random_uniform_terrain_step,
                        downsampled_scale=random_uniform_terrain_downsampled_scale,
                    ).height_field_raw

                elif subterrain_type == "sloped_terrain":
                    subterrain_height_field = isaacgym_terrain_utils.sloped_terrain(
                        new_subterrain,
                        slope=sloped_terrain_slope,
                    ).height_field_raw

                elif subterrain_type == "pyramid_sloped_terrain":
                    subterrain_height_field = isaacgym_terrain_utils.pyramid_sloped_terrain(
                        new_subterrain,
                        slope=pyramid_sloped_terrain_slope,
                    ).height_field_raw
                elif subterrain_type == "pyramid_sloped_neg_terrain":
                    subterrain_height_field = isaacgym_terrain_utils.pyramid_sloped_terrain(
                        new_subterrain,
                        slope=-pyramid_sloped_terrain_slope,
                    ).height_field_raw
                elif subterrain_type == "discrete_obstacles_terrain":
                    subterrain_height_field = isaacgym_terrain_utils.discrete_obstacles_terrain(
                        new_subterrain,
                        max_height=discrete_obstacles_terrain_max_height,
                        min_size=discrete_obstacles_terrain_min_size,
                        max_size=discrete_obstacles_terrain_max_size,
                        num_rects=discrete_obstacles_terrain_num_rects,
                    ).height_field_raw

                elif subterrain_type == "wave_terrain":
                    subterrain_height_field = isaacgym_terrain_utils.wave_terrain(
                        new_subterrain,
                        num_waves=wave_terrain_num_waves,
                        amplitude=wave_terrain_amplitude,
                    ).height_field_raw

                elif subterrain_type == "stairs_terrain":
                    subterrain_height_field = isaacgym_terrain_utils.stairs_terrain(
                        new_subterrain,
                        step_width=stairs_terrain_step_width,
                        step_height=stairs_terrain_step_height,
                    ).height_field_raw
                elif subterrain_type == "stairs_neg_terrain":
                    subterrain_height_field = isaacgym_terrain_utils.stairs_terrain(
                        new_subterrain,
                        step_width=stairs_terrain_step_width,
                        step_height=-stairs_terrain_step_height,
                    ).height_field_raw
                elif subterrain_type == "pyramid_stairs_terrain":
                    subterrain_height_field = isaacgym_terrain_utils.pyramid_stairs_terrain(
                        new_subterrain,
                        step_width=stairs_terrain_step_width,
                        step_height=stairs_terrain_step_height,
                    ).height_field_raw
                elif subterrain_type == "pyramid_stairs_neg_terrain":
                    subterrain_height_field = isaacgym_terrain_utils.pyramid_stairs_terrain(
                        new_subterrain,
                        step_width=stairs_terrain_step_width,
                        step_height=-stairs_terrain_step_height,
                    ).height_field_raw
                elif subterrain_type == "stepping_stones_terrain":
                    subterrain_height_field = isaacgym_terrain_utils.stepping_stones_terrain(
                        new_subterrain,
                        stone_size=stepping_stones_terrain_stone_size,
                        stone_distance=stepping_stones_terrain_stone_distance,
                        max_height=stepping_stones_terrain_max_height,
                        platform_size=stepping_stones_terrain_platform_size,
                    ).height_field_raw

                else:
                    gs.raise_exception(f"Unsupported subterrain type: {subterrain_type}")

                if not morph.randomize:
                    np.random.set_state(saved_state)

                heightfield[
                    i * subterrain_rows : (i + 1) * subterrain_rows, j * subterrain_cols : (j + 1) * subterrain_cols
                ] = subterrain_height_field

    else:
        heightfield = morph.height_field

    tmesh, sdf_tmesh = convert_heightfield_to_watertight_trimesh(
        heightfield,
        horizontal_scale=morph.horizontal_scale,
        vertical_scale=morph.vertical_scale,
    )
    vmesh = gs.Mesh.from_trimesh(mesh=tmesh, surface=surface)
    mesh = gs.Mesh.from_trimesh(
        mesh=tmesh,
        surface=gs.surfaces.Collision(),
        metadata={
            "horizontal_scale": morph.horizontal_scale,
            "sdf_mesh": sdf_tmesh,
            "height_field": heightfield,
        },
    )
    return vmesh, mesh, heightfield


def fractal_terrain(terrain, levels=8, scale=1.0):
    """
    Generates a fractal terrain

    Parameters
        terrain (SubTerrain): the terrain
        levels (int, optional): granurarity of the fractal terrain. Defaults to 8.
        scale (float, optional): scales vertical variation. Defaults to 1.0.
    """
    width = terrain.width
    length = terrain.length
    height = np.zeros((width, length))
    for level in range(1, levels + 1):
        step = 2 ** (levels - level)
        for y in range(0, width, step):
            y_skip = (1 + y // step) % 2
            for x in range(step * y_skip, length, step * (1 + y_skip)):
                x_skip = (1 + x // step) % 2
                xref = step * (1 - x_skip)
                yref = step * (1 - y_skip)
                mean = height[y - yref : y + yref + 1 : 2 * step, x - xref : x + xref + 1 : 2 * step].mean()
                variation = 2 ** (-level) * np.random.uniform(-1, 1)
                height[y, x] = mean + scale * variation

    height /= terrain.vertical_scale
    terrain.height_field_raw = height.astype(np.int16)
    return terrain


def convert_heightfield_to_watertight_trimesh(height_field_raw, horizontal_scale, vertical_scale, slope_threshold=None):
    """
    将高度场数组转换为由顶点和三角形表示的三角网格。
    可选地，修正超过给定斜率阈值的垂直表面。

    参数:
        height_field_raw (np.array): 输入的高度场数组
        horizontal_scale (float): 高度场的水平比例尺 [米]
        vertical_scale (float): 高度场的垂直比例尺 [米]
        slope_threshold (float): 斜率阈值，超过该阈值的表面将被修正为垂直。如果为 None，则不进行修正 (默认: None)

    返回:
        vertices (np.array(float)): 形状为 (num_vertices, 3) 的数组，每一行表示每个顶点的位置 [米]
        triangles (np.array(int)): 形状为 (num_triangles, 3) 的数组，每一行表示连接三个顶点的三角形的索引
    """
    start_time = time.time()
    # 获取高度场的行数和列数
    hf = height_field_raw
    num_rows = hf.shape[0]  # 获取高度场的行数
    num_cols = hf.shape[1]  # 获取高度场的列数

    # 创建 x 和 y 坐标网格
    y = np.linspace(0, (num_cols - 1) * horizontal_scale, num_cols)  # 创建 y 坐标网格
    x = np.linspace(0, (num_rows - 1) * horizontal_scale, num_rows)  # 创建 x 坐标网格
    yy, xx = np.meshgrid(y, x)  # 创建二维网格坐标

    # 如果提供了斜率阈值，则进行修正
    if slope_threshold is not None:
        assert False  # 当前的 SDF 表示法不支持陡峭斜坡

        # 调整斜率阈值以适应比例尺
        slope_threshold *= horizontal_scale / vertical_scale  # 根据比例尺调整斜率阈值
        
        # 计算需要移动的顶点
        move_x = np.zeros((num_rows, num_cols))  # 初始化 x 方向的移动矩阵
        move_y = np.zeros((num_rows, num_cols))  # 初始化 y 方向的移动矩阵
        move_corners = np.zeros((num_rows, num_cols))  # 初始化角落的移动矩阵
        
        # 计算 x 方向的移动
        move_x[: num_rows - 1, :] += hf[1:num_rows, :] - hf[: num_rows - 1, :] > slope_threshold  # 检查相邻行的高度差是否超过阈值
        move_x[1:num_rows, :] -= hf[: num_rows - 1, :] - hf[1:num_rows, :] > slope_threshold  # 检查相邻行的高度差是否超过阈值
        
        # 计算 y 方向的移动
        move_y[:, : num_cols - 1] += hf[:, 1:num_cols] - hf[:, : num_cols - 1] > slope_threshold  # 检查相邻列的高度差是否超过阈值
        move_y[:, 1:num_cols] -= hf[:, : num_cols - 1] - hf[:, 1:num_cols] > slope_threshold  # 检查相邻列的高度差是否超过阈值
        
        # 计算角落的移动
        move_corners[: num_rows - 1, : num_cols - 1] += (
            hf[1:num_rows, 1:num_cols] - hf[: num_rows - 1, : num_cols - 1] > slope_threshold
        )  # 检查对角线高度差是否超过阈值
        move_corners[1:num_rows, 1:num_cols] -= (
            hf[: num_rows - 1, : num_cols - 1] - hf[1:num_rows, 1:num_cols] > slope_threshold
        )  # 检查对角线高度差是否超过阈值
        
        # 更新 x 和 y 坐标
        xx += (move_x + move_corners * (move_x == 0)) * horizontal_scale  # 更新 x 坐标
        yy += (move_y + move_corners * (move_y == 0)) * horizontal_scale  # 更新 y 坐标

    # 创建顶部平面的顶点和三角形
    vertices_top = np.zeros((num_rows * num_cols, 3), dtype=np.float32)  # 初始化顶部平面的顶点数组
    vertices_top[:, 0] = xx.flatten()  # 设置 x 坐标
    vertices_top[:, 1] = yy.flatten()  # 设置 y 坐标
    vertices_top[:, 2] = hf.flatten() * vertical_scale  # 设置 z 坐标
    
    triangles_top = -np.ones((2 * (num_rows - 1) * (num_cols - 1), 3), dtype=np.uint32)  # 初始化顶部平面的三角形数组
    for i in range(num_rows - 1):
        ind0 = np.arange(0, num_cols - 1) + i * num_cols  # 计算当前行的第一个索引
        ind1 = ind0 + 1  # 计算当前行的第二个索引
        ind2 = ind0 + num_cols  # 计算下一行的第一个索引
        ind3 = ind2 + 1  # 计算下一行的第二个索引
        start = 2 * i * (num_cols - 1)  # 计算当前行的起始索引
        stop = start + 2 * (num_cols - 1)  # 计算当前行的结束索引
        triangles_top[start:stop:2, 0] = ind0  # 设置第一个三角形的第一个顶点索引
        triangles_top[start:stop:2, 1] = ind3  # 设置第一个三角形的第三个顶点索引
        triangles_top[start:stop:2, 2] = ind1  # 设置第一个三角形的第二个顶点索引
        triangles_top[start + 1 : stop : 2, 0] = ind0  # 设置第二个三角形的第一个顶点索引
        triangles_top[start + 1 : stop : 2, 1] = ind2  # 设置第二个三角形的第三个顶点索引
        triangles_top[start + 1 : stop : 2, 2] = ind3  # 设置第二个三角形的第二个顶点索引

    # 创建底部平面的顶点和三角形
    z_min = np.min(vertices_top[:, 2]) - 1.0  # 计算底部平面的最低 z 坐标

    vertices_bottom = np.zeros((num_rows * num_cols, 3), dtype=np.float32)  # 初始化底部平面的顶点数组
    vertices_bottom[:, 0] = xx.flatten()  # 设置 x 坐标
    vertices_bottom[:, 1] = yy.flatten()  # 设置 y 坐标
    vertices_bottom[:, 2] = z_min  # 设置 z 坐标
    
    triangles_bottom = -np.ones((2 * (num_rows - 1) * (num_cols - 1), 3), dtype=np.uint32)  # 初始化底部平面的三角形数组
    for i in range(num_rows - 1):
        ind0 = np.arange(0, num_cols - 1) + i * num_cols  # 计算当前行的第一个索引
        ind1 = ind0 + 1  # 计算当前行的第二个索引
        ind2 = ind0 + num_cols  # 计算下一行的第一个索引
        ind3 = ind2 + 1  # 计算下一行的第二个索引
        start = 2 * i * (num_cols - 1)  # 计算当前行的起始索引
        stop = start + 2 * (num_cols - 1)  # 计算当前行的结束索引
        triangles_bottom[start:stop:2, 0] = ind0  # 设置第一个三角形的第一个顶点索引
        triangles_bottom[start:stop:2, 2] = ind3  # 设置第一个三角形的第三个顶点索引
        triangles_bottom[start:stop:2, 1] = ind1  # 设置第一个三角形的第二个顶点索引
        triangles_bottom[start + 1 : stop : 2, 0] = ind0  # 设置第二个三角形的第一个顶点索引
        triangles_bottom[start + 1 : stop : 2, 2] = ind2  # 设置第二个三角形的第三个顶点索引
        triangles_bottom[start + 1 : stop : 2, 1] = ind3  # 设置第二个三角形的第二个顶点索引
    triangles_bottom += num_rows * num_cols  # 调整索引以匹配底部平面的顶点
    
    # 创建侧面的三角形
    triangles_side_0 = np.zeros([2 * (num_rows - 1), 3], dtype=np.uint32)  # 初始化第一个侧面的三角形数组
    for i in range(num_rows - 1):
        ind0 = i * num_cols  # 计算当前行的第一个索引
        ind1 = (i + 1) * num_cols  # 计算下一行的第一个索引
        ind2 = ind0 + num_rows * num_cols  # 计算底部平面的对应索引
        ind3 = ind1 + num_rows * num_cols  # 计算底部平面的对应索引
        triangles_side_0[2 * i] = [ind0, ind2, ind1]  # 设置第一个三角形的顶点索引
        triangles_side_0[2 * i + 1] = [ind1, ind2, ind3]  # 设置第二个三角形的顶点索引

    triangles_side_1 = np.zeros([2 * (num_cols - 1), 3], dtype=np.uint32)  # 初始化第二个侧面的三角形数组
    for i in range(num_cols - 1):
        ind0 = i  # 计算当前列的第一个索引
        ind1 = i + 1  # 计算下一列的第一个索引
        ind2 = ind0 + num_rows * num_cols  # 计算底部平面的对应索引
        ind3 = ind1 + num_rows * num_cols  # 计算底部平面的对应索引
        triangles_side_1[2 * i] = [ind0, ind1, ind2]  # 设置第一个三角形的顶点索引
        triangles_side_1[2 * i + 1] = [ind1, ind3, ind2]  # 设置第二个三角形的顶点索引

    triangles_side_2 = np.zeros([2 * (num_rows - 1), 3], dtype=np.uint32)  # 初始化第三个侧面的三角形数组
    for i in range(num_rows - 1):
        ind0 = i * num_cols + num_cols - 1  # 计算当前行的最后一个索引
        ind1 = (i + 1) * num_cols + num_cols - 1  # 计算下一行的最后一个索引
        ind2 = ind0 + num_rows * num_cols  # 计算底部平面的对应索引
        ind3 = ind1 + num_rows * num_cols  # 计算底部平面的对应索引
        triangles_side_2[2 * i] = [ind0, ind1, ind2]  # 设置第一个三角形的顶点索引
        triangles_side_2[2 * i + 1] = [ind1, ind3, ind2]  # 设置第二个三角形的顶点索引

    triangles_side_3 = np.zeros([2 * (num_cols - 1), 3], dtype=np.uint32)  # 初始化第四个侧面的三角形数组
    for i in range(num_cols - 1):
        ind0 = i + (num_rows - 1) * num_cols  # 计算最后一行的当前列索引
        ind1 = i + 1 + (num_rows - 1) * num_cols  # 计算最后一行的下一列索引
        ind2 = ind0 + num_rows * num_cols  # 计算底部平面的对应索引
        ind3 = ind1 + num_rows * num_cols  # 计算底部平面的对应索引
        triangles_side_3[2 * i] = [ind0, ind2, ind1]  # 设置第一个三角形的顶点索引
        triangles_side_3[2 * i + 1] = [ind1, ind2, ind3]  # 设置第二个三角形的顶点索引

    # 合并所有顶点和三角形
    vertices = np.concatenate([vertices_top, vertices_bottom], axis=0)  # 合并顶部和底部平面的顶点
    triangles = np.concatenate(
        [triangles_top, triangles_bottom, triangles_side_0, triangles_side_1, triangles_side_2, triangles_side_3],
        axis=0,
    )  # 合并所有三角形
    tri_time = time.time() 
    print(f"Terrain generation took {tri_time-start_time:.2f} seconds.")
    # 创建一个均匀分布的完整网格，用于更快的 SDF 生成
    sdf_mesh = trimesh.Trimesh(vertices, triangles, process=False)  # 创建完整的网格对象
    init_time = time.time() 
    print(f"Terrain generation took {init_time-tri_time:.2f} seconds.")
    # 创建一个简化后的网格，用于非 SDF 目的，以节省内存
    #mesh = sdf_mesh.simplify_quadric_decimation(face_count = 0)
    final_time = time.time() 
    print(f"Terrain simplify took {final_time-init_time:.2f} seconds.")
    return sdf_mesh, sdf_mesh  # 返回简化后的网格和完整的网格

def convert_heightfield_to_trimesh(height_field_raw, horizontal_scale, vertical_scale, slope_threshold=None):
    """
    Convert a heightfield array to a triangle mesh represented by vertices and triangles.
    Optionally, corrects vertical surfaces above the provide slope threshold:

        If (y2-y1)/(x2-x1) > slope_threshold -> Move A to A' (set x1 = x2). Do this for all directions.
                   B(x2,y2)
                  /|
                 / |
                /  |
        (x1,y1)A---A'(x2',y1)

    Parameters:
        height_field_raw (np.array): input heightfield
        horizontal_scale (float): horizontal scale of the heightfield [meters]
        vertical_scale (float): vertical scale of the heightfield [meters]
        slope_threshold (float): the slope threshold above which surfaces are made vertical. If None no correction is applied (default: None)
    Returns:
        vertices (np.array(float)): array of shape (num_vertices, 3). Each row represents the location of each vertex [meters]
        triangles (np.array(int)): array of shape (num_triangles, 3). Each row represents the indices of the 3 vertices connected by this triangle.
    """
    start_time = time.time()
    hf = height_field_raw
    num_rows = hf.shape[0]
    num_cols = hf.shape[1]

    y = np.linspace(0, (num_cols-1)*horizontal_scale, num_cols)
    x = np.linspace(0, (num_rows-1)*horizontal_scale, num_rows)
    yy, xx = np.meshgrid(y, x)

    if slope_threshold is not None:

        slope_threshold *= horizontal_scale / vertical_scale
        move_x = np.zeros((num_rows, num_cols))
        move_y = np.zeros((num_rows, num_cols))
        move_corners = np.zeros((num_rows, num_cols))
        move_x[:num_rows-1, :] += (hf[1:num_rows, :] - hf[:num_rows-1, :] > slope_threshold)
        move_x[1:num_rows, :] -= (hf[:num_rows-1, :] - hf[1:num_rows, :] > slope_threshold)
        move_y[:, :num_cols-1] += (hf[:, 1:num_cols] - hf[:, :num_cols-1] > slope_threshold)
        move_y[:, 1:num_cols] -= (hf[:, :num_cols-1] - hf[:, 1:num_cols] > slope_threshold)
        move_corners[:num_rows-1, :num_cols-1] += (hf[1:num_rows, 1:num_cols] - hf[:num_rows-1, :num_cols-1] > slope_threshold)
        move_corners[1:num_rows, 1:num_cols] -= (hf[:num_rows-1, :num_cols-1] - hf[1:num_rows, 1:num_cols] > slope_threshold)
        xx += (move_x + move_corners*(move_x == 0)) * horizontal_scale
        yy += (move_y + move_corners*(move_y == 0)) * horizontal_scale

    # create triangle mesh vertices and triangles from the heightfield grid
    vertices = np.zeros((num_rows*num_cols, 3), dtype=np.float32)
    vertices[:, 0] = xx.flatten()
    vertices[:, 1] = yy.flatten()
    vertices[:, 2] = hf.flatten() * vertical_scale
    triangles = -np.ones((2*(num_rows-1)*(num_cols-1), 3), dtype=np.uint32)
    for i in range(num_rows - 1):
        ind0 = np.arange(0, num_cols-1) + i*num_cols
        ind1 = ind0 + 1
        ind2 = ind0 + num_cols
        ind3 = ind2 + 1
        start = 2*i*(num_cols-1)
        stop = start + 2*(num_cols-1)
        triangles[start:stop:2, 0] = ind0
        triangles[start:stop:2, 1] = ind3
        triangles[start:stop:2, 2] = ind1
        triangles[start+1:stop:2, 0] = ind0
        triangles[start+1:stop:2, 1] = ind2
        triangles[start+1:stop:2, 2] = ind3
    tri_time = time.time() 
    print(f"Terrain generation took {tri_time-start_time:.2f} seconds.")
    # 创建一个均匀分布的完整网格，用于更快的 SDF 生成
    sdf_mesh = trimesh.Trimesh(vertices, triangles, process=False)  # 创建完整的网格对象
    init_time = time.time() 
    print(f"Terrain generation took {init_time-tri_time:.2f} seconds.")
    # 创建一个简化后的网格，用于非 SDF 目的，以节省内存
    mesh = sdf_mesh.simplify_quadric_decimation(face_count = 0)
    final_time = time.time() 
    print(f"Terrain simplify took {final_time-init_time:.2f} seconds.")



    return mesh, sdf_mesh