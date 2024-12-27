import numpy as np
import genesis as gs
from genesis.ext import trimesh
from typing import Any, List, Optional, Tuple, Union
from genesis.ext.isaacgym import terrain_utils as isaacgym_terrain_utils
from genesis.options.morphs import Morph

class RoughTerrain(Morph):
    mesh_type: str = 'trimesh'  # "heightfield" # none, plane, heightfield or trimesh
    horizontal_scale: float = 0.1  # [m]
    vertical_scale: float = 0.005  # [m]
    border_size: float = 25  # [m]
    curriculum: bool = True
    static_friction: float = 1.0
    dynamic_friction: float = 1.0
    restitution: float = 0.0
    # rough terrain only:
    measure_heights: bool = True
    measured_points_x: List[float] = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]  # 1mx1.6m rectangle (without center line)
    measured_points_y: List[float] = [-0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    selected: bool = False  # select a unique terrain type and pass all arguments
    terrain_kwargs: Optional[dict] = None  # Dict of arguments for selected terrain
    max_init_terrain_level: int = 5  # starting curriculum state
    terrain_length = 8.
    terrain_width = 8.
    num_rows: int = 10  # number of terrain rows (levels)
    num_cols: int = 20  # number of terrain cols (types)
    # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete]
    terrain_proportions: List[float] = [0.1, 0.2, 0.3, 0.3, 0.1]
    # trimesh only:
    slope_treshold: float = 0.75  # slopes above this threshold will be corrected to vertical surfaces

    def __init__(self, **data):
        super().__init__(**data)    
        self.type = self.mesh_type
        # 设置环境尺寸和比例
        self.env_length = self.terrain_length
        self.env_width = self.terrain_width
        self.proportions = [np.sum(self.terrain_proportions[:i+1]) for i in range(len(self.terrain_proportions))]

        self.num_sub_terrains = self.num_rows * self.num_cols
        self.env_origins = np.zeros((self.num_rows, self.num_cols, 3))

        # 计算每个环境的像素宽度和长度
        self.width_per_env_pixels = int(self.env_width / self.horizontal_scale)
        self.length_per_env_pixels = int(self.env_length / self.horizontal_scale)

        # 边界大小和总行数、列数
        self.border = int(self.border_size/self.horizontal_scale)
        self.tot_cols = int(self.num_cols * self.width_per_env_pixels) + 2 * self.border
        self.tot_rows = int(self.num_rows * self.length_per_env_pixels) + 2 * self.border

        # 初始化高度场
        self.height_field_raw = np.zeros((self.tot_rows, self.tot_cols), dtype=np.int16)
        




def randomized_terrain(terrain):
    """
    随机生成地形。
    """
    for k in range(terrain.num_sub_terrains):
        # 获取环境坐标
        (i, j) = np.unravel_index(k, (terrain.num_rows, terrain.num_cols))

        choice = np.random.uniform(0, 1)
        difficulty = np.random.choice([0.5, 0.75, 0.9])
        terrain_obj = make_terrain(choice, difficulty, terrain.horizontal_scale, terrain.vertical_scale, terrain.env_length, terrain.env_width)
        add_terrain_to_map(terrain, terrain_obj, i, j)
    
def curiculum(terrain:RoughTerrain):
    """
    按照课程表生成地形。
    """
    for j in range(terrain.num_cols):
        for i in range(terrain.num_rows):
            difficulty = i / terrain.num_rows
            choice = j / terrain.num_cols + 0.001

            terrain_obj = make_terrain(choice, difficulty, terrain.horizontal_scale, terrain.vertical_scale, terrain.env_length, terrain.env_width)
            add_terrain_to_map(terrain, terrain_obj, i, j)

def selected_terrain(terrain:RoughTerrain):
    """
    选择特定类型的地形。
    """
    terrain_type = terrain.terrain_kwargs.pop('type')
    for k in range(terrain.num_sub_terrains):
        # 获取环境坐标
        (i, j) = np.unravel_index(k, (terrain.num_rows, terrain.num_cols))

        terrain_obj = isaacgym_terrain_utils.SubTerrain("terrain",
                          width=terrain.width_per_env_pixels,
                          length=terrain.width_per_env_pixels,
                          vertical_scale=terrain.vertical_scale,
                          horizontal_scale=terrain.horizontal_scale)

        eval(terrain_type)(terrain_obj, **terrain.terrain_kwargs.terrain_kwargs)
        add_terrain_to_map(terrain, terrain_obj, i, j)

def add_terrain_to_map(terrain:RoughTerrain, terrain_obj:isaacgym_terrain_utils.SubTerrain, row, col):
    """
    将生成的地形添加到地图中。

    参数:
        terrain_obj (SubTerrain): 生成的地形对象。
        row (int): 行索引。
        col (int): 列索引。
    """
    # 初始化行和列索引
    i = row
    j = col
    
    # 计算地形在地图上的坐标范围
    start_x = terrain.border + i * terrain.length_per_env_pixels
    end_x = terrain.border + (i + 1) * terrain.length_per_env_pixels
    start_y = terrain.border + j * terrain.width_per_env_pixels
    end_y = terrain.border + (j + 1) * terrain.width_per_env_pixels
    
    # 将生成的地形高度数据添加到地图的相应位置
    terrain.height_field_raw[start_x: end_x, start_y:end_y] = terrain_obj.height_field_raw

    # 计算环境原点在地图上的位置
    env_origin_x = (i + 0.5) * terrain.env_length
    env_origin_y = (j + 0.5) * terrain.env_width

    # 计算环境原点的z轴坐标
    x1 = int((terrain.env_length/2. - 1) / terrain.horizontal_scale)
    x2 = int((terrain.env_length/2. + 1) / terrain.horizontal_scale)
    y1 = int((terrain.env_width/2. - 1) / terrain.horizontal_scale)
    y2 = int((terrain.env_width/2. + 1) / terrain.horizontal_scale)
    env_origin_z = np.max(terrain_obj.height_field_raw[x1:x2, y1:y2])*terrain.vertical_scale

    # 更新环境原点数组
    terrain.env_origins[i, j] = [env_origin_x, env_origin_y, env_origin_z]

def make_terrain(choice, difficulty, horizontal_scale, vertical_scale, env_length, env_width):
    """
    根据选择和难度生成地形。
    
    参数:
    - choice: 用户的选择，用于决定生成哪种地形。
    - difficulty: 地形的难度，影响地形的具体参数。
    - horizontal_scale: 地形的水平比例。
    - vertical_scale: 地形的垂直比例。
    - env_length: 地形的长度。
    - env_width: 地形的宽度。
    
    返回:
    - 生成的地形对象。
    """
    # 根据用户的选择生成不同类型的地形
    if choice < 0.1:
        # 生成斜坡地形，斜率受难度影响
        return isaacgym_terrain_utils.sloped_terrain(
            width=env_width,
            length=env_length,
            vertical_scale=vertical_scale,
            horizontal_scale=horizontal_scale,
            slope=-0.05 * difficulty
        )
    elif choice < 0.3:
        # 生成离散障碍物地形，障碍物的高度受难度影响
        return isaacgym_terrain_utils.discrete_obstacles_terrain(
            width=env_width,
            length=env_length,
            vertical_scale=vertical_scale,
            horizontal_scale=horizontal_scale,
            max_height=0.05 * difficulty,
            min_size=1.0,
            max_size=5.0,
            num_rects=20
        )
    elif choice < 0.6:
        # 生成楼梯地形，楼梯的高度受难度影响
        return isaacgym_terrain_utils.stairs_terrain(
            width=env_width,
            length=env_length,
            vertical_scale=vertical_scale,
            horizontal_scale=horizontal_scale,
            step_width=0.75,
            step_height=-0.1 * difficulty
        )
    elif choice < 0.9:
        # 生成金字塔斜坡地形，斜率受难度影响
        return isaacgym_terrain_utils.pyramid_sloped_terrain(
            width=env_width,
            length=env_length,
            vertical_scale=vertical_scale,
            horizontal_scale=horizontal_scale,
            slope=-0.1 * difficulty
        )
    else:
        # 生成波浪地形，波浪的振幅受难度影响
        return isaacgym_terrain_utils.wave_terrain(
            width=env_width,
            length=env_length,
            vertical_scale=vertical_scale,
            horizontal_scale=horizontal_scale,
            num_waves=2.0,
            amplitude=0.1 * difficulty
        )
        

def parse_curriculum_terrain(morph: RoughTerrain, surface):
# 在 genesis/utils/terrain.py 文件中
    # 根据配置生成不同类型的地形
    if morph.curriculum:
        curiculum(morph)
    elif morph.selected:
        selected_terrain(morph)
    else:    
        randomized_terrain(morph)   
    
    heightfield = morph.height_field_raw


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
    Adapted from Issac Gym's `convert_heightfield_to_trimesh` function.
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
    hf = height_field_raw
    num_rows = hf.shape[0]
    num_cols = hf.shape[1]

    y = np.linspace(0, (num_cols - 1) * horizontal_scale, num_cols)
    x = np.linspace(0, (num_rows - 1) * horizontal_scale, num_rows)
    yy, xx = np.meshgrid(y, x)

    if slope_threshold is not None:
        assert False  # our sdf representation doesn't support steep slopes well

        slope_threshold *= horizontal_scale / vertical_scale
        move_x = np.zeros((num_rows, num_cols))
        move_y = np.zeros((num_rows, num_cols))
        move_corners = np.zeros((num_rows, num_cols))
        move_x[: num_rows - 1, :] += hf[1:num_rows, :] - hf[: num_rows - 1, :] > slope_threshold
        move_x[1:num_rows, :] -= hf[: num_rows - 1, :] - hf[1:num_rows, :] > slope_threshold
        move_y[:, : num_cols - 1] += hf[:, 1:num_cols] - hf[:, : num_cols - 1] > slope_threshold
        move_y[:, 1:num_cols] -= hf[:, : num_cols - 1] - hf[:, 1:num_cols] > slope_threshold
        move_corners[: num_rows - 1, : num_cols - 1] += (
            hf[1:num_rows, 1:num_cols] - hf[: num_rows - 1, : num_cols - 1] > slope_threshold
        )
        move_corners[1:num_rows, 1:num_cols] -= (
            hf[: num_rows - 1, : num_cols - 1] - hf[1:num_rows, 1:num_cols] > slope_threshold
        )
        xx += (move_x + move_corners * (move_x == 0)) * horizontal_scale
        yy += (move_y + move_corners * (move_y == 0)) * horizontal_scale

    # create triangle mesh vertices and triangles from the heightfield grid
    vertices_top = np.zeros((num_rows * num_cols, 3), dtype=np.float32)
    vertices_top[:, 0] = xx.flatten()
    vertices_top[:, 1] = yy.flatten()
    vertices_top[:, 2] = hf.flatten() * vertical_scale
    triangles_top = -np.ones((2 * (num_rows - 1) * (num_cols - 1), 3), dtype=np.uint32)
    for i in range(num_rows - 1):
        ind0 = np.arange(0, num_cols - 1) + i * num_cols
        ind1 = ind0 + 1
        ind2 = ind0 + num_cols
        ind3 = ind2 + 1
        start = 2 * i * (num_cols - 1)
        stop = start + 2 * (num_cols - 1)
        triangles_top[start:stop:2, 0] = ind0
        triangles_top[start:stop:2, 1] = ind3
        triangles_top[start:stop:2, 2] = ind1
        triangles_top[start + 1 : stop : 2, 0] = ind0
        triangles_top[start + 1 : stop : 2, 1] = ind2
        triangles_top[start + 1 : stop : 2, 2] = ind3

    # bottom plane
    z_min = np.min(vertices_top[:, 2]) - 1.0

    vertices_bottom = np.zeros((num_rows * num_cols, 3), dtype=np.float32)
    vertices_bottom[:, 0] = xx.flatten()
    vertices_bottom[:, 1] = yy.flatten()
    vertices_bottom[:, 2] = z_min
    triangles_bottom = -np.ones((2 * (num_rows - 1) * (num_cols - 1), 3), dtype=np.uint32)
    for i in range(num_rows - 1):
        ind0 = np.arange(0, num_cols - 1) + i * num_cols
        ind1 = ind0 + 1
        ind2 = ind0 + num_cols
        ind3 = ind2 + 1
        start = 2 * i * (num_cols - 1)
        stop = start + 2 * (num_cols - 1)
        triangles_bottom[start:stop:2, 0] = ind0
        triangles_bottom[start:stop:2, 2] = ind3
        triangles_bottom[start:stop:2, 1] = ind1
        triangles_bottom[start + 1 : stop : 2, 0] = ind0
        triangles_bottom[start + 1 : stop : 2, 2] = ind2
        triangles_bottom[start + 1 : stop : 2, 1] = ind3
    triangles_bottom += num_rows * num_cols

    # side face
    triangles_side_0 = np.zeros([2 * (num_rows - 1), 3], dtype=np.uint32)
    for i in range(num_rows - 1):
        ind0 = i * num_cols
        ind1 = (i + 1) * num_cols
        ind2 = ind0 + num_rows * num_cols
        ind3 = ind1 + num_rows * num_cols
        triangles_side_0[2 * i] = [ind0, ind2, ind1]
        triangles_side_0[2 * i + 1] = [ind1, ind2, ind3]

    triangles_side_1 = np.zeros([2 * (num_cols - 1), 3], dtype=np.uint32)
    for i in range(num_cols - 1):
        ind0 = i
        ind1 = i + 1
        ind2 = ind0 + num_rows * num_cols
        ind3 = ind1 + num_rows * num_cols
        triangles_side_1[2 * i] = [ind0, ind1, ind2]
        triangles_side_1[2 * i + 1] = [ind1, ind3, ind2]

    triangles_side_2 = np.zeros([2 * (num_rows - 1), 3], dtype=np.uint32)
    for i in range(num_rows - 1):
        ind0 = i * num_cols + num_cols - 1
        ind1 = (i + 1) * num_cols + num_cols - 1
        ind2 = ind0 + num_rows * num_cols
        ind3 = ind1 + num_rows * num_cols
        triangles_side_2[2 * i] = [ind0, ind1, ind2]
        triangles_side_2[2 * i + 1] = [ind1, ind3, ind2]

    triangles_side_3 = np.zeros([2 * (num_cols - 1), 3], dtype=np.uint32)
    for i in range(num_cols - 1):
        ind0 = i + (num_rows - 1) * num_cols
        ind1 = i + 1 + (num_rows - 1) * num_cols
        ind2 = ind0 + num_rows * num_cols
        ind3 = ind1 + num_rows * num_cols
        triangles_side_3[2 * i] = [ind0, ind2, ind1]
        triangles_side_3[2 * i + 1] = [ind1, ind2, ind3]

    vertices = np.concatenate([vertices_top, vertices_bottom], axis=0)
    triangles = np.concatenate(
        [triangles_top, triangles_bottom, triangles_side_0, triangles_side_1, triangles_side_2, triangles_side_3],
        axis=0,
    )

    # This a uniformly-distributed full mesh, which gives faster sdf generation
    sdf_mesh = trimesh.Trimesh(vertices, triangles, process=False)
    # This is the mesh used for non-sdf purposes. It's losslessly simplified from the full mesh, to save memory cost for storing verts and faces
    mesh = sdf_mesh.simplify_quadric_decimation(face_count=0, maximum_error=0.0)

    return mesh, sdf_mesh
