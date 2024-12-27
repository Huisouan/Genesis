
import torch
from torch import Tensor
from typing import Tuple, Dict
from .math import quat_apply_yaw


def init_base_height_points(self):
    """ Returns points at which the height measurments are sampled (in base frame)

    Returns:
        [torch.Tensor]: Tensor of shape (num_envs, self.num_base_height_points, 3)
    """
    y = torch.tensor([-0.2, -0.15, -0.1, -0.05, 0., 0.05, 0.1, 0.15, 0.2], device=self.device, requires_grad=False)
    x = torch.tensor([-0.15, -0.1, -0.05, 0., 0.05, 0.1, 0.15], device=self.device, requires_grad=False)
    grid_x, grid_y = torch.meshgrid(x, y)

    self.num_base_height_points = grid_x.numel()
    points = torch.zeros(self.num_envs, self.num_base_height_points, 3, device=self.device, requires_grad=False)
    points[:, :, 0] = grid_x.flatten()
    points[:, :, 1] = grid_y.flatten()
    return points

def get_heights(self, env_ids=None):
    """ Samples heights of the terrain at required points around each robot.
        The points are offset by the base's position and rotated by the base's yaw

    Args:
        env_ids (List[int], optional): Subset of environments for which to return the heights. Defaults to None.

    Raises:
        NameError: [description]

    Returns:
        [type]: [description]
    """
    if self.cfg.terrain.mesh_type == 'plane':
        return torch.zeros(self.num_envs, self.num_height_points, device=self.device, requires_grad=False)
    elif self.cfg.terrain.mesh_type == 'none':
        raise NameError("Can't measure height with terrain mesh type 'none'")

    if env_ids:
        points = quat_apply_yaw(self.base_quat[env_ids].repeat(1, self.num_height_points), self.height_points[env_ids]) + (self.root_states[env_ids, :3]).unsqueeze(1)
    else:
        points = quat_apply_yaw(self.base_quat.repeat(1, self.num_height_points), self.height_points) + (self.root_states[:, :3]).unsqueeze(1)


    points += self.terrain.cfg.border_size
    points = (points/self.terrain.cfg.horizontal_scale).long()
    px = points[:, :, 0].view(-1)
    py = points[:, :, 1].view(-1)
    px = torch.clip(px, 0, self.height_samples.shape[0]-2)
    py = torch.clip(py, 0, self.height_samples.shape[1]-2)

    heights1 = self.height_samples[px, py]
    heights2 = self.height_samples[px+1, py]
    heights3 = self.height_samples[px, py+1]
    heights = torch.min(heights1, heights2)
    heights = torch.min(heights, heights3)

    return heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale