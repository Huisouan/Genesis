# reward_functions.py
import torch

@torch.jit.script
def _reward_tracking_lin_vel(commands: torch.Tensor, base_lin_vel: torch.Tensor, tracking_sigma: float) -> torch.Tensor:
    lin_vel_error = torch.sum(torch.square(commands[:, :2] - base_lin_vel[:, :2]), dim=1)
    return torch.exp(-lin_vel_error / tracking_sigma)

@torch.jit.script
def _reward_tracking_ang_vel(commands: torch.Tensor, base_ang_vel: torch.Tensor, tracking_sigma: float) -> torch.Tensor:
    ang_vel_error = torch.square(commands[:, 2] - base_ang_vel[:, 2])
    return torch.exp(-ang_vel_error / tracking_sigma)

@torch.jit.script
def _reward_lin_vel_z(base_lin_vel: torch.Tensor) -> torch.Tensor:
    return torch.square(base_lin_vel[:, 2])

@torch.jit.script
def _reward_action_rate(last_actions: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    return torch.sum(torch.square(last_actions - actions), dim=1)

@torch.jit.script
def _reward_similar_to_default(dof_pos: torch.Tensor, default_dof_pos: torch.Tensor) -> torch.Tensor:
    return torch.sum(torch.abs(dof_pos - default_dof_pos), dim=1)

@torch.jit.script
def _reward_base_height(base_pos: torch.Tensor, base_height_target: float) -> torch.Tensor:
    return torch.square(base_pos[:, 2] - base_height_target)
    
@torch.jit.script
def _reward_ang_vel_xy(base_ang_vel: torch.Tensor) -> torch.Tensor:
    return torch.sum(torch.square(base_ang_vel[:, :2]), dim=1)
    
@torch.jit.script
def _reward_orientation(projected_gravity: torch.Tensor) -> torch.Tensor:
    return torch.sum(torch.square(projected_gravity[:, :2]), dim=1)
    
@torch.jit.script
def _reward_dof_acc(last_dof_vel: torch.Tensor, dof_vel: torch.Tensor, dt: float) -> torch.Tensor:
    return torch.sum(torch.square((last_dof_vel - dof_vel) / dt), dim=1)
    
@torch.jit.script
def _reward_joint_power(dof_vel: torch.Tensor, torques: torch.Tensor) -> torch.Tensor:
    return torch.sum(torch.abs(dof_vel) * torch.abs(torques), dim=1)

@torch.jit.script
def _reward_smoothness(actions: torch.Tensor, last_actions: torch.Tensor, last_last_actions: torch.Tensor) -> torch.Tensor:
    return torch.sum(torch.square(actions - last_actions - last_actions + last_last_actions), dim=1)
    
@torch.jit.script
def _reward_torques(torques: torch.Tensor) -> torch.Tensor:
    return torch.sum(torch.square(torques), dim=1)

@torch.jit.script
def _reward_dof_vel(dof_vel: torch.Tensor) -> torch.Tensor:
    return torch.sum(torch.square(dof_vel), dim=1)
    
@torch.jit.script
def _reward_collision(contact_forces: torch.Tensor, penalised_contact_indices: torch.Tensor) -> torch.Tensor:
    return torch.sum(1.*(torch.norm(contact_forces[:, penalised_contact_indices, :], dim=-1) > 0.1), dim=1)
    
@torch.jit.script
def _reward_termination(reset_buf: torch.Tensor, time_out_buf: torch.Tensor) -> torch.Tensor:
    return reset_buf * ~time_out_buf
    
@torch.jit.script
def _reward_dof_pos_limits(dof_pos: torch.Tensor, dof_pos_limits: torch.Tensor) -> torch.Tensor:
    out_of_limits = -(dof_pos - dof_pos_limits[:, 0]).clip(max=0.) # lower limit
    out_of_limits += (dof_pos - dof_pos_limits[:, 1]).clip(min=0.)
    return torch.sum(out_of_limits, dim=1)

@torch.jit.script
def _reward_dof_vel_limits(dof_vel: torch.Tensor, dof_vel_limits: torch.Tensor, soft_dof_vel_limit: float) -> torch.Tensor:
    return torch.sum((torch.abs(dof_vel) - dof_vel_limits*soft_dof_vel_limit).clip(min=0., max=1.), dim=1)

@torch.jit.script
def _reward_torque_limits(torques: torch.Tensor, torque_limits: torch.Tensor, soft_torque_limit: float) -> torch.Tensor:
    return torch.sum((torch.abs(torques) - torque_limits*soft_torque_limit).clip(min=0.), dim=1)

@torch.jit.script
def _reward_feet_air_time(contact_forces: torch.Tensor, commands: torch.Tensor, dt: float, base_lin_vel: torch.Tensor, feet_air_time: torch.Tensor, last_contacts: torch.Tensor) -> torch.Tensor:
    contact = contact_forces[:, :, 2] > 1.
    contact_filt = torch.logical_or(contact, last_contacts) 
    last_contacts = contact
    first_contact = (feet_air_time > 0.) * contact_filt
    feet_air_time += dt
    rew_airTime = torch.sum((feet_air_time - 0.5) * first_contact, dim=1) # reward only on first contact with the ground
    rew_airTime *= torch.norm(commands[:, :2], dim=1) > 0.1 #no reward for zero command
    feet_air_time *= ~contact_filt
    return rew_airTime, feet_air_time, last_contacts
    
@torch.jit.script
def _reward_stumble(contact_forces: torch.Tensor, feet_indices: torch.Tensor) -> torch.Tensor:
    return torch.any(torch.norm(contact_forces[:, feet_indices, :2], dim=2) > 5 * torch.abs(contact_forces[:, feet_indices, 2]), dim=1)
        
@torch.jit.script
def _reward_stand_still(dof_pos: torch.Tensor, commands: torch.Tensor, default_dof_pos: torch.Tensor) -> torch.Tensor:
    return torch.sum(torch.abs(dof_pos - default_dof_pos), dim=1) * (torch.norm(commands[:, :2], dim=1) < 0.1)

@torch.jit.script
def _reward_feet_contact_forces(contact_forces: torch.Tensor, feet_indices: torch.Tensor, max_contact_force: float) -> torch.Tensor:
    return torch.sum((torch.norm(contact_forces[:, feet_indices, :], dim=-1) - max_contact_force).clip(min=0.), dim=1)