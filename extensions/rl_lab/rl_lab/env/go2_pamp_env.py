import torch
from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat, transform_quat_by_quat
import genesis as gs
from . import Go2BaseEnv
from rl_lab.assets.motion_loader import AMPLoader
from rl_lab.utils.kinematics import urdf


class Go2PAmpEnv(Go2BaseEnv):
    def __init__(self,num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg,show_viewer=False, device="cuda"):
        super().__init__(num_envs=num_envs, 
                        env_cfg=env_cfg, 
                        obs_cfg=obs_cfg, 
                        reward_cfg=reward_cfg, 
                        command_cfg=command_cfg,
                        show_viewer=show_viewer,
                        device=device)

        self.amp_cfg = self.env_cfg["amp"]
        self.chain_ee = []
        for ee_name in self.amp_cfg["ee_names"]:
            with open(self.amp_cfg["urdf_path"], "rb") as urdf_file:
                urdf_content = urdf_file.read()
                chain_ee_instance = urdf.build_serial_chain_from_urdf(urdf_content, ee_name).to(device=self.device)
                self.chain_ee.append(chain_ee_instance)

        self.amp_loader = AMPLoader(
            device=self.device,
            motion_files=self.amp_cfg["amp_motion_files"],
            time_between_frames=self.dt ,
            num_preload_transitions = self.amp_cfg["amp_replay_buffer_size"],
            preload_transitions = self.amp_cfg["preload_transitions"],
        )


    def get_amp_observations(self):
        joint_pos = self.dof_pos
        joint_vel = self.dof_vel
        foot_pos = []
        with torch.no_grad():
            for i, chain_ee in enumerate(self.chain_ee):
                foot_pos.append(chain_ee.forward_kinematics(joint_pos[:, i * 3 : i * 3 + 3]).get_matrix()[:, :3, 3])
        foot_pos = torch.cat(foot_pos, dim=-1)
        base_lin_vel = self.base_lin_vel
        base_ang_vel = self.base_ang_vel
        z_pos = self.base_pos[:, 2].unsqueeze(-1)
        # joint_pos(0-11) foot_pos(12-23) base_lin_vel(24-26) base_ang_vel(27-29) joint_vel(30-41) z_pos(42)
        return torch.cat((joint_pos, foot_pos, base_lin_vel, base_ang_vel, joint_vel, z_pos), dim=-1)

    def step(self, actions):
        self.actions = torch.clip(actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"])
        exec_actions = self.last_actions if self.simulate_action_latency else self.actions
        target_dof_pos = exec_actions * self.env_cfg["action_scale"] + self.default_dof_pos
        self.robot.control_dofs_position(target_dof_pos, self.motor_dofs)
        self.scene.step()

        # update buffers
        self.episode_length_buf += 1
        self.base_pos[:] = self.robot.get_pos()
        self.base_quat[:] = self.robot.get_quat()
        self.base_euler = quat_to_xyz(
            transform_quat_by_quat(torch.ones_like(self.base_quat) * self.inv_base_init_quat, self.base_quat)
        )
        inv_base_quat = inv_quat(self.base_quat)
        self.base_lin_vel[:] = transform_by_quat(self.robot.get_vel(), inv_base_quat)
        self.base_ang_vel[:] = transform_by_quat(self.robot.get_ang(), inv_base_quat)
        self.projected_gravity = transform_by_quat(self.global_gravity, inv_base_quat)
        self.dof_pos[:] = self.robot.get_dofs_position(self.motor_dofs)
        self.dof_vel[:] = self.robot.get_dofs_velocity(self.motor_dofs)

        # resample commands
        envs_idx = (
            (self.episode_length_buf % int(self.env_cfg["resampling_time_s"] / self.dt) == 0)
            .nonzero(as_tuple=False)
            .flatten()
        )
        self._resample_commands(envs_idx)

        # check termination and reset
        self.reset_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf |= torch.abs(self.base_euler[:, 1]) > self.env_cfg["termination_if_pitch_greater_than"]
        self.reset_buf |= torch.abs(self.base_euler[:, 0]) > self.env_cfg["termination_if_roll_greater_than"]

        time_out_idx = (self.episode_length_buf > self.max_episode_length).nonzero(as_tuple=False).flatten()
        self.extras["time_outs"] = torch.zeros_like(self.reset_buf, device=self.device, dtype=gs.tc_float)
        self.extras["time_outs"][time_out_idx] = 1.0
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()

        terminal_amp_states = self.get_amp_observations()[reset_env_ids]
        self.reset_idx(reset_env_ids)
        
        self.extras["reset_env_ids"] = reset_env_ids
        self.extras["terminal_amp_states"] = terminal_amp_states


        # compute reward
        self.rew_buf[:] = 0.0
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew

        self.compute_observations()

        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]

        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras
