import torch
from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat, transform_quat_by_quat
import genesis as gs
from . import Go2BaseEnv
from rl_lab.assets.loder_for_algs import AmpMotion
from rl_lab.utils.kinematics import urdf
class Go2AseEnv(Go2BaseEnv):
    def __init__(self,num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg):
        super().__init__(num_envs=num_envs, env_cfg=env_cfg, obs_cfg=obs_cfg, reward_cfg=reward_cfg, command_cfg=command_cfg)

        self.chain_ee = []
        for ee_name in self.cfg.ee_names:
            with open(self.cfg.urdf_path, "rb") as urdf_file:
                urdf_content = urdf_file.read()
                chain_ee_instance = urdf.build_serial_chain_from_urdf(urdf_content, ee_name).to(device=self.device)
                self.chain_ee.append(chain_ee_instance)

        self.amp_loader = AmpMotion(
            data_dir ="datasets/mocap_motions_go2",
            datatype="isaacgym",
            file_type="txt",
            data_spaces = None,
            env_step_duration = 0.005,
        )

    def get_amp_observations(self):
        self.observation_manager._group_obs_concatenate["AMP"] = False
        group_obs = self.observation_manager.compute_group("AMP")
        self.observation_manager._group_obs_concatenate["AMP"] = True
        # Isaac Sim uses breadth-first joint ordering, while Isaac Gym uses depth-first joint ordering
        joint_pos = group_obs["joint_pos"]
        joint_vel = group_obs["joint_vel"]
        joint_pos = self.amp_loader.reorder_from_isaacsim_to_isaacgym_tool(joint_pos)
        joint_vel = self.amp_loader.reorder_from_isaacsim_to_isaacgym_tool(joint_vel)
        foot_pos = []
        with torch.no_grad():
            for i, chain_ee in enumerate(self.chain_ee):
                foot_pos.append(chain_ee.forward_kinematics(joint_pos[:, i * 3 : i * 3 + 3]).get_matrix()[:, :3, 3])
        foot_pos = torch.cat(foot_pos, dim=-1)
        base_lin_vel = group_obs["base_lin_vel"]
        base_ang_vel = group_obs["base_ang_vel"]
        z_pos = group_obs["base_pos_z"]
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

        self.reset_idx(self.reset_buf.nonzero(as_tuple=False).flatten())

        # compute reward
        self.rew_buf[:] = 0.0
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew

        # compute observations
        self.obs_buf["policy"] = torch.cat(
            [
                self.base_ang_vel * self.obs_scales["ang_vel"],  # 3
                self.projected_gravity,  # 3
                (self.dof_pos - self.default_dof_pos) * self.obs_scales["dof_pos"],  # 12
                self.dof_vel * self.obs_scales["dof_vel"],  # 12
                self.actions,  # 12
            ],
            axis=-1,
        )
        self.
        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]

        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras    