import torch
from .vecenv_wrapper import RslRlVecEnvWrapper
from rl_lab.env.go2_env import Go2Env
class RslRlVecEnvWrapperextra(RslRlVecEnvWrapper):
    def __init__(self, env:Go2Env):
        """Initializes the wrapper.

        Note:
            The wrapper calls :meth:`reset` at the start since the RSL-RL runner does not call reset.

        Args:
            env: The environment to wrap around.

        Raises:
            ValueError: When the environment is not an instance of :class:`ManagerBasedRLEnv` or :class:`DirectRLEnv`.
        """
        # initialize the wrapper
        self.env = env
        # store information required by wrapper
        self.num_envs = self.env.num_envs
        self.device = self.env.device
        self.max_episode_length = self.env.max_episode_length
        self.num_actions = self.env.num_actions
        self.num_obs = self.env.num_obs
        # -- privileged observations
        if self.env.num_privileged_obs is not None:
            self.num_privileged_obs = self.env.num_privileged_obs
        # reset at the start since the RSL-RL runner does not call reset
        self.env.reset()
    
    def get_observations(self) -> tuple[torch.Tensor, dict]:
        """Returns the current observations of the environment."""
        obs_dict = self.env.get_observations()
        return obs_dict["policy"], {"observations": obs_dict}