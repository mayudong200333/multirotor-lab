from pybullet_multirotor.envs.base_drone_env import BaseDroneEnv, DroneModel
from gymnasium import spaces
import numpy as np


class BaseDroneRlEnv(BaseDroneEnv):
    """Base class for RL environments with drones."""

    def __init__(self,
                 drone_model: DroneModel = DroneModel.MULTIROTOR,
                 num_drones: int = 1,
                 initial_position=None,
                 episode_len_sec: int = 5,
                 freq: int = 240,
                 aggr_phy_steps: int = 1,
                 gui=False,
                 ):
        self.episode_len_sec = episode_len_sec
        super().__init__(drone_model, num_drones, initial_position, freq, aggr_phy_steps, gui)
        self.HOVER_RPM = np.sqrt(9.8 / (4 * self.kf))

    def _action_space(self):
        """Returns the action space of the environment."""
        return spaces.Box(low=-1 * np.ones(4), high=np.ones(4), dtype=np.float32)

    def _preprocess_action(self, action):
        """Preprocesses the action."""
        return np.array(self.HOVER_RPM * (1 + 0.05 * action))

    def _observation_space(self):
        """Returns the observation space of the environment."""
        # ########################################################### ### OBS OF SIZE 20 (WITH QUATERNION AND RPMS)
        # ### Observation vector ### X        Y        Z       Q1   Q2   Q3   Q4   R       P       Y       VX
        # VY       VZ       WX       WY       WZ       P0            P1            P2            P3 obs_lower_bound =
        # np.array([-1,      -1,      0,      -1,  -1,  -1,  -1,  -1,     -1,     -1,     -1,      -1,      -1,
        # -1,      -1,      -1,      -1,           -1,           -1,           -1]) obs_upper_bound = np.array([1,
        # 1,       1,      1,   1,   1,   1,   1,      1,      1,      1,       1,       1,       1,       1,
        # 1,       1,            1,            1,            1]) return spaces.Box( low=obs_lower_bound,
        # high=obs_upper_bound, dtype=np.float32 ) ###########################################################
        return spaces.Box(low=np.array([-1, -1, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1]),
                          high=np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
                          dtype=np.float32
                          )

    def _compute_obs(self):
        """Computes the observation."""
        obs = self._clip_and_normalize_state(self._get_drone_state_vector(0))
        ret = np.hstack([obs[0:3], obs[7:10], obs[10:13], obs[13:16]]).reshape(12, )
        return ret.astype('float32')

    def _clip_and_normalize_state(self, state):
        """Clips and normalizes the state."""
        return NotImplementedError
