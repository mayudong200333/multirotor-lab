import numpy as np

from pybullet_multirotor.envs.base_drone_rl_env import BaseDroneRlEnv


class SingleDroneRlHoverEnv(BaseDroneRlEnv):
    """Single drone RL hover environment."""

    def __init__(self,
                 target_position=np.array([0, 0, 1])
                 ):
        self.target_position = target_position
        super().__init__()

    def _compute_reward(self):
        state = self._get_drone_state_vector(0)
        reward = -np.linalg.norm(state[0:3] - self.target_position) ** 2
        return reward

    def _compute_done(self):
        if self.step_counter / self.freq > self.episode_len_sec:
            return True
        else:
            return False

    def _compute_info(self):
        return {}

    def _clip_and_normalize_state(self, state):
        MAX_LIN_VEL_XY = 3
        MAX_LIN_VEL_Z = 1

        MAX_XY = MAX_LIN_VEL_XY * self.episode_len_sec
        MAX_Z = MAX_LIN_VEL_Z * self.episode_len_sec

        MAX_PITCH_ROLL = np.pi  # Full range

        clipped_pos_xy = np.clip(state[0:2], -MAX_XY, MAX_XY)
        clipped_pos_z = np.clip(state[2], 0, MAX_Z)
        clipped_rp = np.clip(state[7:9], -MAX_PITCH_ROLL, MAX_PITCH_ROLL)
        clipped_vel_xy = np.clip(state[10:12], -MAX_LIN_VEL_XY, MAX_LIN_VEL_XY)
        clipped_vel_z = np.clip(state[12], -MAX_LIN_VEL_Z, MAX_LIN_VEL_Z)

        if self.gui:
            self._clip_and_normalize_state_warning(state,
                                                   clipped_pos_xy,
                                                   clipped_pos_z,
                                                   clipped_rp,
                                                   clipped_vel_xy,
                                                   clipped_vel_z
                                                   )

        normalized_pos_xy = clipped_pos_xy / MAX_XY
        normalized_pos_z = clipped_pos_z / MAX_Z
        normalized_rp = clipped_rp / MAX_PITCH_ROLL
        normalized_y = state[9] / np.pi  # No reason to clip
        normalized_vel_xy = clipped_vel_xy / MAX_LIN_VEL_XY
        normalized_vel_z = clipped_vel_z / MAX_LIN_VEL_XY
        normalized_ang_vel = state[13:16] / np.linalg.norm(state[13:16]) if np.linalg.norm(
            state[13:16]) != 0 else state[13:16]

        norm_and_clipped = np.hstack([normalized_pos_xy,
                                      normalized_pos_z,
                                      state[3:7],
                                      normalized_rp,
                                      normalized_y,
                                      normalized_vel_xy,
                                      normalized_vel_z,
                                      normalized_ang_vel,
                                      state[16:20]
                                      ]).reshape(20, )

        return norm_and_clipped

    def _clip_and_normalize_state_warning(self,
                                          state,
                                          clipped_pos_xy,
                                          clipped_pos_z,
                                          clipped_rp,
                                          clipped_vel_xy,
                                          clipped_vel_z,
                                          ):
        """Debugging printouts associated to `_clipAndNormalizeState`.

        Print a warning if values in a state vector is out of the clipping range.

        """
        if not (clipped_pos_xy == np.array(state[0:2])).all():
            print("[WARNING] it", self.step_counter,
                  "in HoverAviary._clipAndNormalizeState(), clipped xy position [{:.2f} {:.2f}]".format(state[0],
                                                                                                        state[1]))
        if not (clipped_pos_z == np.array(state[2])).all():
            print("[WARNING] it", self.step_counter,
                  "in HoverAviary._clipAndNormalizeState(), clipped z position [{:.2f}]".format(state[2]))
        if not (clipped_rp == np.array(state[7:9])).all():
            print("[WARNING] it", self.step_counter,
                  "in HoverAviary._clipAndNormalizeState(), clipped roll/pitch [{:.2f} {:.2f}]".format(state[7],
                                                                                                       state[8]))
        if not (clipped_vel_xy == np.array(state[10:12])).all():
            print("[WARNING] it", self.step_counter,
                  "in HoverAviary._clipAndNormalizeState(), clipped xy velocity [{:.2f} {:.2f}]".format(state[10],
                                                                                                        state[11]))
        if not (clipped_vel_z == np.array(state[12])).all():
            print("[WARNING] it", self.step_counter,
                  "in HoverAviary._clipAndNormalizeState(), clipped z velocity [{:.2f}]".format(state[12]))
