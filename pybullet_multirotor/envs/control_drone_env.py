import numpy as np
from gym import spaces

from pybullet_multirotor.envs.base_drone_env import BaseDroneEnv,DroneModel

class ControlDroneEnv(BaseDroneEnv):

    def __init__(self,
                 drone_model: DroneModel = DroneModel.MULTIROTOR,
                 num_drones: int = 1,
                 initial_position=None,
                 freq: int = 240,
                 aggr_phy_steps: int = 1,
                 gui=True,
                 ):
        super().__init__(drone_model=drone_model,
                         num_drones=num_drones,
                         initial_position=initial_position,
                         freq=freq,
                         aggr_phy_steps=aggr_phy_steps,
                         gui=gui)


    def _actionSpace(self):
        if self.drone_model == DroneModel.MULTIROTOR:
            act_lower_bound = np.array([[0.]*4 for _ in range(self.num_drones)])
            act_upper_bound = np.array([[self.max_rpm]*4 for _ in range(self.num_drones)])
        else:
            return NotImplementedError

        return spaces.Box(low=act_lower_bound,high=act_upper_bound,dtype=np.float32)

    def _observationSpace(self):
        if self.drone_model == DroneModel.MULTIROTOR:
            ## Observation vector: [X,Y,Z,R,P,Y]
            obs_lower_bound = np.array([[-np.inf,-np.inf,0.,-np.pi,-np.pi,-np.pi] for _ in range(self.num_drones)])
            obs_upper_bound = np.array([[np.inf,np.inf,0.,np.pi,np.pi,np.pi] for _ in range(self.num_drones)])
        else:
            return NotImplementedError

        return spaces.Box(low=obs_lower_bound,high=obs_upper_bound,dtype=np.float32)

    def _computeObs(self):
        obs = np.hstack([self.pos,self.rpy])

        return obs

    def _preprocessAction(self, action):
        clipped_action = np.zeros((self.num_drones,4))
        for i in range(self.num_drones):
            clipped_action[i,:] = np.clip(action[i,:],0,self.max_rpm)

        return clipped_action

    def _computeReward(self):
        return -1

    def _computeDone(self):
        return False

    def _computeInfo(self):
        return {"GOOD"}





