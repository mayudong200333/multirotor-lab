from enum import Enum
import os
import gym
import numpy as np
import pybullet as p
import pybullet_data

class DroneModel(Enum):
    MULTIROTOR = "multirotor"
    TILTROTOR = "tiltrotor"

class BaseDroneEnv(gym.Env):
    metadata = {'render.modes':['human']}
    
    def __init__(self,
                 drone_model:DroneModel = DroneModel.MULTIROTOR,
                 num_drones:int = 1,
                 initial_position = None,
                 freq:int = 240,
                 aggr_phy_steps:int=1,
                 gui = False,
                 ):

        ## parameters ##
        self.drone_model = drone_model
        self.num_drones = num_drones
        self.freq = freq
        self.timestep = 1/self.freq
        self.aggr_phy_steps = aggr_phy_steps
        self.gui = gui
        self.urdf = self.drone_model.value + ".urdf"
        self.initial_position = initial_position
        self.R = 3 if num_drones != 1 else 0

        ## connect to PyBullet ##
        if self.gui:
            self.client = p.connect(p.GUI)
            p.resetDebugVisualizerCamera(
                cameraDistance = 1,
                cameraYaw = -30,
                cameraTargetPosition = [0,0,0],
                physicsClientId = self.client
            )
        else:
            self.client = p.connect(p.DIRECT)

        ## set initial poses ##
        if self.initial_position is None:
            self.initial_position = np.vstack(
                (
                    np.array([self.R * np.cos(2 * np.pi / self.num_drones * i) for i in range(self.num_drones)]),
                    np.array([self.R * np.sin(2 * np.pi / self.num_drones * i) for i in range(self.num_drones)]),
                    [0] * self.num_drones
                )
            ).transpose().reshape((self.num_drones,3))
        self.initial_rpys = np.zeros((self.num_drones,3))

        ## get action and observation space ##
        self.action_space = self._actionSpace()
        self.observation_space = self._observationSpace()

        ## initialize the pybullet env ##
        self._constructDrones()
        self._createMap()

        ## update and stores the
    def step(self,action):
        self.last_action = np.reshape(action,(self.num_drones,4))
        clipped_action = np.reshape(self._preprocessAction(action),(self.num_drones,4))


    
    def reset(self):
        p.resetSimulation(physicsClientId=self.client)
        self._constructDrones()
        self._createMap()
        self._updateKinetic()
        return self._computeObs()
    
    def render(self):
        pass 
    
    def close(self):
        pass

    def _constructDrones(self):
        ## initialize Drones obs ##
        self.pos = np.zeros((self.num_drones,3))
        self.quat = np.zeros((self.num_drones,4))
        self.rpy = np.zeros((self.num_drones,3))
        self.vel = np.zeros((self.num_drones,3))
        self.ang_v = np.zeros((self.num_drones,3))

        ## initialize pybullet ##
        p.setGravity(0,0,-9.8,physicsClientId=self.client)
        p.setRealTimeSimulation(0,physicsClientId=self.client)
        p.setTimeStep(self.timestep,physicsClientId=self.client)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        ## Load drone model ##
        self.plane_id = p.loadURDF("plane.urdf",physicsClientId=self.client)
        self.drone_ids = np.array([p.loadURDF(os.path.dirname(os.path.abspath(__file__))+"/../assets/"+self.urdf,
                                              basePosition=self.initial_position[i,:],
                                              baseOrientation=p.getQuaternionFromEuler(self.initial_rpys),
                                              physicsClientId=self.client)
                                   for i in range(self.num_drones)])

    def _updateKinetic(self):
        for i in range(self.num_drones):
            self.pos[i],self.quat[i] = p.getBasePositionAndOrientation(self.drone_ids[i],physicsClientId = self.client)
            self.rpy[i] = p.getEulerFromQuaternion(self.quat[i])
            self.vel[i],self.ang_v[i] = p.getBaseVelocity(self.drone_ids[i],physicsClientId = self.client)

    def _createMap(self):
        pass

    def _actionSpace(self):
        return NotImplementedError

    def _observationSpace(self):
        return NotImplementedError

    def _computeObs(self):
        return NotImplementedError

    def _preprocessAction(self):
        return NotImplementedError

if __name__ == '__main__':
    pass