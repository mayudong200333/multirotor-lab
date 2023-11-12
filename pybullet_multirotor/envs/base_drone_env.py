import os
import xml.etree.ElementTree as etxml
from enum import Enum

import gymnasium as gym
import numpy as np
import pybullet as p
import pybullet_data


class DroneModel(Enum):
    MULTIROTOR = "multirotor"
    TILTROTOR = "tiltrotor"


class BaseDroneEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self,
                 drone_model: DroneModel = DroneModel.MULTIROTOR,
                 num_drones: int = 1,
                 initial_position=None,
                 freq: int = 240,
                 aggr_phy_steps: int = 1,
                 gui=False,
                 ):

        ## parameters ##
        self.drone_model = drone_model
        self.num_drones = num_drones
        self.freq = freq
        self.timestep = 1 / self.freq
        self.aggr_phy_steps = aggr_phy_steps
        self.step_counter = 0
        self.gui = gui
        self.urdf = self.drone_model.value + ".urdf"
        self.initial_position = initial_position
        self.R = 3 if num_drones != 1 else 0
        self.kf, self.km, self.thrust2weight = self._parseURDFParameters()
        self.max_rpm = np.sqrt((self.thrust2weight * 9.8) / (4 * self.kf))

        ## connect to PyBullet ##
        if self.gui:
            self.client = p.connect(p.GUI)
            p.resetDebugVisualizerCamera(
                cameraDistance=2,
                cameraYaw=-30,
                cameraPitch=-30,
                cameraTargetPosition=[0, 0, 0],
                physicsClientId=self.client
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
            ).transpose().reshape((self.num_drones, 3))
        self.initial_rpys = np.zeros((self.num_drones, 3))

        ## get action and observation space ##
        self.action_space = self._actionSpace()
        self.observation_space = self._observationSpace()

        ## initialize the pybullet env ##
        self._constructDrones()
        self._createMap()

        self._updateKinetic()

    def step(self, action):
        self.last_action = np.reshape(action, (self.num_drones, 4))
        clipped_action = np.reshape(self._preprocessAction(action), (self.num_drones, 4))

        for _ in range(self.aggr_phy_steps):
            for i in range(self.num_drones):
                self._physics(clipped_action[i, :], i)

            p.stepSimulation(physicsClientId=self.client)

        self._updateKinetic()

        obs = self._computeObs()
        reward = self._computeReward()
        done = self._computeDone()
        info = self._computeInfo()

        self.step_counter += self.aggr_phy_steps

        return obs, reward, done, info

    def reset(self):
        self.step_counter = 0
        p.resetSimulation(physicsClientId=self.client)
        self._constructDrones()
        self._createMap()
        self._updateKinetic()
        return self._computeObs()

    def render(self):
        for i in range(self.num_drones):
            print("[INFO] BaseDroneEnv.render() ——— drone {:d}".format(i),
                  "——— x {:+06.2f}, y {:+06.2f}, z {:+06.2f}".format(self.pos[i, 0], self.pos[i, 1], self.pos[i, 2]),
                  "——— velocity {:+06.2f}, {:+06.2f}, {:+06.2f}".format(self.vel[i, 0], self.vel[i, 1], self.vel[i, 2]),
                  "——— roll {:+06.2f}, pitch {:+06.2f}, yaw {:+06.2f}".format(self.rpy[i, 0] * 180 / np.pi,
                                                                              self.rpy[i, 1] * 180 / np.pi,
                                                                              self.rpy[i, 2] * 180 / np.pi),
                  "——— angular velocity {:+06.4f}, {:+06.4f}, {:+06.4f} ——— ".format(self.ang_v[i, 0], self.ang_v[i, 1],
                                                                                     self.ang_v[i, 2]))

    def close(self):
        p.disconnect(physicsClientId=self.client)

    def _parseURDFParameters(self):
        tree = etxml.parse(os.path.dirname(os.path.abspath(__file__)) + '/../assets/multirotor.urdf').getroot()
        kf = float(tree[0].attrib['kf'])
        km = float(tree[0].attrib['km'])
        thrust2weight = float(tree[0].attrib['thrust2weight'])

        return kf, km, thrust2weight

    def _constructDrones(self):
        ## initialize Drones obs ##
        self.pos = np.zeros((self.num_drones, 3))
        self.quat = np.zeros((self.num_drones, 4))
        self.rpy = np.zeros((self.num_drones, 3))
        self.vel = np.zeros((self.num_drones, 3))
        self.ang_v = np.zeros((self.num_drones, 3))

        ## initialize pybullet ##
        p.setGravity(0, 0, -9.8, physicsClientId=self.client)
        p.setRealTimeSimulation(0, physicsClientId=self.client)
        p.setTimeStep(self.timestep, physicsClientId=self.client)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        ## Load drone model ##
        self.plane_id = p.loadURDF("plane.urdf", physicsClientId=self.client)
        self.drone_ids = np.array([p.loadURDF(os.path.dirname(os.path.abspath(__file__)) + "/../assets/" + self.urdf,
                                              basePosition=self.initial_position[i, :],
                                              baseOrientation=p.getQuaternionFromEuler(self.initial_rpys[i, :]),
                                              physicsClientId=self.client)
                                   for i in range(self.num_drones)])

    def _updateKinetic(self):
        for i in range(self.num_drones):
            self.pos[i], self.quat[i] = p.getBasePositionAndOrientation(self.drone_ids[i], physicsClientId=self.client)
            self.rpy[i] = p.getEulerFromQuaternion(self.quat[i])
            self.vel[i], self.ang_v[i] = p.getBaseVelocity(self.drone_ids[i], physicsClientId=self.client)

    def _physics(self, rpm, nth_drone):
        if self.drone_model == DroneModel.MULTIROTOR:
            forces = np.array(rpm ** 2) * self.kf
            torques = np.array(rpm ** 2) * self.km
            z_torque = (-torques[0] + torques[1] - torques[2] + torques[3])
            for i in range(4):
                p.applyExternalForce(self.drone_ids[nth_drone],
                                     i,
                                     forceObj=[0, 0, forces[i]],
                                     posObj=[0, 0, 0],
                                     flags=p.LINK_FRAME,
                                     physicsClientId=self.client)
            p.applyExternalTorque(self.drone_ids[nth_drone],
                                  4,
                                  torqueObj=[0, 0, z_torque],
                                  flags=p.LINK_FRAME,
                                  physicsClientId=self.client
                                  )
        else:
            ## For TILTROTOR, UNCOMPLETED
            pass

    def _createMap(self):
        pass

    def _actionSpace(self):
        return NotImplementedError

    def _observationSpace(self):
        return NotImplementedError

    def _preprocessAction(self, action):
        return NotImplementedError

    def _computeObs(self):
        return NotImplementedError

    def _computeReward(self):
        return NotImplementedError

    def _computeDone(self):
        return NotImplementedError

    def _computeInfo(self):
        return NotImplementedError
