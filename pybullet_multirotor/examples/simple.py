import numpy as np
from time import sleep
from pybullet_multirotor.envs.control_drone_env import ControlDroneEnv

env = ControlDroneEnv()

action = np.array([[15000, 15000, 15000, 15000]])
for i in range(30000):
    if i % 100 == 0:
        obs, reward, done, info = env.step(action)
sleep(60)
env.close()
