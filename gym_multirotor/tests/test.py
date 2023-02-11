import pybullet as p
import os
from time import sleep


client = p.connect(p.GUI)
p.setGravity(0, 0, -10)
p.resetDebugVisualizerCamera(
                cameraDistance = 1,
                cameraYaw = -30,
                cameraPitch = -30,
                cameraTargetPosition = [0,0,0],
                physicsClientId = client
            )
multirotor = p.loadURDF(os.path.dirname(os.path.abspath(__file__))+'/../assets/multirotor.urdf')

sleep(60)