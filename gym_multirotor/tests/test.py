import pybullet as p
from time import sleep



p.connect(p.GUI)
p.setGravity(0, 0, -10)
multirotor = p.loadURDF('../assets/multirotor.urdf')

sleep(60)