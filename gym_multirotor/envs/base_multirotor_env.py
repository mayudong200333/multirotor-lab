import gym
import numpy as np 
import pybullet as p 

class BaseMultirotorEnv(gym.Env):
    metadata = {'render.modes':['human']}
    
    def __init__(self):
        pass 
    
    def step(self,action):
        pass 
    
    def reset(self):
        pass 
    
    def render(self):
        pass 
    
    def close(self):
        pass