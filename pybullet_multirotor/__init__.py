from gym.envs.registration import register

register(
    id = 'control-v0',
    entry_point= 'pybullet_multirotor.envs:ControlDroneEnv'
)