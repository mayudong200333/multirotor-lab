import time
import argparse
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO

from pybullet_multirotor.envs.single_drone_rl_hover_env import SingleDroneRlHoverEnv

DEFAULT_GUI = True
DEFAULT_RECORD_VIDEO = False
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_COLAB = False


def run():
    env = gym.make("single-hover-v0")
    print("[INFO] Action space:", env.action_space)
    print("[INFO] Observation space:", env.observation_space)

    model = PPO("MlpPolicy",
                env,
                verbose=1
                )
    model.learn(total_timesteps=10000)

    env = SingleDroneRlHoverEnv()
    obs, info = env.reset(seed=42, options={})
    start = time.time()
    for i in range(3 * env.freq):
        action, _states = model.predict(obs,
                                        deterministic=True
                                        )
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        if terminated:
            obs = env.reset(seed=42, options={})
    env.close()


if __name__ == "__main__":
    run()
