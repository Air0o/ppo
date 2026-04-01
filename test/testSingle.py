import gymnasium as gym

from test import runEnvTest

print("Running single test...")
print(runEnvTest(list(gym.envs.registry.keys())[0]))