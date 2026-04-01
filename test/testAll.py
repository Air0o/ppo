from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


from test import runEnvTest

import gymnasium as gym

envNames = list(gym.envs.registry.keys())
results = {}

# Run up to 4 environments in parallel
with ThreadPoolExecutor(max_workers=4) as executor:
    futures = {executor.submit(runEnvTest, envName): envName for envName in envNames}
    
    with tqdm(total=len(envNames), desc="Running complete test") as bar:
        for future in as_completed(futures):
            envName, result = future.result()
            results[envName] = result
            bar.update(1)

for key in results.keys():
    print(f"{key}:\t {results[key]}\n")