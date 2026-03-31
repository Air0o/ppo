from pathlib import Path
import sys
import warnings
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

warnings.filterwarnings("ignore", message="std\\(\\): degrees of freedom is <= 0")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from trainer import train

import gymnasium as gym

def run_env_test(envName):
    """Run training for a single environment."""
    args = {
        "env_name": envName,
        "continue_from_name": None,
        "name": "test",
        "force": True,
        "resume": False,
        "inference_only": False,
        "force_device": None,
        "verbose": 0,
        "config_path": "test/testConfig.yaml",
        "show_result": False,
    }
    
    try:
        train(args)
        return envName, f"Success:\t{envName}"
    except Exception as e:
        return envName, f"Failed:\t{envName} ({e})"

envNames = list(gym.envs.registry.keys())
results = {}

# Run up to 4 environments in parallel
with ThreadPoolExecutor(max_workers=4) as executor:
    futures = {executor.submit(run_env_test, envName): envName for envName in envNames}
    
    with tqdm(total=len(envNames)) as bar:
        for future in as_completed(futures):
            envName, result = future.result()
            results[envName] = result
            bar.update(1)

for key in results.keys():
    print(f"{key}:\t {results[key]}\n")