from pathlib import Path
import sys
import warnings

warnings.filterwarnings("ignore", message="std\\(\\): degrees of freedom is <= 0")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from trainer import train

def runEnvTest(envName):
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