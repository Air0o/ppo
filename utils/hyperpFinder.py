import os
import logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["RL_WARNINGS"] = "0"
logging.disable(logging.INFO)

from itertools import product
import gymnasium as gym
import math
import numpy as np
import torch

from torchrl.envs.libs.gym import GymEnv

from ppo import PPOAgent

from statistics import stdev, mean

from tqdm import tqdm

class HPPFinder:

    def testSettings(self, settings, envName:str):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        env = GymEnv(envName, device=device)

        agent = PPOAgent(
            baseEnv=env,
            hiddenLayers=settings["hidden_layers"],
            hiddenLayerSize=settings["hidden_layer_size"],
            epochs=settings["epochs"],
            totalFrames=settings["total_frames"],
            framesPerBatch=settings["frames_per_batch"],
            subBatchSize=settings["sub_batch_size"],
            entropy=settings["entropy"],
            epsilon=settings["epsilon"],
            gamma=settings["gamma"],
            learningRate=settings["learning_rate"],
            lmbda=settings["lambda"],  
            verbose=0, 
        )

        std = stdev(agent.averageRewards)
        avg = mean(agent.averageRewards)

        valueCoeff = avg - 0.5 * std

        return valueCoeff

    def __init__(self, settings:dict, envName:str):
            
        generator = product(*settings.values())
        
        combinationsFound = {
            "worst": {"value": 10, "setting": {}},
            "best": {"value": -10, "setting": {}}
        }
        
        bar = tqdm(total=math.prod(len(i) for i in settings.values()))
        for combination in generator:
            setting = {}
            for i, key in enumerate(settings.keys()):
                setting[key] = combination[i]

            barDescription = ""
            value = self.testSettings(settings=setting, envName=envName)
            if value < combinationsFound["worst"]["value"]:
                barDescription = f"{value} - Found new worst setting: {setting}"
                combinationsFound["worst"]["value"] = value
                combinationsFound["worst"]["setting"] = setting
            elif value > combinationsFound["best"]["value"]:
                barDescription = f"{value} - Found new best setting: {setting}"
                combinationsFound["best"]["value"] = value
                combinationsFound["best"]["setting"] = setting

            os.system("cls")
            bar.set_description(barDescription)
            bar.update()
            print(f"Worst setting: {combinationsFound["worst"]}\t({combinationsFound["best"]["value"]})")
            print(f"Best setting:  {combinationsFound["best"]}\t({combinationsFound["best"]["value"]})")




if __name__ == "__main__":
    os.system("cls")
    settings = {
        "hidden_layers": [2],
        "hidden_layer_size": [128],
        "epochs": [8],
        "total_frames": [20000],
        "frames_per_batch": [1024],
        "sub_batch_size": [64, 128],
        "entropy": [0.005, 0.01],
        "epsilon": [0.15, 0.2],
        "gamma": [0.95,0.99],
        "learning_rate": [0.0005, 0.001],
        "lambda": [0.95, 0.99],
    }

    finder = HPPFinder(settings, envName="Reacher-v4")
            