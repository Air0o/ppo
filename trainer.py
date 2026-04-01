import os
import logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["RL_WARNINGS"] = "0"
logging.disable(logging.INFO)

import platform
import gymnasium as gym
import torch
import argparse
import gymnasium_robotics
from utils.configReader import getConfig
from torchrl.envs.libs.gym import GymEnv
from model.ppo import PPOAgent

gym.register_envs(gymnasium_robotics)


def initAgent(config:dict, args:dict, env):
    agent = PPOAgent(
        baseEnv=env,
        config=config,
        args=args,
    )
    return agent


def train(args:dict) -> PPOAgent:
    config = getConfig(args["config_path"])

    if args["force_device"] == "cuda" and not torch.cuda.is_available():
        raise Exception(f"Cannot use device '{args["force_device"]}' because it is unavailable")
    
    env = GymEnv(
        env_name=args["env_name"], 
        device=args["force_device"] if args["force_device"] is not None else "cpu"
    )

    
    agent = initAgent(config=config, env=env, args=args)

    if not args["inference_only"]:
        agent.train()
        agent.save()


    if platform.system() == "Linux":
        print("Skipping inference on Linux")
        return agent

    if not args["show_result"]:
        return agent
    
    env = gym.make(args["env_name"], render_mode="human")

    while True:
        obs, _ = env.reset()
        done = False
        while not done:
            action = agent.inference(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Trainer",
        description="Manager for PPO model training"
    )

    parser.add_argument("env_name")
    parser.add_argument("--name")
    parser.add_argument("--continue-from")
    parser.add_argument("--force-device")
    parser.add_argument("--verbose")
    parser.add_argument("--config")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--inference-only", action="store_true")
    parser.add_argument("--show-result", action="store_true")

    args = parser.parse_args()

    if args.env_name is None:
        print("You need to specify the training environment!")
        exit()

    args = {
        "env_name": args.env_name,
        "continue_from_name": args.continue_from,
        "name": args.name,
        "force": args.force,
        "inference_only": args.inference_only,
        "force_device": args.force_device,
        "verbose": args.verbose,
        "config_path": args.config,
        "show_result": args.show_result,
    }

    train(args)