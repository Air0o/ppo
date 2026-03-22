import os
import shutil

import json

import torch
from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from torch import nn

from utils.paramCalculator import getParams

from math import inf

from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.envs import Compose, DoubleToFloat, ObservationNorm, StepCounter, TransformedEnv

import gymnasium as gym
from torchrl.envs.utils import check_env_specs, ExplorationType, set_exploration_type
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator, OneHotCategorical
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from tqdm import tqdm

from tensordict import TensorDict

from datetime import datetime 

import tensorflow as tf

def make_mlp(hidden_size: int, out_size: int, num_layers: int, device: str, pyramid: bool | None = False) -> nn.Sequential:
    currentSize = hidden_size
    layers = []
    for _ in range(num_layers):
        layers.append(nn.LazyLinear(currentSize, device=device))
        layers.append(nn.Tanh())
        if pyramid:
            currentSize //= 2
    layers.append(nn.LazyLinear(out_size, device=device))
    return nn.Sequential(*layers)

class PPOAgent:
    def __init__(self, baseEnv: gym.Env, settings:dict, args:dict):
        self.name = datetime.now().strftime("%Y-%m-%d-%H-%M-%S") if args["name"] is None else args["name"]
        self.continueFrom = args["continue_from_name"]
        self.verbose = int(args["verbose"]) if args["verbose"] is not None else 2
        self.device = args["force_device"] if args["force_device"] is not None else "cpu"
        self.savePath = f"saves/{self.name}"
        self.logPath = f"logs/{self.name}"
        self.startingStep = 0
        self.maxGradNorm = 1.0
        self.executedFrames = 0

        self.epochs = settings["epochs"]
        self.framesPerBatch = settings["frames_per_batch"]
        self.subBatchSize = settings["sub_batch_size"]
        self.totalFrames = settings["total_frames"]
        hiddenLayerSize = settings["hidden_layer_size"]
        hiddenLayers = settings["hidden_layers"]
        learningRate = settings["learning_rate"]
        lmbda = settings["lambda"]
        entropy = settings["entropy"]
        epsilon = settings["epsilon"]
        gamma = settings["gamma"]
        force = args["force"]
        resume = args["resume"]

        self.isDiscrete = isinstance(baseEnv.action_space, gym.spaces.Discrete)

        if resume:
            self.continueFrom = self.name

        try:
            if os.path.exists(self.savePath):
                if not force and self.continueFrom != self.name:
                    print("A model with the same name already exists! If you want to overwrite it use --force")
                    exit()
                elif self.continueFrom != self.name:
                    shutil.rmtree(self.logPath, ignore_errors=True)
                    shutil.rmtree(self.savePath)
            os.makedirs(self.savePath)
        except Exception as e:
            print(e)

        self.env = TransformedEnv(
            baseEnv,
            Compose(
                ObservationNorm(in_keys=["observation"]),
                DoubleToFloat(),
                StepCounter(),
            ),
        )

        if self.continueFrom is not None:
            self._load()
        else:
            self.env.transform[0].init_stats(num_iter=1000, reduce_dim=0, cat_dim=0)
            self.valueNet = make_mlp(
                hiddenLayerSize, 
                1, 
                hiddenLayers, 
                self.device,
                pyramid=settings["pyramid"],
            )

        check_env_specs(self.env)
        obs_size = self.env.observation_spec["observation"].shape[-1]
        dummy = torch.zeros(1, obs_size, device=self.device)

        if self.isDiscrete:
            n_actions = baseEnv.action_space.n
            if self.continueFrom is None:
                self.actorNet = make_mlp(
                    hiddenLayerSize, 
                    n_actions, 
                    hiddenLayers, 
                    self.device,
                    pyramid=settings["pyramid"],
                )
                self.actorNet(dummy)
                self.valueNet(dummy)
            else:
                self.actorNet = torch.load(f"saves/{self.continueFrom}/actor.pt", weights_only=False)

            self.policy_module = TensorDictModule(
                self.actorNet,
                in_keys=["observation"],
                out_keys=["logits"],       # Categorical reads "logits"
            )
            self.policy_module = ProbabilisticActor(
                module=self.policy_module,
                spec=self.env.action_spec,
                in_keys=["logits"],
                distribution_class=OneHotCategorical,   # <-- was Categorical
                return_log_prob=True,
            )

        else:
            if self.continueFrom is None:
                self.actorNet = make_mlp(
                    hiddenLayerSize,
                    2 * self.env.action_spec.shape[-1],  # loc + scale
                    hiddenLayers,
                    self.device,
                    pyramid=settings["pyramid"],
                )
                self.actorNet.append(NormalParamExtractor())
                self.actorNet(dummy)
                self.valueNet(dummy)
            else:
                self.actorNet = torch.load(f"saves/{self.continueFrom}/actor.pt", weights_only=False)


            self.policy_module = TensorDictModule(
                self.actorNet,
                in_keys=["observation"],
                out_keys=["loc", "scale"],
            )
            self.policy_module = ProbabilisticActor(
                module=self.policy_module,
                spec=self.env.action_spec,
                in_keys=["loc", "scale"],
                distribution_class=TanhNormal,
                distribution_kwargs={
                    "low": self.env.action_spec.space.low,
                    "high": self.env.action_spec.space.high,
                },
                return_log_prob=True,
            )

        value_module = ValueOperator(
            module=self.valueNet, 
            in_keys=["observation"]
        )

        self.collector = SyncDataCollector(
            self.env,
            self.policy_module,
            frames_per_batch=self.framesPerBatch,
            total_frames=self.totalFrames,
            split_trajs=False,
            device=self.device,
        )
        

        self.replayBuffer = ReplayBuffer(
            storage=LazyTensorStorage(max_size=self.framesPerBatch),
            sampler=SamplerWithoutReplacement(),
            pin_memory=self.device == "cuda"
        )

        self.advantageModule = GAE(
            gamma=gamma, lmbda=lmbda, value_network=value_module, average_gae=True
        )

        #saved
        self.lossModule = ClipPPOLoss(
            actor_network=self.policy_module,
            critic_network=value_module,
            clip_epsilon=epsilon,
            entropy_bonus=bool(entropy),
            entropy_coeff=entropy,
            critic_coeff=1.0,
            loss_critic_type="smooth_l1",
        )

        #saved
        self.optim = torch.optim.Adam(self.lossModule.parameters(), learningRate)

        tMax = self.loaded_t_max if self.continueFrom is not None and self.loaded_t_max is not None else self.totalFrames // self.framesPerBatch
        #saved
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optim, tMax, 0.0
        )

        if self.continueFrom is not None:
            self.optim.load_state_dict(torch.load(f"saves/{self.continueFrom}/optim.pt"))
            self.scheduler.load_state_dict(torch.load(f"saves/{self.continueFrom}/scheduler.pt"))
            self.lossModule.load_state_dict(torch.load(f"saves/{self.continueFrom}/loss_module.pt"))
        


        if self.verbose != 0:
            print(f"Action spec: {self.env.action_spec}")
            print(f"Observation spec: {self.env.observation_spec}")
            print(f"Initialized PPO agent on device '{self.device}'")
            print(f"Total steps: {self.totalFrames}")
            print(f"Steps per batch: {self.framesPerBatch}")
            print(f"Steps per mini batch: {self.subBatchSize}")
            print(f"Action type: {"discrete" if self.isDiscrete else "continuous"}")
            print(f"Hidden layers: {hiddenLayers}")
            print(f"Pyramid: {settings["pyramid"]}")
            print("Layers sizes: ")
            current = hiddenLayerSize
            for i in range(hiddenLayers):
                print(f"{current} {"- " if i != (hiddenLayers-1) else "\n"}", end="")
                if settings["pyramid"]:
                    current //= 2
    
    
    def train(self):
        try:
            if self.verbose != 0:
                pbar = tqdm(
                    total=self.totalFrames,
                    ncols=0 if self.verbose == 1 else None
                )
                fileWriter = tf.summary.create_file_writer(self.logPath)
                fileWriter.set_as_default()

            lastReward = -inf

            for i, tensordict_data in enumerate(self.collector):
                if i % 25 == 0:
                    self.save()

                self.advantageModule(tensordict_data)
                data_view = tensordict_data.reshape(-1)
                self.replayBuffer.extend(data_view.cpu())
                for _ in range(self.epochs):
                    for _ in range(self.framesPerBatch // self.subBatchSize):
                        self.optim.zero_grad(set_to_none=True)
                        subdata = self.replayBuffer.sample(self.subBatchSize)
                        subdata = subdata.to(self.device, non_blocking=True)
                        loss_vals = self.lossModule(subdata)
                        loss_value = (
                            loss_vals["loss_objective"]
                            + loss_vals["loss_critic"]
                            + loss_vals["loss_entropy"]
                        )
                        loss_value.backward()
                        torch.nn.utils.clip_grad_norm_(self.lossModule.parameters(), self.maxGradNorm)
                        self.optim.step()


                rewardData = tensordict_data["next", "reward"].mean().item()
                maxEpLenthData = tensordict_data["next", "step_count"].max().item()
                avgEpLenthData = tensordict_data["next", "step_count"].float().mean().item()
                lrData = self.optim.param_groups[0]["lr"]

                saveStr = ""
                if rewardData > lastReward and i % 2 == 0:
                    lastReward = rewardData
                    self.save()
                    saveStr = f"Saved agent to '{self.savePath}'"

                numel = tensordict_data.numel()
                if self.verbose != 0:
                    step = self.startingStep + numel * i
                    tf.summary.scalar("Average reward", data = rewardData, step=step)
                    tf.summary.scalar("Max episode length", data = maxEpLenthData, step=step)
                    tf.summary.scalar("Average episode length", data = avgEpLenthData, step=step)
                    tf.summary.scalar("Learning rate", data = lrData, step=step)

                    if self.verbose == 2:
                        cum_reward_str = f"average reward={rewardData: 4.4f}"
                        stepcount_str = f"Max episode length: {maxEpLenthData}"
                        lr_str = f"lr:{lrData: 4.4f}"
                        pbar.set_description(", ".join([cum_reward_str, stepcount_str, lr_str, saveStr]))
                        
                    pbar.update(numel)
                    
                self.scheduler.step()
                self.executedFrames += numel

        except KeyboardInterrupt:
            print("Keyboard interrupt")

    def inference(self, obs):
        if isinstance(obs, dict):
            obs = obs["observation"]

        with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
            td = TensorDict(
                {"observation": torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)},
                batch_size=[1],
                device=self.device,
            )
            for t in self.env.transform:
                if isinstance(t, StepCounter):
                    continue
                            # Skip transforms that require keys not present in our minimal TensorDict
                if hasattr(t, "in_keys") and not all(td.get(k) is not None for k in t.in_keys):
                    continue
                td = t(td)
                
            td["observation"] = td["observation"].float()
            td = self.policy_module(td)

        action = td["action"].squeeze(0).cpu()
        if self.isDiscrete:
            return action.argmax().item() 
        return action.numpy()
    
    def save(self):
        torch.save(self.actorNet, self.savePath + "/actor.pt")
        torch.save(self.valueNet, self.savePath + "/value.pt")
        stats = {
            "loc": self.env.transform[0].loc,
            "scale": self.env.transform[0].scale,
        }
        torch.save(stats, self.savePath + "/norm_stats.pt")
        torch.save(self.optim.state_dict(), self.savePath + "/optim.pt")
        torch.save(self.scheduler.state_dict(), self.savePath + "/scheduler.pt")
        torch.save(self.lossModule.state_dict(), self.savePath + "/loss_module.pt")

        with open(f"{self.savePath}/stats.json", "w") as file:
            data = {
                "total_steps": self.executedFrames + self.startingStep,
                "scheduler_t_max": self.scheduler.T_max,
            }
            json.dump(data, file)

    def _load(self):
        with open(f"saves/{self.continueFrom}/stats.json") as file:
            data = json.load(file)
            self.startingStep = data["total_steps"]
            self.loaded_t_max = data["scheduler_t_max"]

        stats = torch.load(f"saves/{self.continueFrom}/norm_stats.pt", weights_only=True)
        self.env.transform[0].loc = stats["loc"]
        self.env.transform[0].scale = stats["scale"]

        self.valueNet = torch.load(f"saves/{self.continueFrom}/value.pt", weights_only=False)
        if self.verbose != 0:
            print(f"Loaded agent from 'saves/{self.continueFrom}'")

