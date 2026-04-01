# PPO


Proximal Policy Optimization is a policy gradient Reinforcement Learning algorithm that iteratively improves a policy by collecting trajectories, estimating advantages, and optimizing a clipped surrogate objective which prevents updates from straying too far from the current policy, ensuring stable training.

Features:
- Customizable network topology (depth and size)
- Supports Gymnasium environments
- Training on both CPU or CUDA
- Logging via TensorBoard
- Save/Load trained models
- Inference to view training results


# How to use

#### Create Python virtual environment
Run the `utils/initVenv` script and wait for all the packages to get installed.

#### Start training
With the venv active run the following command to start a new training session: 
`python trainer.py [env name]`
This will start a training session using a Gymnasium environment.

#### Custom arguments
You can add the following flags to the script:
`--name [name]` by default the name of the training session will be the current date and time.

`--continue-from [name]` will and resume training of an already existing model in another session.

`--force-device [device name]` will try to use a specific device.

`--config [path]` will load a configuration file (default is `config.yaml`)

`--force` will overwrite the training session if one with the same name already exists.

`--inference-only` will load the model only for inference and open a view of the environment to show results. The model won't be modified in any way.

`--show-result` will show the results of the training sessions once it has ended. It is the equivalent of running the script with `--inference-only` after training has finished.

#### View training statistics
The default path for logs is `logs/[name]`.
Run the following command to start a TensorBoard server to view them: `tensorboard --logdir logs`

The logs contain 4 graphs:
- Average reward
- Average episode length
- Max episode length
- Learning rate

The data is collected and logged at each training iteration, which is every `frames_per_batch` steps.

# Network Configuration

| Parameter | Description |
|---|---|
| `network_topology` | Structure of the neural network, a list of integers representing hidden layer sizes |
| `epochs` | Number of update passes over the same collected batch |
| `total_frames` | Total number of frames/steps in the training session |
| `frames_per_batch` | Steps collected before each policy update; together with `epochs` and `sub_batch_size` determines update frequency |
| `sub_batch_size` | Mini-batch size within each epoch; should divide `frames_per_batch` evenly |
| `entropy` | Entropy bonus coefficient, encourages exploration by penalizing overconfident policies |
| `epsilon` | PPO clip range - controls how much the new policy can deviate from the previous one |
| `gamma` | Discount factor, values close to 1 prioritize long-term returns |
| `learning_rate` | Step size for the optimizer, too high causes instability, too low slows convergence |
| `lambda` | GAE |
