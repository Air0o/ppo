# PPO


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
