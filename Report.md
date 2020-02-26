# Improved DQN Implementation Report

## Hyperparameters

    GAMMA = 0.99
    LEARNING_RATE = 1e-4
    BATCH_SIZE = 64
    STOP_REWARD = 13.0
    SEED = 1

    TARGET_NET_SYNC = 1000

    STEP_COUNTS = 2

    REPLAY_SIZE = 100000
    REPLAY_INITIAL = 10000

    QUANT = 5

## Algorithms

### model.py

<p align=justify>The main machine learning algorithm of this implementation is Deep Q Networks (<b>DQN</b>). This approach utilizes neural networks to calculate state-action values (or Q-values) to train a reinforcement learning agent to maximize reward within an environment.</p>

<p align=justify>To enable progress and discover new possible better strategies, one must balance optimization of the current policy with exploration of new ones. While this has typically been done by introducing propabilistically induced exploration through a paremeter epsilon (0,1) , where an agent behaves randomly epsilon times and follows the policy 1-epsilon times, I use a different approach.</p>

<p align=justify>I implement <b>Noisy Linear</b> layers within the main DQN model that make exploration more efficient by adding noise to the network's weights. 

### agent.py



### n_step.py

<b>Multi-step bootstrapping</b>

### replaybuffer.py



### helpers.py


### Navigation.pynb



## Plot of Rewards



## Future Work
