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

<p align=justify>I implement <b>Noisy Linear</b> layers within the main DQN model that make exploration more efficient by adding noise to the network's weights. This layer is an equivalent of the default PyTorch fully connected (linear) layer but with additional random values sampled from a Gaussian (normal) distribution every time <i>forward()</i> gets called. The parameters <i>µ</i> and <i>σ</i> are stored inside the Noisy Linear layer and trained through backprop.</p>

<p align=justify>Additionally, instead of calculating a mean Q-value to represent the complexity of the whole state and action spaces squeezed into one number, I implement a <b>Quantile Regression Distributional DQN</b> that explicitly models the distribution over returns. Thus, the neural network captures the stochasticity inherent to the value distribution (as opposed to the value function approximated in baseline DQN) by defining it in terms of a number of quantiles predetermined by me, each with equal probability. I used 5 quantiles in my experiment.</p>

<p align=justify>Everything combined, the final neural network model, named RainbowDQN, is divided in two sub modules: self.common and self.fc. This is just purely for future improvements reasons, when I try to implement Dueling DQN as well.</p>

<p align=justify> self.common is a sequence of three default linear layers of shape (37, 128), (256, 256), (256, 256) where 37 is the state space. Each layer is followed by a Rectified Linear Unit non-linearity. The output of self.common is then processed through two Noisy Linear layers in self.fc and a ReLU layer in-between the two. These layers are of shape (256, 512) and (512, 5*4) where 5 is the number of quantiles and 4 the action space.</p> 

### agent.py

<p align=justify>This file creates the agent object responsible for selecting actions. It allows the DQN model to process states, averages over the quantiles for each action and the selects the action through argmax.</p>

### n_step.py

<p align=justify>This file instances an experience generator that facilitates the agent interaction with the environment, collecting rewards and state information. Basic environment exploration was improved with <b>multi-step bootstrapping</b>, where the agent unrolls the same action n-times (2 in my implementation) to speed-up training</p>

### replaybuffer.py

<p align=justify>The experience replay buffer implemented in this model iterates over the experience generator to collect and store experiences and allows the agent to learn from mini-batches randomly chosen from the buffer. The buffer has a size of 100000 and the minibatches size of 64.</p>

### helpers.py



### Navigation.pynb



## Plot of Rewards

<p align=justify> The plots below shows episodic rewards and the mean reward for the past 100 episodes, with the y axis representing reward value and the x axis relative runtime. The goal of obtaining obtaining mean reward_100 > 13.0 was reached after 23 minutes of training in episode 497 with a mean reward of 13.02. The agent kept training thereafter, however I have modified the code in this repository to stop training the moment it accomplishes the goal for future runs.</p>

<p align="center"><img src="https://github.com/inigo-irigaray/DQN-Navigation/blob/master/imgs/reward_plot.png"></p>

<p align="center"><img src="https://github.com/inigo-irigaray/DQN-Navigation/blob/master/imgs/reward100_plot.png"></p>

## Future Work
