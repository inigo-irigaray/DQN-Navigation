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



### n_step.py

<b>Multi-step bootstrapping</b>

### replaybuffer.py



### helpers.py


### Navigation.pynb



## Plot of Rewards



## Future Work
