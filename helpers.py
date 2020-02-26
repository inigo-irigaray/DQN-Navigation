import sys
import time
import numpy as np

import torch
import torch.nn.functional as F

from collections import deque




class RewardTracker:
    def __init__(self, writer, stop_reward):
        """
        Creates reward tracker.
        ----------
        Input:
        ----------
            writer: Tensorboard writer.
            stop_reward: Reward threshold that marks the solution to the problem.
        """
        self._writer = writer
        self._stop_reward = stop_reward
        self._reward100 = deque(maxlen=100)

    def __enter__(self):
        self.ts = time.time()
        self.ts_frame = 0
        self.total_rewards = []
        return self

    def __exit__(self, *args):
        self._writer.close()
    def reward(self, reward, frame, epsilon=None):
        """
        Writes data into tensorboard API and checks if the model has reached the stop_reward (solved the problem).
        """
        self.total_rewards.append(reward)
        speed = (frame - self.ts_frame) / (time.time() - self.ts)
        self.ts_frame = frame
        self.ts = time.time()
        mean_reward = np.mean(self.total_rewards[-100:])
        epsilon_str = "" if epsilon is None else ", eps %.2f" % epsilon
        print('reward %.3f' % reward)
        print("%d: done %d games, episode reward %.3f, mean reward-100 %.3f, speed %.2f f/s%s" % (frame, len(self.total_rewards),
                                                                                                 reward, mean_reward, speed,
                                                                                                 epsilon_str))
        sys.stdout.flush()
        if epsilon is not None:
            self._writer.add_scalar("epsilon", epsilon, frame)
        self._writer.add_scalar("speed", speed, frame)
        self._writer.add_scalar("reward_100", mean_reward, frame)
        self._writer.add_scalar("reward", reward, frame)
        if mean_reward > self._stop_reward:
            print("Solved in %d frames!" % frame)
            return True
        return False


    
    
def unpack_batch(batch):
    """
    Unpacks batch of transitions into arrays of states, actions, rewards, dones and last_states.
    """
    states, actions, rewards, dones, last_states = [], [], [], [], []
    for exp in batch:
        state = np.array(exp.state, copy=False)
        states.append(state)
        actions.append(exp.action)
        rewards.append(exp.reward)
        dones.append(exp.last_state is None)
        if exp.last_state is None:
            last_states.append(state)       # the result will be masked anyway
        else:
            last_states.append(np.array(exp.last_state, copy=False))
    return np.array(states, copy=False), np.array(actions), np.array(rewards, dtype=np.float32), \
        np.array(dones, dtype=np.uint8), np.array(last_states, copy=False)




def calc_loss(batch, net, tgt_net, quant, gamma, device="cpu"):
    """
    Calculates total accumulated loss (huber loss for quantile regression) for backpropagation.
    """
    states, actions, rewards, dones, next_states = unpack_batch(batch)
    batch_size = len(batch)
    
    states_v = torch.FloatTensor(states).to(device)
    actions_v = torch.tensor(actions).to(device)
    next_states_v = torch.FloatTensor(next_states).to(device)
    
    theta = net(states_v)
    actions_v = actions_v.unsqueeze(1).expand(-1, 1, quant)
    theta_a = theta.gather(1, actions_v).squeeze(1)

    next_theta = tgt_net(next_states_v) # batch_size * action * quant
    next_actions_v = next_theta.mean(dim=2).max(1)[1] # batch_size
    next_actions_v = next_actions_v.unsqueeze(1).unsqueeze(1).expand(batch_size, 1, quant)
    next_theta_a = next_theta.gather(1, next_actions_v).squeeze(1) # batch_size * quant
    
    rewards = torch.tensor(rewards).to(device)
    dones = torch.tensor(dones).to(device)
    next_theta_a[dones.unsqueeze(1).repeat(1, 5)] = 0.0
    T_theta = rewards.unsqueeze(1) + gamma * next_theta_a# * dones.unsqueeze(1)

    T_theta_tile = T_theta.view(-1, quant, 1).expand(-1, quant, quant)
    theta_a_tile = theta_a.view(-1, 1, quant).expand(-1, quant, quant)

    error_loss = T_theta_tile - theta_a_tile            
    huber_loss = F.smooth_l1_loss(theta_a_tile, T_theta_tile.detach(), reduction='none')
    tau = torch.arange(0.5 * (1 / quant), 1, 1 / quant).view(1, quant)

    loss = (tau.to('cpu') - (error_loss.to('cpu') < 0).float()).abs() * huber_loss.to('cpu')
    loss = loss.mean(dim=2).sum(dim=1).mean()
    
    return loss
