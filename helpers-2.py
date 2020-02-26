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
        print("%d: done %d games, mean reward %.3f, speed %.2f f/s%s" % (frame, len(self.total_rewards), mean_reward,
                                                                         speed, epsilon_str))
        sys.stdout.flush()
        if epsilon is not None:
            self._writer.add_scalar("epsilon", epsilon, frame)
        self._writer.add_scalar("speed", speed, frame)
        self._writer.add_scalar("reward_100", mean_reward, frame)
        self._writer.add_scalar("reward", reward, frame)
        self._reward100.append(mean_reward)
        if np.mean(self._reward100) > self._stop_reward:
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


def distr_projection(next_distr, rewards, dones, Vmin, Vmax, N_ATOMS, GAMMA):
    """
    Approximates the probability distribution of states.
    ----------
    Input:
    ----------
        next_distr: Distribution of states.
        rewards: Transition rewards.
        dones: Marks where if the episode is finished.
        Vmin, Vmax: Range of values.
        N_ATOMS: Number of atoms for the distribution.
        GAMMA: Gamma for reward discounting.
    ----------
    Output:
    ----------
        proj_distr: Approximation to probability distribution with N_ATOMS nodes.
    """
    batch_size = len(rewards)
    proj_distr = np.zeros((batch_size, N_ATOMS), dtype=np.float32)
    delta_z = (Vmax - Vmin) / (N_ATOMS - 1)
    
    for atom in range(N_ATOMS):
        tz_j = np.minimum(Vmax, np.maximum(Vmin, rewards + (Vmin + atom * delta_z) * GAMMA))
        #tz_j = np.maximum(Vmin, np.minimum(Vmax, rewards + (Vmax + atom * delta_z) * GAMMA)) also valid
        b_j = (tz_j - Vmin) / delta_z #where the projected sample lies (on which atom)
        l = np.floor(b_j).astype(np.int64)
        u = np.ceil(b_j).astype(np.int64)
        
        eq_mask = u == l
        proj_distr[eq_mask, l[eq_mask]] += next_distr[eq_mask, atom]
        
        ne_mask = u != l
        proj_distr[ne_mask, l[ne_mask]] += next_distr[ne_mask, atom] * (u - b_j)[ne_mask]
        proj_distr[ne_mask, u[ne_mask]] += next_distr[ne_mask, atom] * (b_j - l)[ne_mask]
        
    if dones.any():
        proj_distr[dones] = 0.0
        tz_j = np.minimum(Vmax, np.maximum(Vmin, rewards[dones]))
        b_j = (tz_j - Vmin) / delta_z
        l = np.floor(b_j).astype(np.int64)
        u = np.ceil(b_j).astype(np.int64)
        
        eq_mask = u == l
        eq_dones = dones.copy()
        eq_dones[dones] = eq_mask
        if eq_dones.any():
            proj_distr[dones, l] = 1.0
            
        ne_mask = u != l
        ne_dones = dones.copy()
        ne_dones[dones] = ne_mask
        if ne_dones.any():
            proj_distr[ne_dones, l] = (u - b_j)[ne_mask]
            proj_distr[ne_dones, u] = (b_j - l)[ne_mask]
            
    return proj_distr


"""def calc_loss(batch, batch_weights, net, tgt_net, Vmin, Vmax, N_ATOMS, gamma, device="cpu"):
    ""
    Calculates total accumulated loss for backpropagation and individual loss values for every batch to update
    priorities in the replay buffer.
    ""
    states, actions, rewards, dones, next_states = unpack_batch(batch)
    batch_size = len(batch)
    
    states_v = torch.tensor(states).to(device)
    actions_v = torch.tensor(actions).to(device)
    next_states_v = torch.tensor(next_states).to(device)
    batch_weights_v = torch.tensor(batch_weights).to(device)
    
    distr_v, qvals_v = net.both(torch.cat((states_v, next_states_v)))
    next_qvals_v = qvals_v[batch_size:]
    distr_v = distr_v[:batch_size]
    
    next_actions_v = next_qvals_v.max(1)[1]
    next_distr_v = tgt_net(next_states_v)
    next_best_distr_v = next_distr_v[range(batch_size), next_actions_v.data]
    next_best_distr_v = tgt_net.apply_softmax(next_best_distr_v)
    next_best_distr = next_best_distr_v.data.cpu().numpy()
    
    dones = dones.astype(np.bool)
    proj_distr = distr_projection(next_best_distr, rewards, dones, Vmin, Vmax, N_ATOMS, gamma)
    
    state_action_values = distr_v[range(batch_size), actions_v.data]
    state_log_sm_v = F.log_softmax(state_action_values, dim=1)
    
    proj_distr_v = torch.tensor(proj_distr).to(device)
    loss_v = -state_log_sm_v * proj_distr_v
    loss_v = batch_weights_v * loss_v.sum(dim=1)
    
    return loss_v.mean(), loss_v + 1e-5"""


def calc_loss(batch, net, tgt_net, quant, gamma, device="cpu"):
    """
    Calculates total accumulated loss for backpropagation and individual loss values for every batch to update
    priorities in the replay buffer.
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


"""
import gym


class ImageToPyTorch(gym.ObservationWrapper):
    def __init__(self, env):
        ""
        Wrapper adapts image format from HWC to CHW as required by PyTorch.
        ""
        super(ImageToPyTorch, self).__init__(env)
        old_shape = self.observation_space.shape
        self.obervation_space = gym.spaces.Box(low=0, high=255, shape=(old_shape[-1], old_shape[1], old_shape[2]),
                                              dtype=np.float32)
        
    def observation(self, observation):
        return np.moveaxis(observation, 2, 0)

"""