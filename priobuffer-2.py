import numpy as np
import collections
import operator
import random

#from baselines.common.segment_tree import SumSegmentTree, MinSegmentTree


class ReplayBuffer:
    def __init__(self, size, exp_source):
        """
        Creates replay buffer.
        ----------
        Input:
        ----------
            size: Max capacity of the buffer. As new entries are added, the old ones are dropped.
            exp_source: Generator of experiences.
        """
        assert isinstance(size, int)
        self._buffer = []
        self._capacity = size
        self._exp_source = iter(exp_source)
        self._next_idx = 0

    def __len__(self):
        return len(self._buffer)

    def _add(self, sample):#obs_t, action, reward, obs_tp1, done):
        """
        Adds transition into the buffer.                
        ----------
        Input:
        ----------
            sample: Transition (s,a,r,s',done).
        """
        #data = (obs_t, action, reward, obs_tp1, done)

        if self._next_idx >= len(self._buffer): #appends data if max capacity not reached yet
            self._buffer.append(sample)
        else:
            self._buffer[self._next_idx] = sample #drops old entry and appends new data if max capacity
        self._next_idx = (self._next_idx + 1) % self._capacity


    def sample(self, batch_size):
        """
        Samples random batches of experience from buffer.
        ----------
        Input:
        ----------
            batch_size: Number of samples to get.
        ----------
        Output:
        ----------
            Randomly selected batches of experience.
        """
        if len(self._buffer) <= batch_size:
            print("There are only %d batches in the experience buffer." % len(self._buffer))
            return self._buffer
        idxes = [random.randint(0, len(self._buffer) - 1) for _ in range(batch_size)]
        return [self._buffer[idx] for idx in idxes]
    
    
    def populate(self, pop_size):
        """
        Populates buffer with samples of experience.
        ----------
        Input:
        ----------
            pop_size: Number of experiences to populate into the buffer.
        """
        for _ in range(pop_size):
            sample = next(self._exp_source)
            self._add(sample)


class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, size, exp_source, alpha):
        """
        Creates prioritized replay buffer.
        ----------
        Input:
        ----------
            size: size: Max capacity of the buffer. As new entries are added, the old ones are dropped.
            exp_source: Generator of experiences.
            alpha: Degree of prioritization, where 0 is no prioritization at all and 1 full prioritization.
        """
        super(PrioritizedReplayBuffer, self).__init__(size, exp_source)
        assert alpha >= 0
        self._alpha = alpha

        it_capacity = 1
        while it_capacity < size:
            it_capacity *= 2

        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 1.0

    def _add(self, *args, **kwargs):
        """
        Overrides _add() internal utility to account for prioritization.
        """
        idx = self._next_idx
        super()._add(*args, **kwargs)
        self._it_sum[idx] = self._max_priority ** self._alpha
        self._it_min[idx] = self._max_priority ** self._alpha

    def _sample_proportional(self, batch_size):
        """
        Adds transition into the buffer.                
        ----------
        Input:
        ----------
            obs_t, action, reward, obs_tp1, done: Transition (s,a,r,s',done).
        """
        res = []
        p_total = self._it_sum.sum(0, len(self._buffer) - 1)
        every_range_len = p_total / batch_size
        for i in range(batch_size):
            mass = random.random() * every_range_len + i * every_range_len
            idx = self._it_sum.find_prefixsum_idx(mass)
            res.append(idx)
        return res

    def sample(self, batch_size, beta):
        """
        Samples batches of experience from buffer and returns their importance weights and idxes.
        ----------
        Input:
        ----------
            batch_size: Number of samples to get.
            beta: To what degree to use importance weights, where 0 is no corrections and 1 full correction.
        ----------
        Output:
        ----------
            weights: Array of importance weights of sampled transitions.
            idxes: Array of buffer idexes of sampled transitions.
        """
        assert beta > 0

        idxes = self._sample_proportional(batch_size)

        weights = []
        samples = []
        p_min = self._it_min.min() / self._it_sum.sum()
        max_weight = (p_min * len(self._buffer)) ** (-beta)


        for idx in idxes:
            p_sample = self._it_sum[idx] / self._it_sum.sum()
            weight = (p_sample * len(self._buffer)) ** (-beta)
            weights.append(weight / max_weight)
            samples.append(self._buffer[idx])
        weights = np.array(weights, dtype=np.float32)
        return samples, weights, idxes

    def update_priorities(self, idxes, priorities):
        """Updates priorities of sampled transitions, sets priority of transition at index idxes[i] in buffer
        to priorities[i].
        ----------
        Input:
        ----------
            idxes: Idxes of sampled transitions
            priorities: Updated priorities of transitions at the sampled idxes.
        """
        assert len(idxes) == len(priorities)
        for idx, priority in zip(idxes, priorities):
            assert priority > 0
            assert 0 <= idx < len(self._buffer)
            self._it_sum[idx] = priority ** self._alpha
            self._it_min[idx] = priority ** self._alpha

            self._max_priority = max(self._max_priority, priority)