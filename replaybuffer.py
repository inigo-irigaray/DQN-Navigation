import numpy as np
import collections
import operator
import random




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
