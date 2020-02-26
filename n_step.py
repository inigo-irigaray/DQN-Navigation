import numpy as np
import collections




Experience = collections.namedtuple('Experience', ['state', 'action', 'reward', 'done'])

class ExperienceSource:
    def __init__(self, env, agent, step_count, brain_name):
        """
        Starts experience generator.
        """
        assert isinstance(step_count, int)
        self.agent = agent
        self.env = env
        self.brain_name = brain_name
        self.env_info = self.env.reset(train_mode=True)[self.brain_name]
        self.step_count = step_count
        self.total_rewards = []
        
    def __iter__(self):
        states, agent_states, histories, cur_rewards, cur_steps, env_lens = [], [], [], [], [], []
        obs = self.env_info.vector_observations[0]
        obs_len = len(obs)
        states.append(obs)
        env_lens.append(obs_len)
        for _ in range(obs_len):
            histories.append(collections.deque(maxlen=self.step_count))
            cur_rewards.append(0.0)
            cur_steps.append(0)
            agent_states.append(self.agent.initial_state())
        
        iter_idx = 0
        while True:
            actions = [None] * len(states)
            states_input, states_indices = [], []
            for idx, state in enumerate(states):
                if state is None: #take random action
                    actions[idx] = np.random.randint(self.env.brains[self.brain_name].vector_action_space_size)
                else:
                    states_input.append(state)
                    states_indices.append(idx)

            if states_input:
                states_actions, next_agent_states = self.agent(states_input, agent_states)
                for idx, action in enumerate(states_actions):
                    g_idx = states_indices[idx]
                    actions[g_idx] = action
                    agent_states[g_idx] = next_agent_states[idx]

            self.env_info = self.env.step(actions[0])[self.brain_name]
            next_state_n = [self.env_info.vector_observations[0]] #put as a list to allow iteration in for loop below
            r_n = [self.env_info.rewards[0]]
            is_done_n = [self.env_info.local_done[0]]
                
            for ofs, (action, next_state, r, is_done) in enumerate(zip(actions, next_state_n, r_n, is_done_n)):
                idx = ofs
                state = states[idx]
                history = histories[idx]
                
                cur_rewards[idx] += r
                cur_steps[idx] += 1
                
                if state is not None:
                    history.append(Experience(state=state, action=action, reward=r, done=is_done))
                    
                if len(history) == self.step_count:
                    yield tuple(history)
                
                states[idx] = next_state
                if is_done:
                    if 0 < len(history) < self.step_count:
                        yield tuple(history)
                    
                    while len(history) > 1:
                        history.popleft()
                        yield tuple(history)    
                    self.total_rewards.append(cur_rewards[idx])
                    #self.total_steps.append(cur_steps[idx])
                    cur_rewards[idx] = 0.0
                    cur_steps[idx] = 0
                        
                    self.env_info = self.env.reset(train_mode=True)[self.brain_name]
                    states[idx] = self.env_info.vector_observations[0]
                    #states[idx] = None
                    agent_states[idx] = self.agent.initial_state()
                    history.clear()
    
    def pop_total_rewards(self):
        r = self.total_rewards
        if r:
            self.total_rewards = []
        return r

    
    

ExperienceFirstLast = collections.namedtuple('ExperienceFirstLast', ('state', 'action', 'reward', 'last_state'))


class ExperienceSourceFirstLast(ExperienceSource):
    def __init__(self, env, agent, gamma, step_count, brain_name):
        super(ExperienceSourceFirstLast, self).__init__(env, agent, step_count+1, brain_name)
        self.gamma = gamma
        self.steps = step_count
        
    def __iter__(self):
        """
        Overrides ExperienceSource.__iter__() to pack first and last states instead of the whole transition.
        """
        for exp in super(ExperienceSourceFirstLast, self).__iter__():
            if exp[-1].done and len(exp) <= self.steps:
                last_state = None
                elems = exp
            else:
                last_state = exp[-1].state
                elems = exp[:-1]
            total_reward = 0.0
            for e in reversed(elems):
                total_reward *= self.gamma
                total_reward += e.reward
            yield ExperienceFirstLast(state=exp[0].state, action=exp[0].action, reward=total_reward, last_state=last_state)
                                
