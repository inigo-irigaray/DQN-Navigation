import torch
import numpy as np




class Agent:
    def __init__(self, dqn_model, device="cpu"):
        """
        Agent that calculates Q values from observations and  converts them into actions using argmax.
        """
        self.dqn_model = dqn_model
        self.device = device
    
    def initial_state(self):
        """
        Creates initial empty state for the agent.
        """
        return None

    @torch.no_grad()
    def __call__(self, states, agent_states=None):
        if agent_states is None:
            agent_states = [None] * len(states)
        
        #process states to convert them into variable
        if len(states) == 1:
            np_states = np.expand_dims(states[0], 0)
        else:
            np_states = np.array([np.array(s, copy=False) for s in states], copy=False)
        states = torch.FloatTensor(np_states)
        if torch.is_tensor(states):
            states = states.to(self.device)
            
        theta = self.dqn_model(states)
        Q = theta.mean(dim=2, keepdim=True)
        Q = Q.data.cpu().numpy()
        actions = np.argmax(Q, axis=1)
        return actions, agent_states
