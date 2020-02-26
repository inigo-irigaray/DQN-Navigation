import math
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class NoisyLinear(nn.Linear):
    """
    Allows the agent to efficiently explore the environment by adding noise to the weights of the fully-connected
    layers of the DQN and adjusting the parameters of the noise during training with backprop.
    """
    def __init__(self, in_shape, out_shape, sigma_init=0.017, bias=True):
        super(NoisyLinear, self).__init__(in_shape, out_shape, bias=bias)
        self.sigma_weight = nn.Parameter(torch.full((out_shape, in_shape), sigma_init))
        self.register_buffer("epsilon_weight", torch.zeros(out_shape, in_shape))
        if bias:
            self.sigma_bias = nn.Parameter(torch.full((out_shape,), sigma_init))
            self.register_buffer("epsilon_bias", torch.zeros(out_shape))
        self._reset_parameters()
        
    def _reset_parameters(self):
        std = math.sqrt(3 / self.in_features)
        self.weight.data.uniform_(-std, std)
        self.bias.data.uniform_(-std, std)
        
    def forward(self, x):
        self.epsilon_weight.normal_()
        bias = self.bias
        if bias is not None:
            self.epsilon_bias.normal_()
            bias = bias + self.sigma_bias * self.epsilon_bias
        return F.linear(x, self.weight + self.sigma_weight * self.epsilon_weight, bias)


class RainbowDQN(nn.Module):
    def __init__(self, in_shape, n_actions, quant):
        super(RainbowDQN, self).__init__() 
        self.quant = quant
        self.n_actions = n_actions
        self.common = nn.Sequential(nn.Linear(in_shape, 128),
                                 nn.ReLU(),
                                 nn.Linear(128, 256),
                                 nn.ReLU(),
                                 nn.Linear(256, 256),
                                 nn.ReLU())
 
        
        self.fc = nn.Sequential(NoisyLinear(256, 512),
                                   nn.ReLU(),
                                   NoisyLinear(512, quant * n_actions))
        
        self.t = torch.FloatTensor((2 * np.arange(quant) + 1) / (2.0 * quant)).view(1, -1)
                
    def forward(self, x):
        x = self.common(x)
        x = self.fc(x)
        theta = x.view(-1, self.n_actions, self.quant)
        return theta


class TargetNet:
    def __init__(self, model):
        """
        Creates a separate target net to solve correlation issues while training. Syncs with model net every 'x' time.
        """
        self.model = model
        self.target_model = copy.deepcopy(model)

    def sync(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def alpha_sync(self, alpha):
        """
        Blend target net parameters with model parameters.
        ----------
        Input:
        ----------
            alpha: Importance of target net parameters in future blend with model parameters, where 0 is null, 1 full.
        """
        assert isinstance(alpha, float)
        assert 0.0 < alpha <= 1.0
        state = self.model.state_dict()
        tgt_state = self.target_model.state_dict()
        for k, v in state.items():
            tgt_state[k] = tgt_state[k] * alpha + (1 - alpha) * v
        self.target_model.load_state_dict(tgt_state)


"""class NoisyLinear(nn.Linear):
    ""
    Allows the agent to efficiently explore the environment by adding noise to the weights of the fully-connected
    layers of the DQN and adjusting the parameters of the noise during training with backprop.
    ""
    def __init__(self, in_shape, out_shape, sigma_init=0.017, bias=True):
        super(NoisyLinear, self).__init__(in_shape, out_shape, bias=bias)
        self.sigma_weight = nn.Parameter(torch.full((out_shape, in_shape), sigma_init))
        self.register_buffer("epsilon_weight", torch.zeros(out_shape, in_shape))
        if bias:
            self.sigma_bias = nn.Parameter(torch.full((out_shape,), sigma_init))
            self.register_buffer("epsilon_bias", torch.zeros(out_shape))
        self._reset_parameters()
        
    def _reset_parameters(self):
        std = math.sqrt(3 / self.in_features)
        self.weight.data.uniform_(-std, std)
        self.bias.data.uniform_(-std, std)
        
    def forward(self, x):
        self.epsilon_weight.normal_()
        bias = self.bias
        if bias is not None:
            self.epsilon_bias.normal_()
            bias = bias + self.sigma_bias * self.epsilon_bias
        return F.linear(x, self.weight + self.sigma_weight * self.epsilon_weight, bias)


class RainbowDQN(nn.Module):
    def __init__(self, in_shape, n_actions, N_ATOMS, Vmin, Vmax, DELTA_Z):
        super(RainbowDQN, self).__init__()
        self.N_ATOMS = N_ATOMS
        
        self.common = nn.Sequential(nn.Linear(in_shape, 128),
                                 nn.ReLU(),
                                 nn.Linear(128, 256),
                                 nn.ReLU(),
                                 nn.Linear(256, 256),
                                 nn.ReLU())
        
        self.fc_val = nn.Sequential(NoisyLinear(256, 512),
                                   nn.ReLU(),
                                   NoisyLinear(512, self.N_ATOMS))
        
        self.fc_adv = nn.Sequential(NoisyLinear(256, 512),
                                   nn.ReLU(),
                                   NoisyLinear(512, self.N_ATOMS * n_actions))
        
        self.register_buffer("supports", torch.arange(Vmin, Vmax + DELTA_Z, DELTA_Z))
        
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        batch_size = x.size()[0]
        fx = x.float()
        common_out = self.common(fx).view(batch_size, -1)
        val_out = self.fc_val(common_out).view(batch_size, 1, self.N_ATOMS)
        adv_out = self.fc_adv(common_out).view(batch_size, -1, self.N_ATOMS)
        adv_mean = adv_out.mean(dim=1, keepdim=True)
        return val_out + adv_out - adv_mean
    
    def both(self, x):
        cat_out = self(x)
        probs = self.apply_softmax(cat_out)
        weights = probs * self.supports
        res = weights.sum(dim=2)
        return cat_out, res
    
    def qvals(self, x):
        return self.both(x)[1]
    
    def apply_softmax(self, t):
        return self.softmax(t.view(-1, self.N_ATOMS)).view(t.size())


class TargetNet:
    def __init__(self, model):
        ""
        Creates a separate target net to solve correlation issues while training. Syncs with model net every 'x' time.
        ""
        self.model = model
        self.target_model = copy.deepcopy(model)

    def sync(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def alpha_sync(self, alpha):
        ""
        Blend target net parameters with model parameters.
        ----------
        Input:
        ----------
            alpha: Importance of target net parameters in future blend with model parameters, where 0 is null, 1 full.
        ""
        assert isinstance(alpha, float)
        assert 0.0 < alpha <= 1.0
        state = self.model.state_dict()
        tgt_state = self.target_model.state_dict()
        for k, v in state.items():
            tgt_state[k] = tgt_state[k] * alpha + (1 - alpha) * v
        self.target_model.load_state_dict(tgt_state)
        """
        
 
"""
class VisualRainbowDQN(nn.Module):
    def __init__(self, in_shape, n_actions, N_ATOMS, Vmin, Vmax, DELTA_Z):
        super(RainbowDQN, self).__init__()
        self.N_ATOMS = N_ATOMS
        
        self.conv = nn.Sequential(nn.Conv2d(in_shape[0], 32, kernel_size=8, stride=4),
                                 nn.ReLU(),
                                 nn.Conv2d(32, 64, kernel_size=4, stride=2),
                                 nn.ReLU(),
                                 nn.Conv2d(64, 64, kernel_size=3, stride=1),
                                 nn.ReLU())
        
        conv_out_size = self._get_conv_out(in_shape)
        
        self.fc_val = nn.Sequential(NoisyLinear(conv_out_size, 512),
                                   nn.ReLU(),
                                   NoisyLinear(512, self.N_ATOMS))
        
        self.fc_adv = nn.Sequential(NoisyLinear(conv_out_size, 512),
                                   nn.ReLU(),
                                   NoisyLinear(512, self.N_ATOMS * n_actions))
        
        self.register_buffer("supports", torch.arange(Vmin, Vmax + DELTA_Z, DELTA_Z))
        
        self.softmax = nn.Softmax(dim=1)
    
    def _get_conv_out(shape):
        out = self.conv(torch.zeros(1, *shape))
        return int(np.prod(out.size()))
        
    def forward(self, x):
        batch_size = x.size()[0]
        fx = x.float() / 256
        conv_out = self.conv(fx).view(batch_size, -1)
        val_out = self.fc_val(conv_out).view(batch_size, 1, self.N_ATOMS)
        adv_out = self.fc_adv(conv_out).view(batch_size, -1, self.N_ATOMS)
        adv_mean = adv_out.mean(dim=1, keepdim=True)
        return val_out + adv_out - adv_mean
    
    def both(self, x):
        cat_out = self(x)
        probs = self.apply_softmax(cat_out)
        weights = probs * self.supports
        res = weights.sum(dim=2)
        return cat_out, res
    
    def qvals(self, x):
        return self.both(x)[1]
    
    def apply_softmax(self, t):
        return self.softmax(t.view(-1, self.N_ATOMS)).view(t.size())
"""