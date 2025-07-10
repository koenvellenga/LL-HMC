import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample

import torch
import torch.nn as nn
import torch.nn.functional as F

from copy import deepcopy
import numpy as np

class LL_BNN(PyroModule):
    def __init__(self, out_dim=2, hid_dim=768, prior_scale=50.,
                 pretrained_weights=None, pretrained_biases=None):
        super().__init__()
        assert out_dim > 0 and hid_dim > 0
        self.layer_sizes = [hid_dim] + [out_dim]
        
        layer_list = [
            PyroModule[nn.Linear](self.layer_sizes[idx - 1], self.layer_sizes[idx]) 
            for idx in range(1, len(self.layer_sizes))
        ]
        self.layers = PyroModule[nn.ModuleList](layer_list)

        # Initialize weights and biases for each layer
        for layer_idx, layer in enumerate(self.layers):
            if pretrained_weights is not None and pretrained_biases is not None:
                layer.weight = PyroSample(dist.Normal(pretrained_weights, prior_scale * np.sqrt(2 / self.layer_sizes[layer_idx]))
                                          .expand([self.layer_sizes[layer_idx + 1], self.layer_sizes[layer_idx]])
                                          .to_event(2))
                layer.bias = PyroSample(dist.Normal(pretrained_biases, prior_scale)
                                        .expand([self.layer_sizes[layer_idx + 1]])
                                        .to_event(1))
             
            else:
                layer.weight = PyroSample(dist.Normal(0., prior_scale * np.sqrt(2 / self.layer_sizes[layer_idx]))
                                          .expand([self.layer_sizes[layer_idx + 1], self.layer_sizes[layer_idx]])
                                          .to_event(2))
                layer.bias = PyroSample(dist.Normal(0., prior_scale)
                                        .expand([self.layer_sizes[layer_idx + 1]])
                                        .to_event(1))

    def forward(self, x, y=None):
        x = x.float() 
        logits = self.layers[-1](x)
        probs = F.softmax(logits, dim=-1)  
        with pyro.plate("data", x.shape[0]):
            obs = pyro.sample("obs", dist.Categorical(probs=probs), obs=y)
        return probs

class VmappedLinearLayer(torch.nn.Module):
    def __init__(self, weight_samples, bias_samples):
        super(VmappedLinearLayer, self).__init__()
        self.weight_samples = weight_samples
        self.bias_samples = bias_samples
    
    @staticmethod
    def linear(weight, bias, x):
        return torch.nn.functional.linear(x, weight, bias)
        
    def forward(self, x):
        outputs = torch.vmap(self.linear, in_dims=(0, 0, None))(self.weight_samples, self.bias_samples, x)
        return outputs
