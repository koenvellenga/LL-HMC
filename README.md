# Last layer Hamiltonian Monte Carlo
Implementation for the Last Layer Hamiltonian Monte Carlo paper.

### Example to perform sampling and the evaluation

The following LL-HMC example is for a single chain and expects that the latent representations from the base model are already extracted.
After loading the data, define the last layer dimensions, and run the HMC sampling.
Lastly, we use a model wrapper for the sampled parameters and perform the evaluation.

```
import torch
import numpy as np
import pyro
from pyro.infer import MCMC, NUTS

seed_number = 42 # or any other number
torch.manual_seed(seed_number)
pyro.set_rng_seed(seed_number)
np.random.seed(seed_number)

from .evaluation import evaluate
from .models import LL_BNN, VmappedLinearLayer

# define the dataset and experiment type
dataset = 'aide' # b4c, road
experiment_type = 'regular' # ood_min, ood_max

# Load the extracted latent representations 
X_train = torch.load(f"{dataset}_processed/{experiment_type}_train_embed.pth") # Size: [number of train instances, embedding size]
y_train = torch.load(f"{dataset}_processed/{experiment_type}_train_embed.pth") # Size: [number of train instances,] << sparse labels
X_test = torch.load(f"{dataset}_processed/{experiment_type}_test_embed.pth") # Size: [number of test instances, embedding size]
y_test = torch.load(f"{dataset}_processed/{experiment_type}_train_embed.pth") # Size: [number of test instances,] << sparse labels
if "ood" in exp_type:
    ood = torch.load(f"{dataset}_processed/{experiment_type}_ood_embed.pth")
else:
    ood = None

# Define the HMC parameters and last layer model for Pyro
model_type = 'llhmc'
target_acceptance = 0.8
num_samples = 50
burnin_samples = 100
prior = 1.0
n_chains = 1 # multiple chains will be executed in parallel (except for jax), be aware of the potential RAM that might require
hidden_dim=X_train.shape[-1] 
n_classes=len(torch.unique(y_train))

# Instantiate the last layer pyro model and the sampler
LL_model = LL_BNN(prior_scale=prior, hid_dim=hidden_dim, out_dim=n_classes)
nuts_kernel = NUTS(LL_model, jit_compile=True, target_accept_prob=target_accept) 
mcmc = MCMC(nuts_kernel, num_samples=num_samples, warmup_steps=burnin_samples, num_chains=n_chains)

# Run the HMC sampling and obtain the sampled parameters
mcmc.run(X_train, y_train)
posterior_samples = mcmc.get_samples()
weight_samples = torch.concat([weight_samples, posterior_samples['layers.0.weight']])
bias_samples = torch.concat([bias_samples, posterior_samples['layers.0.bias']])

# Model torch wrapper for the sampled last layer parameters 
model = VmappedLinearLayer(weight_samples, bias_samples)
# Run the evaluation for the test dataset (and OOD instances)
results = evaluate(model_type=model_type, model=model, X_test=X_test, y_test=y_test, ood=ood)
```
