import numpy as np
from sampler import MCSampler

# There are two options for initialization:

# 1. Use a predefined transition matrix

alpha = np.array([
    [0.00, 0.25, 0.25, 0.25],
    [0.25, 0.00, 0.25, 0.25],
    [0.25, 0.25, 0.00, 0.25],
    [0.25, 0.25, 0.25, 0.00]
])

gamma = np.array([0.33, 0.33, 0.34, 0.00])

mc_sampler = MCSampler(alpha, gamma)

# 2. Randomly initialize the sampler

n_classes = 4

mc_sampler = MCSampler.random_init(n_classes)

# To generate a sequence run:

sequence = mc_sampler.gen_sequence()
print(sequence)
