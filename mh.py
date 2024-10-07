# Metropolis-Hastings with Gaussian proposal distribution

import numpy as np

class GaussianMHSampler:
    def __init__(self, p, default_init_pt, proposal_var, rng = np.random.default_rng()):
        self.p = p
        self.dim = p.dim
        self.default_init_pt = default_init_pt
        self.proposal_var = proposal_var
        self.density = p.density
        self.rng = rng

    def step(self, x):
        num_chains = x.shape[0]
        x_new = x + self.rng.normal(0, self.proposal_var, size=x.shape)
        p_old = self.density(x)
        p_new = self.density(x_new)
        accept = self.rng.random(num_chains) < p_new / p_old
        accept = accept.reshape(-1, 1)
        return np.where(accept, x_new, x)

    def sample(self, num_chains, num_samples_per_chain, burnin=1000, init_pt=None, merge=True):
        samples = np.zeros((num_chains, num_samples_per_chain, self.dim))
        x = init_pt
        if init_pt is None:
            x = self.default_init_pt.reshape(1, -1)* np.ones((num_chains, self.dim))

        for _ in range(burnin):
            x = self.step(x)

        for i in range(num_samples_per_chain):
            x = self.step(x)
            samples[:, i, :] = x.reshape(num_chains, self.dim)

        if merge:
            samples = samples.reshape(-1, self.dim)
        return samples
    
