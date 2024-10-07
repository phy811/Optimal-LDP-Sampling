# Implementation of the previous mechanism proposed by Husain et al., 2020
# MBDE (Mollified Boosted Density Estimation)

import numpy as np
import torch
import torch.nn as nn
from measure import ContinuousMeasure
from mh import GaussianMHSampler
import scipy.stats as stats

class WeakLearner:
    def __init__(self, generate_classifier, max_val, gamma1=None, gamma2=None):
        self.generate_classifier = generate_classifier
        self.max_val = max_val
        self.gamma1 = gamma1
        self.gamma2 = gamma2

class MBDE:
    def __init__(self, eps, weak_learner, base_dist):
        self.weak_learner = weak_learner
        self.base_dist = base_dist
        self.theta_base = eps/(eps + 4*weak_learner.max_val)

    def perturb(self, dist, num_iter=3, **gen_kwargs):
        Q = self.base_dist
        for t in range(1, num_iter+1):
            c = self.weak_learner.generate_classifier(dist, Q, **gen_kwargs)
            def new_density(old_Q, c, t):
                return lambda x: old_Q.density(x) * np.exp(c(x) * self.theta_base ** t)
            Q = ContinuousMeasure(self.base_dist.dim, new_density(Q,c,t), self.base_dist.ranges)

        return Q.normalize()
    
    def __call__(self, dist, num_iter=3):
        return self.perturb(dist, num_iter)

def mbde_GaussianBase(mean, 
                      cov, 
                      eps,
                      data_radius = None,
                      train_size=10000,
                      batch_size=5000,
                      num_epoch=750,
                      model_generator = None,
                      loss_fn = nn.BCELoss(),
                      optim_generator = lambda model:torch.optim.SGD(model.parameters(), lr=0.01, nesterov=True, momentum=0.9),
                      rng=np.random.default_rng()):
    
    dim = mean.shape[0]
    if model_generator == None:
        # Default model for weak learner, as in Husain et al., 2020
        model_generator = lambda: nn.Sequential(
                                    nn.Linear(dim, 25),
                                    nn.Tanh(),
                                    nn.Linear(25, 25),
                                    nn.Tanh(),
                                    nn.Linear(25, 25),
                                    nn.Tanh(),
                                    nn.Linear(25, 1),
                                    nn.Sigmoid(),
                                    nn.Flatten(-2, -1)
        )
    if data_radius == None:
        rv_range = [[-np.inf, np.inf] for d in range(dim)]
    else:
        rv_range = [[mean[d]-data_radius, mean[d]+data_radius] for d in range(dim)]
    base_dist = ContinuousMeasure(
        dim,
        stats.multivariate_normal(mean=mean, cov=cov).pdf,
        rv_range,
        auto_truncate=True
    ).normalize()
    
    
    def generate_classifier(P, Q):
        sampler_P = GaussianMHSampler(P, mean, np.diag(cov), rng)
        sampler_Q = GaussianMHSampler(Q, mean, np.diag(cov), rng)
        samples_P = torch.tensor(sampler_P.sample(train_size, 1), dtype=torch.float32)
        samples_Q = torch.tensor(sampler_Q.sample(train_size, 1), dtype=torch.float32)
        num_steps = train_size // batch_size
        model = model_generator()
        optimizer = optim_generator(model)
        labels = torch.cat((torch.ones(batch_size), torch.zeros(batch_size)))
        for epoch in range(num_epoch):
            for step in range(num_steps):
                optimizer.zero_grad()
                batch_P = samples_P[step*batch_size:(step+1)*batch_size]
                batch_Q = samples_Q[step*batch_size:(step+1)*batch_size]
                inputs = torch.cat((batch_P, batch_Q), dim=0)
                outputs = model(inputs)
                loss = loss_fn(outputs, labels)
                loss.backward()
                optimizer.step()
        def model_numpy(x):
            if x.ndim == 1:
                x = x.reshape(1, -1)
            x_torch = torch.tensor(x, dtype=torch.float32)
            output_torch = model(x_torch)
            return output_torch.detach().numpy()
        return model_numpy
    
    weak_learner = WeakLearner(generate_classifier, 1)
    return MBDE(eps, weak_learner, base_dist)
    


