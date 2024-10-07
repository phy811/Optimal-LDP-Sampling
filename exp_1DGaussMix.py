# Running the 1D Gaussian mixture experiment, and save the data

import numpy as np
from measure import ContinuousMeasure, GaussianMixture_sameVar
import torch
from div import *
from propMech import ProposedMech_Continuous
from mbde import mbde_GaussianBase
from scipy import stats
from tqdm import trange
from argparse import ArgumentParser

def run_experiment(eps = '1.0',
                    mean_num_modes = 3, 
                    max_num_modes = 10, 
                    maxabs_mode = 1, 
                    var = 1,
                    data_radius = 4,
                    size = 100,
                    seed:int=0):    
    
    filename = f'data_1DGaussMix_eps{eps}.npy'
    eps = float(eps)
    
    # setting random seed
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    # create mechanisms
    mbde = mbde_GaussianBase(mean = np.zeros((1, 1)), cov = np.ones((1, 1)), data_radius = data_radius, eps = eps, rng=rng)
    
    def density_ub(x):
        max_val = 1/(np.sqrt(2*np.pi*var) * (stats.norm.cdf(data_radius-maxabs_mode, scale=np.sqrt(var)) - stats.norm.cdf(-data_radius-maxabs_mode, scale=np.sqrt(var))))
        return np.where(np.abs(x) < maxabs_mode, max_val, max_val*np.exp(-(np.abs(x) - maxabs_mode) ** 2 / (2*var)))
    meas_ub = ContinuousMeasure(1, density_ub, [[-data_radius, data_radius]], auto_truncate=True)
    propMech = ProposedMech_Continuous(eps, meas_ub, 0, 1)

    # run experiment
    kl_list = np.zeros((size, 2))
    tv_list = np.zeros((size, 2))
    hellinger_list = np.zeros((size, 2))


    for i in trange(size):
        num_modes = np.minimum(rng.poisson(mean_num_modes - 1) + 1, max_num_modes)
        modes = rng.uniform(-maxabs_mode, maxabs_mode, (num_modes, 1))
        weights = rng.dirichlet(np.ones(num_modes))
        target = GaussianMixture_sameVar(modes, var, weights).truncate([[-data_radius, data_radius]]).normalize()
        output_mbde = mbde(target)
        output_propMech = propMech(target)

        kl_list[i, 0] = kl(target, output_propMech)
        kl_list[i, 1] = kl(target, output_mbde)
        tv_list[i, 0] = tv(target, output_propMech)
        tv_list[i, 1] = tv(target, output_mbde)
        hellinger_list[i, 0] = hellinger(target, output_propMech)
        hellinger_list[i, 1] = hellinger(target, output_mbde)

    np.save(filename, np.hstack((kl_list, tv_list, hellinger_list)))

        

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--eps', type=str, default='1.0')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--size', type=int, default=100)
    
    run_experiment(**vars(parser.parse_args()))