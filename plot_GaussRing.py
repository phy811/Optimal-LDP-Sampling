# Plot the Gaussian ring distributio and its perturbed sampling distributions, Figure 1 in the paper

import numpy as np
from measure import ContinuousMeasure, GaussianRing
import matplotlib.pyplot as plt
import matplotlib
from mbde import *
from propMech import *
from time import time

matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
    'font.serif' : 'Computer Modern Roman',
    'axes.labelsize'  : 10,
    'xtick.labelsize' : 10,
    'ytick.labelsize' : 10,
    'legend.fontsize' : 10,
    'figure.figsize' : (6, 2)
})

if __name__ == "__main__":
    eps = 0.5
    abs_range = 3
    gaussian_var = 0.5

    seed = 0
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    target = GaussianRing(0, 1, gaussian_var, 3)
    
    t = time()
    mbde = mbde_GaussianBase(np.array([0,0]), np.eye(2), eps, rng=rng)
    time_prep_mbde = time() - t

    t = time()
    def density_ub(x):
        max_val = 1 / (2*np.pi*gaussian_var)
        return np.where(np.linalg.norm(x, axis=-1) < 1, max_val, max_val*np.exp(-(np.linalg.norm(x, axis=-1) - 1) ** 2 / (2*np.sqrt(gaussian_var))))

    meas_ub = ContinuousMeasure(2, density_ub, [[-np.inf, np.inf], [-np.inf, np.inf]])
    
    propMech = ProposedMech_Continuous(eps, meas_ub, 0, 1)
    time_prep_ours = time() - t

    t0 = time()
    Q_mbde = mbde(target)
    t1 = time()
    Q_propMech = propMech(target)
    t2 = time()
    
    print("Running time:")
    print(f"Preparation, previous (MBDE): {time_prep_mbde:.2f} seconds")
    print(f"Preparation, ours: {time_prep_ours:.2f} seconds")
    print(f"perturbation, previous (MBDE): {t1-t0:.2f} seconds")
    print(f"perturbation, ours: {t2-t1:.2f} seconds")

    x = np.linspace(-abs_range, abs_range, 500)
    y = np.linspace(-abs_range, abs_range, 500)
    xx, yy = np.meshgrid(x, y)
    points = np.dstack((xx, yy))
    original_densities = target.density(points)
    mbde_output = Q_mbde.density(points)
    propMech_output = Q_propMech.density(points)

    plt.subplot(1, 3, 1)
    plt.imshow(original_densities, extent=(-abs_range, abs_range, -abs_range, abs_range), cmap='plasma', aspect='auto')
    plt.title('Original dist.')

    plt.subplot(1, 3, 2)
    plt.imshow(mbde_output, extent=(-abs_range, abs_range, -abs_range, abs_range), cmap='plasma', aspect='auto')
    plt.title('Sampling dist. [35]')
    plt.yticks([])
    

    plt.subplot(1, 3, 3)
    plt.imshow(propMech_output, extent=(-abs_range, abs_range, -abs_range, abs_range), cmap='plasma', aspect='auto')
    plt.title('Sampling dist. (ours)')
    plt.yticks([])

    plt.tight_layout()
    
    plt.savefig('ringGauss.eps')