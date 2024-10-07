# A data structure representing an absolutely continuous measure on a Euclidean space

import numpy as np
import scipy.integrate as integrate

def truncation(f, ranges):
    ranges_np = np.array(ranges)
    lb = ranges_np[:, 0].reshape([-1])
    ub = ranges_np[:, 1].reshape([-1])
    def new_density(x):
        return np.where(np.all(np.logical_and(lb <= x, x <= ub), axis=-1), f(x), 0)
    return new_density

class ContinuousMeasure:
    def __init__(self, dim, density, ranges, auto_truncate=False):
        self.dim = dim
        self.density = density
        if auto_truncate:
            self.density = truncation(density, ranges)
        self.ranges = np.array(ranges)

    def density_pack(self, *x):
        return self.density(np.array(x))
        
    def total_mass(self):
        return integrate.nquad(self.density_pack, self.ranges)[0]
    
    def integrate(self, f):
        return integrate.nquad(lambda *x: f(np.array(x)) * self.density(np.array(x)), self.ranges)[0]
    
    def normalize(self):
        total_mass = self.total_mass()
        return ContinuousMeasure(
            self.dim,
            lambda x: self.density(x) / total_mass,
            self.ranges
        )

    def truncate(self, ranges):
        return ContinuousMeasure(
            self.dim,
            truncation(self.density, ranges),
            ranges
        )

class GaussianMixture_sameVar(ContinuousMeasure):
    def __init__(self, peaks, var, weights):
        self.dim = peaks.shape[1]
        self.peaks = peaks
        self.var = var
        self.weights = weights
        self.ranges = np.array([[-np.inf, np.inf] for _ in range(self.dim)])

    def density(self, x):
        peaks = np.expand_dims(self.peaks.T, axis=tuple(range(x.ndim-1)))
        x = np.expand_dims(x, axis=-1)
        exponents = np.linalg.norm(x-peaks, axis=-2) ** 2 / (2 * self.var)
        return np.average(np.exp(-exponents), axis=-1, weights=self.weights) / (2 * np.pi * self.var)


class GaussianRing(GaussianMixture_sameVar):
    def __init__(self, center, radius, var, num_modes):
        self.dim = 2
        self.peaks = center + np.array([([radius * np.cos(2*np.pi*i/num_modes), radius * np.sin(2*np.pi*i/num_modes)]) for i in range(num_modes)])
        self.var = var
        self.ranges = np.array([[-np.inf, np.inf], [-np.inf, np.inf]])
        self.weights = np.ones(num_modes)
    

