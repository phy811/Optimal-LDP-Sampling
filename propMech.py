# Implementation of the proposed mechanism for continuous space

import numpy as np
import scipy.integrate as integrate
from measure import ContinuousMeasure

class ProposedMech_Continuous:
    def __init__(self, 
                 eps, base_measure, density_lb, density_ub,
                 isclose_atol=1e-5):
        max_err_rate = (1+isclose_atol)/(1-isclose_atol)
        self.isclose_atol = isclose_atol

        exp_eps = np.exp(eps)

        # adjust epsilon used in the mechanism
        # to account for the error tolerance
        # so that original epsilon is guaranteed
        exp_eps = exp_eps / max_err_rate

        base_measure_int = base_measure.total_mass()
        self.base_measure_norm = ContinuousMeasure(base_measure.dim, lambda x: base_measure.density(x) / base_measure_int, base_measure.ranges)
        lb_norm = density_lb * base_measure_int
        ub_norm = density_ub * base_measure_int

        if base_measure_int < 0 and not np.isclose(base_measure_int, 0):
            raise ValueError("base_measure must be non-negative.")
        if density_lb < 0 and not np.isclose(density_lb, 0):
            raise ValueError("density_lb must be non-negative.")
        if lb_norm > 1 and not np.isclose(lb_norm, 1):
            raise ValueError("density_lb with normalization must be at most 1.")
        if ub_norm < 1 and not np.isclose(ub_norm, 1):
            raise ValueError("density_ub with normalization must be at least 1.")

        alpha = (1 - lb_norm) / (ub_norm - lb_norm)
        self.mech_lb_norm = 1/(alpha * exp_eps + 1 - alpha)
        self.mech_ub_norm = exp_eps * self.mech_lb_norm
        self.r_min = lb_norm * (alpha * exp_eps + 1 - alpha)
        self.r_max = ub_norm * (alpha * exp_eps + 1 - alpha) / exp_eps

    def scale_and_clip(self, r, dist, x):
        return np.clip(dist.density(x)/r, 
                        self.mech_lb_norm * self.base_measure_norm.density(x), 
                        self.mech_ub_norm * self.base_measure_norm.density(x))



    def perturb(self, dist, max_iter=1000): 
        dist_int = dist.total_mass()
        r_min = self.r_min * dist_int
        r_max = self.r_max * dist_int
        int_perturbed = np.inf
        # binary search (bisection method) for r
        iter = 0
        while not np.isclose(int_perturbed, 1, atol=self.isclose_atol, rtol=0) and iter < max_iter:
            r = (r_min + r_max) / 2
            int_perturbed = integrate.nquad(lambda *x: self.scale_and_clip(r, dist, np.array(x)), 
                                            dist.ranges)[0]
            if int_perturbed < 1:
                r_max = r
            else:
                r_min = r
            iter += 1

        if iter == max_iter:
            print(f"Warning: binary search did not converge in {iter} iterations.")

        return ContinuousMeasure(dim=dist.dim, 
                                 density=lambda x: self.scale_and_clip(r, dist, x), 
                                 ranges=dist.ranges)
            

    def __call__(self, dist, max_iter=1000):
        return self.perturb(dist, max_iter)
    

    def worst_dist(self, f):
        r_max_weight = (1-self.r_min) / (self.r_max - self.r_min)
        r_min_weight = 1 - r_max_weight
        return r_max_weight * f(self.r_max) + r_min_weight * f(self.r_min)


