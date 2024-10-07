# Definitions of f-divergences

from scipy.integrate import nquad
import numpy as np

def kl(p, q):
    def integrand(*x):
        return p.density_pack(*x) * (np.log(p.density_pack(*x)) - np.log(q.density_pack(*x)))
    return nquad(integrand, p.ranges)[0]

def tv(p, q):
    def integrand(*x):
        return np.abs(p.density_pack(*x) - q.density_pack(*x))
    return nquad(integrand, p.ranges)[0]/2

def chi_square(p, q):
    def integrand(*x):
        return p.density_pack(*x) ** 2 / q.density_pack(*x)
    return nquad(integrand, p.ranges)[0] - 1

def hellinger(p, q):
    def integrand(*x):
        return (np.sqrt(p.density_pack(*x)) - np.sqrt(q.density_pack(*x)))**2
    return nquad(integrand, p.ranges)[0]/2

def kl_f(x):
    return np.where(np.isclose(x, 0), 0.0, x*np.log(x))

def tv_f(x):
    return np.abs(x-1)/2

def hellinger_f(x):
    return 1-np.sqrt(x)