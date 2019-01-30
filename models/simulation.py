#!/usr/bin/env python
# coding: utf-8

import numpy as np, scipy as sp
from scipy import stats



def simul_poisson(P, lambda_, size = 10): # Simulate an artificial poisson MMM
    y = []
    m = len(lambda_)
    S = []
    s = np.random.choice(m)
    for t in range(size):
        S.append(s)
        y.append(stats.poisson.rvs(mu = lambda_[S[-1]]))
        s = np.random.choice(m, p = P[s])
    return np.array(y),np.array(S)



def simul_gauss(P, mu, sigma, size = 10):
    y = []
    S = []
    m = len(mu)
    s = np.random.choice(m)
    for t in range(size):
        S.append(s)
        y.append(stats.multivariate_normal.rvs( mean = mu[s], cov = sigma[s]) )
        s = np.random.choice(m, p = P[s])
    return np.array(y), np.array(S)


