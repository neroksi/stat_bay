#!/usr/bin/env python
# coding: utf-8

import pandas as pd, numpy as np
from scipy import stats, linalg, special
from sklearn import metrics

def _poisson(y, _lambda): # Poisson cdf
    return np.power(_lambda, y[-1])*np.exp(-_lambda)/special.factorial(y[-1])

# def _poisson2(y, _lambda): # Poisson cdf
#     return stats.poisson.pmf(y[-1], _lambda)

# def similarity(df, groups = ["S0", "S1"]):
#     X = df.sort_values(groups)
#     w = X.loc[:, ~X.columns.isin(groups)] - X.groupby(groups).transform("mean")
#     w = 1/(1+w**2)
#     w = w.sum(1)
#     sims = pd.crosstab(X[groups[0]], X[groups[1]],w, aggfunc= "sum", normalize= "columns")
#     mapto = []
#     for i in sims.index:
#         idxmax = sims.loc[i, ~sims.columns.isin(mapto)].idxmax(1)
#         mapto.append(idxmax)
#     return mapto

# def similarity2(S0, S1, kmax = None):
#     if kmax is None:
#         kmax = max(S0.max(), S1.max()) +1
#     L = np.empty((kmax,kmax)) 

#     for i in range(kmax) :
#         for j in range(kmax) :
#             L[i,j] = metrics.cluster.adjusted_rand_score(S0 == i, S1==j)
#     v = np.arange(kmax)
#     c = list(range(kmax))
#     mapto = []
    
#     for i in range(kmax) : # compute argmax and ensure that there is no duplicates
#         idmax = np.argmax(L[~np.isin(v, mapto), i])
#         mapto.append(c[idmax])
#         c.pop(idmax)
        
#     return mapto

def _Phi(y, mu, omega_inv, **kwargs):
    """Compute the pdf of a multivariate normal dist.
    
    Parameters :
    __________
    y : 1-d array 
        On sample of the observations. Must have a `d`
        size, where `d`is  the observations' dimension.

    mu : 2-d numpy array, Kxd
        The mean of the K sub-populations.
        `K` is the number of components and `d`
        the observations' dimension.

    omega_inv: 3-d numpy array, Kxdxd
        The inverse of the sub-populations' covariance
        matrix. `K` is the number of components and `d`
        the observations' dimension.

    **kwargs : any
        Ignored.
    """
    y = y[-1]
    mu = y -mu
    trace = mu[:,None,:]@omega_inv
    trace = trace[:,-1,:]*mu
    trace = trace.sum(1)
    det = np.linalg.det(omega_inv)
    pdf_y = np.sqrt(det)*np.exp(-trace/2)#det and not 1/det
                        # Since we are working with the inverse
    return pdf_y/pdf_y.sum()


def order(Y, S):
    Y = np.array(Y)
    Y = np.sum(Y,1 if Y.ndim > 1 else None)
    u = np.bincount(S, weights= Y if Y.ndim > 1 else None)
    v = np.bincount(S)
    r = np.argsort(u/v)
    d = S.copy()
    m = S.max() 
    for i in range(len(r)):
        d[d == r[i]] = m + i +1
    return  d-m-1

def replace(old_val, new_val, x):
    order = np.arange(len(x))
    dt = pd.DataFrame({"old":old_val, "new":new_val})
    dt = dt.merge(pd.DataFrame({"x":x, "order":order}), left_on = "old", right_on = "x", how = "right")
    u = ~dt.new.isnull()
    dt.loc[u, "x"] = dt.new[u].values
    return dt.sort_values("order").x.values

def apply_by_group(x, g,axis = 0, func = np.sum, gmax = None):
    o = np.argsort(g)
    g  =g[o]
    x = x[o]
    indices = [0]*(g[0] +1)
    for i in range(1, len(g)):
        if g[i-1] != g[i] :
            for j in range(g[i-1], g[i]):
                indices.append(i)
    gmax = g.max() if gmax is None else gmax
    for j in range(g[i], gmax):
        indices.append(i)
    indices.append(i+1)
    
    res = [ func(x[indices[i-1]:indices[i] ], axis = axis) for i in range(1,len(indices))]
    res = np.nan_to_num(res, copy = False)
    if len(indices) != gmax +2 :
        print(gmax,g, indices)
    return np.array(indices), res

def return_default_if_null(three_uples):
    d = {}
    for three_uple in three_uples :
        if three_uple[1] is None or not len(three_uple[1]) :
            d[three_uple[0]] = three_uple[2]
        else:
            d[three_uple[0]] = three_uple[1]
    return d
                  

def set_default_if_null(obj , **kwargs):
    for arg, value in kwargs.items():
        try :
            v = getattr(obj, arg) 
        except AttributeError:
            v = obj[arg]
            
        if v is None or not len(v):
            try:
                setattr(obj, arg, value)
            except:
                obj[arg] = value

def _var(x, axis = 0):
    x = x- x.mean(axis)
    return x.transpose()@x


class dist(object):
    def __init__(self, rvs = None,  params = {}, default = {}):
        self._rvs = rvs
        self.params = params
        self.default = default
        
    def update(self ,Y= None, S = None, P = None,Theta = None): 
        pass
            
    def _set(self, **kwargs):
        for key, value in kwargs.items() :
            self.params[key] = value
    
    def _set_default(self, **kwargs):
        for key, value in kwargs.items() :
            self.default[key] = value
            
    def rvs(self):
        return self._rvs(**self.params)


class gamma(dist):
    def __init__(self,rvs = None, a = 1, scale = 1, a0 = 1, scale0 = 1):
        super().__init__(rvs)
        set_default_if_null(self,_rvs = stats.gamma.rvs, params = {"a": a, "scale": scale}, 
                   default  = {"a0": a0, "scale0": scale0})   
    
    def update(self, Y, S, Theta = None, P = None ) :
        u = np.bincount(S, weights= Y)
        v = np.bincount(S)
        self._set(a = self.default["a"] + u, scale = 1/(1/self.default["scale"] + v))


class norm(dist):
    def __init__(self, rvs = None, mu = None, sigma = None, mu0 = None, sigma0_inv = None ):
        super().__init__(rvs)
        arr = np.array([np.eye(2), np.eye(2)])
        kwargs = return_default_if_null(
            [("_rvs", self._rvs, stats.multivariate_normal.rvs), ("mu" , mu, np.zeros((2,2))),
             ("sigma", sigma, arr), 
                                ("mu0", mu0, np.zeros((2,2))), ("sigma0_inv", sigma0_inv, arr) ])
    
        params = {"mu": kwargs["mu"], "sigma": kwargs["sigma"]}
        default = {"mu0": kwargs["mu0"], "sigma0_inv": kwargs["sigma0_inv"]}
        self._rvs = kwargs["_rvs"]
        self.params = params
        self.default = default
  
    def rvs(self):
        return np.array(
            [self._rvs(mean =self.params["mu"][i], cov  = self.params["sigma"][i]) 
                 for i in range(len(self.params["mu"]))])
       
            
    def update(self, Y, S, Theta = None, P = None) :
        m = P.shape[1]
        ind, nybar = apply_by_group( Y, S, gmax = m-1)
        N = ind[1:] - ind[:-1] 
        
        if len(N) != m :
            raise ValueError(" %d = len(N) != m = %d, something went wrong"%(len(N), m))
        sigma_inv = self.default["sigma0_inv"] + Theta["omega_inv"]*N[:,None, None]
        sigma = np.linalg.inv(sigma_inv)
        mu_1 = self.default["sigma0_inv"]@self.default["mu0"][:,:,None]
        mu_2 = Theta["omega_inv"]@nybar[:,:,None]
        mu = mu_1 + mu_2
        mu = (sigma@mu)[:,:,-1]
        self._set(sigma = sigma, mu = mu)


class wishart(dist):
    def __init__(self, rvs = None, nu = None, omega = None, nu0 = None, omega0_inv = None ):
        super().__init__(rvs)
        arr = np.array([np.eye(2), np.eye(2)])
        kwargs = return_default_if_null(
            [("_rvs", self._rvs, stats.wishart.rvs), ("nu" , nu, np.zeros(2)), ("omega", omega, arr), 
                                ("nu0", nu0, np.zeros(2)), ("omega0_inv", omega0_inv, arr) ])
    
        params = {"nu": kwargs["nu"], "omega": kwargs["omega"]}
        default = {"nu0": kwargs["nu0"], "omega0_inv": kwargs["omega0_inv"]}
        self._rvs = kwargs["_rvs"]
        self.params = params
        self.default = default

    
    def rvs(self):
        return np.array(
            [self._rvs(df =self.params["nu"][i], scale  = self.params["omega"][i]) 
                 for i in range(len(self.params["nu"]))])
       
            
    def update(self, Y, S, Theta = None, P = None) :
        m = P.shape[1]
        ind, yvar = apply_by_group( Y , S, func= _var, gmax = m-1 )
        ind, ymean = apply_by_group( Y , S, func= np.mean, gmax = m-1)
        N = ind[1:] - ind[:-1]
        
        ymean = ymean - Theta["mu"]
        ymean = ymean[:, :, None]@ymean[:, None, :]
        ss = yvar + N[:, None, None]*ymean
        omega = self.default["omega0_inv"] + ss

        omega = np.linalg.inv(omega)
        nu =self.default["nu0"] + N
        self._set( nu = nu, omega = omega)


class dirichlet(dist ):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        set_default_if_null(self, _rvs = stats.gamma.rvs,
                                 params = {"a":1 }, default = {"a":1 })

    
    def update(self, Y= None, S= None, Theta = None, **kwargs) :
        n = np.zeros(self.default["a"].shape)
        tab = pd.crosstab(S[:-1], S[1:]) # transitions' count
        n[ :len(tab), tab.columns] = tab.values
        self._set(a = self.default["a"] + n )
        
    def rvs(self) :
        g = self._rvs(**self.params)
        return g/g.sum(1)[:, None]


class simul_s(dist):
    def __init__(self, func = None ):
        super().__init__()
        self.func = func
        self.F = None
        self.P = None
        self.n = None
        self.m = None
        self.S = None
        self.first_S = True
        self.t = 0
    
    def _init(self, P): # Compute the left eigenvector assocated with the eigenvalue 1
        eigval, eigvect = linalg.eig(P, left = True, right=  False)
        p = eigvect[:, eigval.real.round(3) == 1.].reshape(-1).real
        p = p/p.sum()
        return p
        
    def get_f(self, y, Theta):
            v = []
            for theta in Theta :
                v.append(self.func(y , **theta))
            v = np.array(v).ravel()
            return v
        
    def get_f2(self, y, Theta):
            return self.func(y , **Theta)
            
    def update(self, Y= None, Theta={}, P= None, S = None): # Simulate the states
        n = len(Y)
        m = len(P)
        F = np.zeros((n, m))

        v = self.get_f2(Y[:1], Theta = Theta)
        F[0] = self._init(P)*v
        F[0] = F[0]/F[0].sum()
        for i in range(1, n):
            v = self.get_f2(Y[:i+1], Theta = Theta)
            #print(v)
            
            F[i] = (F[i-1]@P)*v
            F[i] = F[i]/F[i].sum()
        self.Y = Y
        self.F = F
        self.P = P
        self.n = n
        self.m = m
        self.first_S = True
        
    def rvs(self):
        assert (not self.F is None) and (not self.P is None)
        S = np.zeros(self.n, dtype = int)
        a = np.arange(self.m)
        p = self.F[-1]
        S[-1] = np.random.choice(a = a , p = p/p.sum())    

        for i in range(self.n-2, -1, -1):
            p = self.F[i]*self.P[:, S[i+1]]
            S[i] = np.random.choice(a = a , p = p/p.sum())
        
        
        if self.first_S :
            self.S = S
            self.first_S = False
        self.t += 1
        return order(self.Y,S)