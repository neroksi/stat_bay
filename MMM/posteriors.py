#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd, numpy as np, scipy as sp, time
from scipy import stats


# In[2]:


def _poisson(y, _lambda): # Poisson cdf
    return np.power(_lambda, y)*np.exp(-_lambda)/sp.special.factorial(y)

def _poisson2(y, _lambda): # Poisson cdf
    return stats.poisson.pmf(y, _lambda)


# In[3]:


def _Phi(y, **kwargs):
    mu = kwargs["mu"] 
    omega = np.linalg.inv(kwargs["omega_inv"])
    v = np.array([stats.multivariate_normal.pdf(y,mean = mu[i], cov =  omega[i] ) for i in range(len(mu))])
    return v


# In[4]:


def order(Y, S):
    Y = np.sum(np.array(Y),1)
    u = np.bincount(S, weights= Y)
    v = np.bincount(S)
    r = np.argsort(u/v)
    d = S.copy()
    m = S.max() 
    for i in range(len(r)):
        d[d == r[i]] = m + i +1
    return  d-m-1


# In[5]:


def apply_by_group(x, g,axis = 0, func = np.sum):
    o = np.argsort(g)
    g  =g[o]
    x = x[o]
    indices = [0]
    for i in range(1, len(g)):
        if g[i-1] != g[i] :
            indices.append(i)
    indices.append(i+1)
    
    res = [ func(x[indices[i-1]:indices[i] ], axis = axis) for i in range(1,len(indices))]
    
    return np.array(indices), np.array(res)
def apply2(x,g):
    df = pd.DataFrame(x)
    df["g"] = g
    res = df.groupby("g").sum()
    return res


# In[22]:


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

            
def set_attr(obj, **kwargs):
    for arg, value in kwargs.items():
        setattr(obj, arg, value)


# In[23]:


def bincount2D(a, weights = None):    
    N = a.max()+1
    a_offs = a + np.arange(a.shape[0])[:,None]*N
    if weights is None :
        return np.bincount(a_offs.ravel(), minlength=a.shape[0]*N).reshape(-1,N)
    else :
        return np.bincount(a_offs.ravel(), minlength=a.shape[0]*N, weights = weights.ravel()).reshape(-1,N)


# In[24]:


def _mean(x, axis = 0):
    return np.mean(x[:,None,:],axis = axis)


# In[25]:


def _var2(x, axis = 0):
    x = x- x.mean(axis)
    return x.transpose()@x


# In[26]:


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


# In[27]:


class gamma(dist):
    def __init__(self,rvs = None, a = 1, scale = 1, a0 = 1, scale0 = 1):
        super().__init__(rvs)
        set_default_if_null(self,_rvs = stats.gamma.rvs, params = {"a": a, "scale": scale}, 
                   default  = {"a0": a0, "scale0": scale0})   
    
    def update(self, Y, S, Theta = None, P = None ) :
        u = np.bincount(S, weights= Y)
        v = np.bincount(S)
        self._set(a = self.default["a"] + u, scale = 1/(1/self.default["scale"] + v))


# In[130]:


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
        ind, nybar = apply_by_group( Y, S)
        N = ind[1:] - ind[:-1] 
        sigma_inv = self.default["sigma0_inv"] + Theta["omega_inv"]*N[:,None, None]
        sigma = np.linalg.inv(sigma_inv)
        mu_1 = self.default["sigma0_inv"]@self.default["mu0"][:,:,None]
        mu_2 = Theta["omega_inv"]@nybar[:,:,None]
        mu = mu_1 + mu_2
        mu = (sigma@mu)[:,:,-1]
        self._set(sigma = sigma, mu = mu)


# In[131]:


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
        ind, yvar = apply_by_group( Y , S, func= _var2)
        ind, ymean = apply_by_group( Y , S, func= np.mean)
        N = ind[1:] - ind[:-1]
        
        ymean = ymean - Theta["mu"]
        ymean = ymean[:, :, None]@ymean[:, None, :]
        ss = yvar + N[:, None, None]*ymean
        omega = self.default["omega0_inv"] + ss

        omega = np.linalg.inv(omega)
        nu =self.default["nu0"] + N
        self._set( nu = nu, omega = omega)


# In[132]:


class dirichlet(dist ):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        set_default_if_null(self, _rvs = stats.gamma.rvs, params = {"a":1 }, default = {"a":1 })

    
    def update(self, Y= None, S= None, Theta = None, **kwargs) :
        n = np.zeros(self.params["a"].shape)
        tab = pd.crosstab(S[:-1], S[1:]) # transitions' count
        n[ :len(tab), tab.columns] = tab.values
        self._set(a = self.default["a"] + n )
        
    def rvs(self) :
        g = self._rvs(**self.params)
        return g/g.sum(1)[:, None]


# In[133]:


class simul_s(dist):
    def __init__(self, func = None ):
        super().__init__()
        self.func = func
        self.F = None
        self.P = None
        self.n = None
        self.m = None
    
    def _init(self, P): # Compute the left eigenvector assocated with the eigenvalue 1
        eigval, eigvect = sp.linalg.eig(P, left = True, right=  False)
        p = eigvect[:, eigval.real.round(3) == 1.].reshape(-1)
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
        
        #print(n,m)

        F = np.zeros((n, m))

        v = self.get_f2(Y[0], Theta = Theta)
        F[0] = self._init(P)*v
        F[0] = F[0]/F[0].sum()
        for i in range(1, n):
            v = self.get_f2(Y[i], Theta = Theta)
            #print(v)
            
            F[i] = (F[i-1]@P)*v
            F[i] = F[i]/F[i].sum()
        self.Y = Y
        self.F = F
        self.P = P
        self.n = n
        self.m = m
        
        #self._rvs = order(Y,S)
        #self._rvs = S
    def rvs(self):
        assert (not self.F is None) and (not self.P is None)
        S = np.zeros(self.n, dtype = int)
        a = np.arange(self.m)
        p = self.F[-1]
        S[-1] = np.random.choice(a = a , p = p/p.sum())    

        for i in range(self.n-2, -1, -1):
            p = self.F[i]*self.P[:, S[i+1]]
            S[i] = np.random.choice(a = a , p = p/p.sum())
        return order(self.Y,S)