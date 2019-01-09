#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd, numpy as np, scipy as sp, time
from scipy import stats
from copy import deepcopy
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


class HMM(object):
    def __init__(self, tListDictDist, sDist, pDist, Y, S, Theta , P, H = None):
        self.tListDictDist = tListDictDist
        self.sDist = sDist
        self.pDist = pDist
        self.Y = Y
        self.S = S
        self.Theta = Theta
        self.P = P
        self.H = None
    
    def one_sample(self, which = "tDist", Y = None, S= None, Theta = None, P = None):
        
        xDist = getattr(self, which)
        if isinstance(xDist, list):
            rvs = []
            for i in range(len(xDist)) :
                d = xDist[i]
                rvsd = {}
                for key, value in d.items() :
                    value.update(Y=Y, S=S, Theta = Theta[i] ,P=P)
                    rvsd[key] = value.rvs()
                rvs.append(rvsd)
        else:
            xDist.update(Y=Y, S=S, Theta = Theta,P=P)
            rvs = xDist.rvs()
        return rvs
    
    def one_sample2(self, which = "tDist",Y = None, S= None, Theta = None, P = None):
        
        xDist = getattr(self, which)
        if isinstance(xDist, dict):
            rvs = {}
            for key, value in xDist.items() :
                value.update(Y= Y,S = S,Theta = Theta,P = P)
                rvs[key] = value.rvs()
        else:
            xDist.update(Y= Y,S = S,Theta = Theta,P = P)
            rvs = xDist.rvs()
        return rvs
        
    def one_step(self, Y = None , Theta = None,S = None, P = None):
        if Y is None :
            Y = self.Y
        if Theta is None :
            Theta = self.Theta
        if S is None:
            S = self.S
        if P is None :
             P = self.P
        self.Theta = self.one_sample2(which = "tListDictDist", Y = Y, S = S, P = P, Theta = self.Theta)
        self.P = self.one_sample2(which = "pDist", Y = Y, S = S, Theta = Theta)
        self.S = self.one_sample2(which = "sDist", Y = Y, P = P, Theta = Theta)
    
    def run(self, rounds = 10, historise = True):
        
        if historise :
            if self.H is None :
                self.H  = {"Theta": [], "S": [], "P": []}
            for i in range(rounds):
                self.H["Theta"].append(self.Theta) #(deepcopy(self.Theta))
                self.H["S"].append(self.S) #(deepcopy(self.S))
                self.H["P"].append(self.P) #(deepcopy(self.P))
                self.one_step(Y = self.Y , S = self.S, Theta = self.Theta)
        else :
            for i in range(rounds):
                self.one_step(Y = self.Y , S = self.S, Theta = self.Theta)


# In[ ]:




