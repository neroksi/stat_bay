#!/usr/bin/env python
# coding: utf-8

#from copy import deepcopy
from datetime import datetime

class HMM(object):
    """A general class for Hidden Markov Models.

    A Hidden Markov Model (HMM) is a list of three distributions which
    describe the paramters :math: `\theta \in \Theta`, the states and 
    the transition matrix.

    Parameters
    __________
    tDictDist : List of dicts
        Each item of the dict must be an array like, of shape
        ...,K where K is the number of components in the population,
        K = 2 for a two components gaussian mixture. This is the whole
        prior over the parameters

    sDist : a dist object
        The prior over S, the states.

    pDist : a dist object
        The prior over P, the transition matrix.

    Y : array like
        The observations.
    S : array like
        The initial values for the states, could be sampled at random.
        It must have the same size as Y.
    P : KxK 2-d array
        The transition matrix.

    H : dict
        The HMM's history.
    """
    
    def __init__(self, tDictDist, sDist, pDist, Y, S, Theta , P, H = None):
        self.tDictDist = tDictDist
        self.sDist = sDist
        self.pDist = pDist
        self.Y = Y
        self.S = S
        self.Theta = Theta
        self.P = P
        self.H = None
    
    def _print(self, t0, t, niter, rounds) :
        elapsed = (t - t0).total_seconds()/60
        t = t.strftime("%Y-%m-%d  %H:%M:%S")
        remain = rounds - niter
        premain = int(100*(1 - niter/rounds))
        
        print("Time : %s   Elapsed : %0.2f min.  N_iter : %04d   Remain : %04d   %%Remain : %02d%%"%(t,elapsed, niter, remain, premain))
#     def one_sample(self, which = "tDist", Y = None, S= None, Theta = None, P = None):
#         """Deliver one sample of the `which` distribution"""
        
#         xDist = getattr(self, which)
#         if isinstance(xDist, list):
#             rvs = []
#             for i in range(len(xDist)) :
#                 d = xDist[i]
#                 rvsd = {}
#                 for key, value in d.items() :
#                     value.update(Y=Y, S=S, Theta = Theta[i] ,P=P)
#                     rvsd[key] = value.rvs()
#                 rvs.append(rvsd)
#         else:
#             xDist.update(Y=Y, S=S, Theta = Theta,P=P)
#             rvs = xDist.rvs()
#         return rvs
    
    def one_sample2(self, which = "tDist", Y = None, S = None, Theta = None, P = None):
        """Deliver one sample of the `which` distribution"""
        
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
        """Perform one pass of the Gibbs sampler"""
        if Y is None :
            Y = self.Y
        if Theta is None :
            Theta = self.Theta
        if S is None:
            S = self.S
        if P is None :
             P = self.P
        self.Theta = self.one_sample2(which = "tDictDist", Y = Y, S = S, P = P, Theta = Theta)
        self.P = self.one_sample2(which = "pDist", Y = Y, S = S, Theta = Theta)
        self.S = self.one_sample2(which = "sDist", Y = Y, P = P, Theta = Theta)
    
    def run(self, rounds = 10, historise = True, verbose = True):
        """Will run  and historicise all the Gibbs sampler steps."""
        t0 = datetime.now()
        if historise :
            if self.H is None :
                self.H  = {"Theta": [], "S": [], "P": []}
            for i in range(rounds):
                self.H["Theta"].append(self.Theta) #(deepcopy(self.Theta))
                self.H["S"].append(self.S) #(deepcopy(self.S))
                self.H["P"].append(self.P) #(deepcopy(self.P))
                self.one_step(Y = self.Y , S = self.S, Theta = self.Theta)
                if verbose and i%500 == 0:
                    self._print(t0, datetime.now(), i, rounds)
        else :
            for i in range(rounds):
                self.one_step(Y = self.Y , S = self.S, Theta = self.Theta)
                if verbose and i%500 == 0 :
                    self._print(t0, datetime.now(), i,rounds)