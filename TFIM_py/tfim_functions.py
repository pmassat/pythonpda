# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 16:31:03 2020

@author: Pierre Massat <pmassat@stanford.edu>
"""

#%% Modules importation
import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt


#%% Phase boundary

def critical_field(t):
    if 0<= t <1:
        hc = fsolve(lambda x: x-np.tanh(x/t), 1)
    else:
        hc = 0;
    return hc


#%% 
if __name__=='__main__':
    t = np.concatenate([np.logspace(-5,-2), 
                        np.linspace(.02,.99,98),
                        1-np.logspace(-2,-8)])
    hc = np.ones(np.size(t))
    rng = range(len(t))
    
    for i in rng:
        hc[i] = critical_field(t[i])
    
    
    # Plot phase boundary
    plt.figure
    plt.plot(hc, t)