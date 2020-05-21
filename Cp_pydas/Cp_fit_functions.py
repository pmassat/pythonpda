# -*- coding: utf-8 -*-
"""
Created on Wed May 20 17:26:26 2020

@author: Pierre Massat <pmassat@stanford.edu>
"""

from matplotlib import pyplot as plt
import numpy as np
from scipy.optimize import brentq 
# brentq is the fastest function to find the root of a *scalar* function on an interval (does NOT work with a numpy array)

np.seterr(under='warn')

def sech(x):
    return 1/np.cosh(x)

def order_parameter(t):
    """
    calculate the order parameter of the MF FQ transition (which equals the
    normalized pseudospin) as a function of reduced temperature t = T/Tc, 
    in the absence of any externally applied longitudinal field
    """
    return brentq(lambda x: x - np.tanh(x/t), 1e-3, 1)
    # if 0 is included in the range of fzero, y will always be zero, since np.tanh(0) = 0
    
    
def OP_TFIM(t, h, e):
    gamma = lambda x: np.sqrt((x+e)**2+h**2)
    fun = lambda x: x - (x+e)/gamma(x) * np.tanh(gamma(x)/t)
    return brentq(fun, 1e-6, 1)


def Cp_LFIM(t, sigma):
    # t is the reduced temperature T/Tc
    y = np.zeros(t.shape)
    tc =1
    
    for i in range(len(t)):
        y[i] = 0
        x = OP_TFIM(t[i], 0, sigma)
        gamma = (x+sigma)
        r = gamma/t[i]# ratio of reduced order_parameter to reduced temperature
        D = 1 - sech(r)**2 / t[i]
        if t[i]<tc or sigma!=0:
            y[i] = r**2*sech(r)**2 / D# mean-field heat capacity in the ordered phase
        else:
            y[i] = 0
            
    return y



def Cp_LFIM_MEC_cst_stress(t, sigma, a, t0):
    # t is the reduced temperature T/Tc
    y = np.zeros(t.shape)
    tc = 1
    
    for i in range(len(t)):
        x = OP_TFIM(t[i],0,sigma)
        gamma = (x + sigma)
        r = gamma/t[i]# ratio of reduced order_parameter to reduced temperature
        D = 1 - sech(r)**2/t[i]
        if t[i]<tc:
            y[i] = r**2*sech(r)**2 / D# mean-field heat capacity in the ordered phase
        elif sigma!=0:
            y[i] = r**2*sech(r)**2 / D + a*t[i]/(t[i]-t0)**3
        else:
            y[i] = a*t[i]/(t[i]-t0)**3
            
    return y


def Cp_TFIM(t,h):
    # t is the reduced temperature T/Tc
    y = np.zeros(t.shape)
    
    if h==0:
        tc = 1
    elif abs(h)<1:
        tc = h / np.arctanh(h)
    else: 
        tc = 0
    
    for i in range(len(t)):
        if t[i]<=0:
            y[i] = 0
        elif t[i]<tc:
    #         r = order_parameter(t[i])/t[i]# ratio of reduced order_parameter to reduced temperature
            r = order_parameter(t[i])/t[i]# ratio of reduced order_parameter to reduced temperature
            y[i] = r**2*sech(r)**2 / (1 - 1/t[i]*sech(r)**2)# mean-field heat capacity in the ordered phase
        else:
            r = h/t[i]
            y[i] = r**2*sech(r)**2# mean-field heat capacity in the disordered phase
            
    return y
    

if __name__=='__main__':
    x = np.linspace(5e-3, 1.5, 300)
    y = np.zeros(x.shape)
    e = 1.5e-3
    h = 0
    plt.figure()
    for idx, val in enumerate(x):
        y[idx] = OP_TFIM(val, 0, e)
    plt.plot(x, y)
