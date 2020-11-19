# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 14:52:13 2020

@author: Pierre Massat <pmassat@stanford.edu>
"""

#%% Import functions
import numpy as np
from matplotlib import pyplot as plt


#%% Define variables

Hc0 = 5e3 # critical field
Hmax0 = 1e4 # Max field, in Oe
dtH0 = 10# Sweeprate, in Oe/s

H = np.linspace(0, Hmax0, num=int(Hmax0)+1) # time array

tau0 = 50 # time constant for leakage of heat to the bath, in seconds


def Tup(H, dtH=dtH0, Hc=Hc0, tau=tau0):
    y = np.ones(np.size(H)) # array of temperatures
    y[H>Hc] = 1 + (H[H>Hc]-Hc)/Hc*np.exp(-(H[H>Hc]-Hc)/(dtH*tau))
    return y

def Tdown(H, dtH=-dtH0, Hc=Hc0, Hmax=Hmax0, tau=tau0):
    y = np.ones(np.size(H)) # array of temperatures
    y[H>Hc] = 1 + (H[H>Hc]-Hmax)/Hmax*np.exp(-(H[H>Hc]-Hmax)/(dtH*tau))
    return y



#%% Plot T vs H
plt.figure
plt.plot(H, Tup(H, dtH=20, tau=50))
plt.plot(H, Tdown(H, dtH=-20, tau=50))


#%% Compute physical quantities of MCE as a function of time

intHmax = int(Hmax0//dtH0 + Hmax0%dtH0) # Maximum number of time steps
t = np.linspace(0, intHmax, num=intHmax+1) # time array

Hup = t*dtH0 # array of magnetic fields for upsweep
Hdown = Hmax0-t*dtH0 # array of magnetic fields for downsweep

Tup = np.ones(np.size(t)) # array of temperatures
Tdown = np.ones(np.size(t)) # array of temperatures

Tup[Hup>Hc] = 1 + dtH0/Hc*(t[Hup>Hc]-t[Hup==Hc])*np.exp(-(t[Hup>Hc]-t[Hup==Hc])/tau)
Tdown[Hdown>Hc] = 1 - dtH0/Hmax0*(t[Hdown>Hc])*np.exp(-(t[Hdown>Hc])/tau)


#%% Plot data

plt.plot(t,Tup)
plt.plot(t,Tdown)

