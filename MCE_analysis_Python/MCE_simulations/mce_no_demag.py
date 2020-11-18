# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 14:52:13 2020

@author: Pierre Massat <pmassat@stanford.edu>
"""

#%% Import functions
import numpy as np
from matplotlib import pyplot as plt


#%% Define variables

Hc = 5e3 # critical field
Hmax = 1e4 # Max field, in Oe
dtH = 10# Sweeprate, in Oe/s
intHmax = int(Hmax//dtH + Hmax%dtH) # Maximum number of time steps
t = np.linspace(0, intHmax, num=intHmax+1) # time array

tau = 200 # time constant for leakage of heat to the bath 

#%% Compute physical quantities of MCE 
Hup = t*dtH # array of magnetic fields for upsweep
Hdown = Hmax-t*dtH # array of magnetic fields for downsweep

Tup = np.ones(np.size(t)) # array of temperatures
Tdown = np.ones(np.size(t)) # array of temperatures

Tup[Hup>Hc] = 1 + dtH/Hc*(t[Hup>Hc]-t[Hup==Hc])*np.exp(-(t[Hup>Hc]-t[Hup==Hc])/tau)
Tdown[Hdown>Hc] = 1 - dtH/Hmax*(t[Hdown>Hc])*np.exp(-(t[Hdown>Hc])/tau)


#%% Plot data

plt.plot(t,Tup)
plt.plot(t,Tdown)

