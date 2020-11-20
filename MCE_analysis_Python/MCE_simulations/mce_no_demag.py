# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 14:52:13 2020

@author: Pierre Massat <pmassat@stanford.edu>
"""

#%% Import functions
import numpy as np
# from scipy.special import expi
from matplotlib import pyplot as plt


#%% Define variables

Hc0 = 5e3 # critical field
Hmax0 = 1e4 # Max field, in Oe
dtH0 = 10# Sweeprate, in Oe/s

H = np.linspace(0, Hmax0, num=int(Hmax0)+1) # time array

tau0 = 50 # time constant for leakage of heat to the bath, in seconds

def Tmce(H, dtH=dtH0, dhT=1e-3, Hc=Hc0, H0=Hc0, tau=tau0, T0=1):
    """
    Parameters
    ----------
    H : 1D array
        Magnetic field data, used as x variable; in Oe units.
    dtH : real scalar, optional
        Magnetic field sweeprate, in Oe/s. The default is dtH0.
    dhT : real scalar, optional
        Initial slope of the MCE effect, in K/Oe. The default is 1e-3.
    Hc : real scalar, optional
        Critical field of the TFIM, in Oe. The default is Hc0.
    H0 : real scalar, optional
        Initial field for the computation, in Oe. The default is Hc0.
    tau : real scalar, optional
        Time constant of thermal relaxation to the bath, in seconds. The default is tau0.
    T0 : real scalar, optional
        Bath temperature, in Kelvin. The default is 1.

    Returns
    -------
    y : TYPE
        DESCRIPTION.

    """
    
    y = T0*np.ones(np.size(H)) # array of temperatures
    a = dtH*tau # magnetic scale of the temperature relaxation
    if dtH>0:
        # y[H>Hc] = H[H>Hc]*np.exp(-H[H>Hc]/a) * (T0/H0*np.exp(H0/a) + T0/a*(expi(H[H>Hc]/a)-expi(H0/a)))
        y[H>Hc] = T0 + dhT*(H[H>Hc]-H0)*np.exp(-(H[H>Hc]-H0)/a)
    elif dtH<0:
        # y[H>Hc] = H[H>Hc]*np.exp(-H[H>Hc]/a) * (T0/H0*np.exp(H0/a) + T0/a*(expi(H[H>Hc]/a)-expi(H0/a)))
        y[H>Hc] = T0 + dhT*(H[H>Hc]-H0)*np.exp(-(H[H>Hc]-H0)/a)
    return y

# def Tdown(H, dtH=-dtH0, Hc=Hc0, Hmax=Hmax0, tau=tau0):
#     y = np.ones(np.size(H)) # array of temperatures
#     y[H>Hc] = 1 + (H[H>Hc]-Hmax)/Hmax*np.exp(-(H[H>Hc]-Hmax)/(dtH*tau))
#     return y



#%% Plot T vs H upsweep
plt.figure(num=0)
plt.plot(H, Tmce(H, dtH=20, dhT=1e-4, tau=40))

#%% Plot T vs H upsweep
plt.figure(num=1)
plt.plot(H, Tmce(H, dtH=-20, dhT=1e-4, H0=Hmax0, tau=40))


#%% Compute physical quantities of MCE as a function of time

intHmax = int(Hmax0//dtH0 + Hmax0%dtH0) # Maximum number of time steps
t = np.linspace(0, intHmax, num=intHmax+1) # time array

Hup = t*dtH0 # array of magnetic fields for upsweep
Hdown = Hmax0-t*dtH0 # array of magnetic fields for downsweep

Tup = np.ones(np.size(t)) # array of temperatures
Tdown = np.ones(np.size(t)) # array of temperatures

# Tup[Hup>Hc] = 1 + dtH0/Hc*(t[Hup>Hc]-t[Hup==Hc])*np.exp(-(t[Hup>Hc]-t[Hup==Hc])/tau)
# Tdown[Hdown>Hc] = 1 - dtH0/Hmax0*(t[Hdown>Hc])*np.exp(-(t[Hdown>Hc])/tau)


#%% Plot data

plt.plot(t,Tup)
plt.plot(t,Tdown)

