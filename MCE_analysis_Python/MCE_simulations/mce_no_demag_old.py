# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 14:52:13 2020

Outdated as of 2020-11-30.

@author: Pierre Massat <pmassat@stanford.edu>

Compute and plot the magneto-caloric effect (MCE), i.e. the evolution with magnetic field of the temperature of a sample, in the case where the sample behaves as described by the transverse field Ising model.
See Kohama et al. 2010, equation 3 for a generic equation of the MCE.
See Stinchcombe 1973 for equations of the TFIM.
The final equation that we are solving for is:
    dt/dh = t(h)/h + (t0-t(h))*k1*(t0/h)**2*cosh(h/t0)**2
where t(h) = T(h)/Tc0, h = H/Hc0, and t0 = Tbath/Tc0, with Tc0 and Hc0 being the transition temperature at zero field and the critical field at zero temperature, respectively.
In the above equation, t0/h and h/t0 should be t(h)/h and h/t(h) but we chose to approximate t(h) in these terms, otherwise the solution is too complicated (and Wolfram Alpha cannot solve it)


"""

#%% Import functions
import numpy as np
from scipy.special import shichi # expi
from scipy.integrate import solve_ivp
from matplotlib import pyplot as plt

#%% Functions used in the equation solution

def exparg(h, k=k1, tb=tbath):
    """
    Argument of the exponentials in the solution of the MCE equation where the magnetic field dependence of the heat capacity C is taken into account.
    See Wolfram alpha for a human readable version of the solution of the equation

    Parameters
    ----------
    h : array
        Reduced magnetic field H/Hc0.
    k : scalar, optional
        Prefactor of the second term in the RHS of the equation. The default is k1.
    tb : scalar, optional
        Reduced bath temperature Tbat/Tc0. The default is tbath.

    Returns
    -------
    y : array
        Final computed quantity.

    """
    y = - k * tb * shichi(2*h/tb)[0] + \
        k * tb**2 / (2*h) * (1 + np.cosh(2*h/tb))
    return y


def solint(h, k=k1, tb=tbath):
    





#%% 

H = np.linspace(0, Hmax0, num=int(Hmax0)+1) # time array


#%% 
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


