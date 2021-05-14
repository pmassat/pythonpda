# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 15:24:17 2020

@author: Pierre Massat <pmassat@stanford.edu>

Compute and plot the magneto-caloric effect (MCE), i.e. the evolution with magnetic field of the temperature of a sample, in the case where the sample behaves as described by the transverse field Ising model.
See Kohama et al. 2010, equation 3 for a generic equation of the MCE.
See Stinchcombe 1973 for equations of the TFIM.
The final equation that we are solving for is:
    dt/dh = t(h)/h + k1 * (t0-t(h)) * (t(h)/h)**2 * cosh(h/t(h))**2
where t(h) = T(h)/Tc0, h = H/Hc0, and t0 = Tbath/Tc0, with Tc0 and Hc0 being the transition temperature at zero field and the critical field at zero temperature, respectively.

"""

#%% Import functions

# Core libraries
# import os
import time
from time import perf_counter as tpc

# Data analysis
import numpy as np
from scipy.integrate import solve_ivp
from scipy.signal import convolve
from warnings import warn
from matplotlib import pyplot as plt

import plotly.graph_objects as go
# from plotly.offline import plot
from lmfit import Parameters

#%% Local modules
from TFIM_py.tfim_functions import critical_field
# from tfim_functions import critical_field

#%% Define physical constants of the problem

# Hmax0 = 1e4 # Max field, in Oe

def mce_parameters(Hc0=5e3, Hc=None, sweeprate=10, kappa=0.1, Tc0=2.2, Tbath=0.8):
    """
    Create parameters for fit of MCE traces using lmfit.

    Parameters
    ----------
    Hc0 : scalar, optional
        Critical field at T=0, in Oersted units. The default is 5e3.
    Hc : scalar, optional
        Critical field at T=Tbath, in Oersted units. The default is None.
    sweeprate : scalar, optional
        Sweeprate, in Oe/s. The default is 10.
    kappa : scalar, optional
        Thermal conductivity, in W/K/mol. The default is 0.1.
    Tc0 : scalar, optional
        Transition temperature at zero field, in Kelvin units. The default is 2.2.
    Tbath : scalar, optional
        Bath temperature, in Kelvin units. The default is 0.8.

    Returns
    -------
    mce_prms : lmfit Paramaters() object.
        Parameters for fit of MCE traces using lmfit.

    """
    
    R = 8.314 # gas constant, in J/K/mol

    mce_prms = Parameters()

    mce_prms.add('Hc0', value=Hc0, min=4e3, vary=False)
    mce_prms.add('Tc0', value=Tc0, vary=False)
    mce_prms.add('ODE_prefactor', value=Hc0*kappa/(abs(sweeprate)*R), min=0)#
    mce_prms.add('Normalized_bath_temperature', value=Tbath/Tc0, vary=False)#
    mce_prms.add('Hc', value=Hc, vary=False)

    if Hc is None:
        mce_prms['Hc'].value = Hc0*critical_field(Tbath/Tc0)[0]
    else:
        mce_prms['Hc'].value = Hc
        
    return mce_prms


#%% Function for ODE solving
def ode_rhs(t, y, k, yb, tc):
    """
    Define the right hand side of the ODE for the MCE problem corresponding to a physical system subject to the TFIM:
        dy/dt = y/t + k * (y/t)**2 * np.cosh(t/y)**2 * (yb - y)
    where y = T/Tc0 is the reduced temperature
    and t = H/Hc0 is the reduced magnetic field
    y and t are the variables used in the definition of the Python function solve_ivp()
    Note that this equation neglects the phonons' contribution to heat capacity, which is fine for magnetic fields of 50 Oe or more, i.e. essentially always fine for our problem, since the MCE is only relevant at high fields of several hundreds of Oersted or more.

    Parameters
    ----------
    t : scalar
        ODE variable.
    y : scalar
        ODE function to solve.
    k : scalar
'            ODE parameter. Prefactor of the cosh. Corresponds to the physical quantity k1 = Hc0*kappa / (dtH0*R).
    yb : scalar
        ODE parameter. Corresponds to the reduced bath temperature Tbath/Tc0.

    Returns
    -------
    Scalar
        The right hand side of the MCE ODE for a system subject to the TFIM.

    """
    if t>tc:
        return y/t + k * (y/t)**2 * np.cosh(t/y)**2 * (yb - y)
    else:
        return k * (y/t)**2 * np.cosh(t/y)**2 * (yb - y)


#%% Residual function to be minimized
def mce_residual(mce_params, H, data=None, trace='upsweep', mfd_hc=None):
    # unpack parameters: extract .value attribute for each parameter
    parvals = mce_params.valuesdict()
    # Hc0 = parvals['Hc0']
    Hc = parvals['Hc']
    Tc0 = parvals['Tc0']
    # dtH0 = parvals['sweeprate']
    tbath = parvals['Normalized_bath_temperature']
    # hc = critical_field(tbath)# corresponding critical field, in the mean-field approximation
    hc = 1
    
    # Solving of ODE    
    # ODE solving parameters
    t0 = np.array([tbath]) # value of reduced temperature at initial field (min(h) for upsweep, max(h) for downsweep)
    rel_tol = 1e-6 # relative tolerance for solving the ODE
    h = H/Hc # reduced magnetic field data    
    
    if trace=='upsweep':
        h_span = (min(h), max(h)) # interval of integration of the ODE
        k1 = parvals['ODE_prefactor']
    elif trace=='downsweep':
        h_span = (max(h), min(h)) # interval of integration of the ODE
        k1 = - parvals['ODE_prefactor']
    else:
        warn('Unrecognized trace type, no MCE residual output.')

    tic = tpc()

    sol = solve_ivp(ode_rhs, h_span, t0, args=(k1,tbath,hc), dense_output=True, rtol=rel_tol)
    
    toc = tpc()
    print(f'Duration of ODE solving = {toc-tic:.3f} s')
    
    # Compute arrays of ODE solution
    try:
        out = sol.sol(h) # continuous solution obtained from dense_output
    except (IndexError, TypeError):
        # If the data array h has too few datapoints, the dense_output computation throws an IndexError; "too few" here means less than the number of datapoints computed for sol.y (I think).
        out = sol.y
    
    # Code added on 2020-12-18; test for bugs, then remove this comment
    if mfd_hc is None:
        Tout = Tc0 * out[0]
    else:
        Tout = Tc0 * convolve(out[0], mfd_hc[::-1], mode='same') / sum(mfd_hc)
                       
    if data is None:
        return Tout
    else:
        return Tout - data


#%% Objective function for fitting multiple MCE traces
def xmce_residual(mce_params, H, data, traces, mfd_hc=None, Tbath=None):
        
    resid = [None for _ in range(len(H))]
    # rdict = {'updown':{0:'upsweep', 1:'downsweep'},
    #          'downup':{0:'downsweep', 1:'upsweep'}}

    for idx in range(len(H)):
        if Tbath is not None:
            mce_params['Normalized_bath_temperature'].value = Tbath[idx] / \
                mce_params['Tc0'].value
            
        resid[idx] = data[idx] - mce_residual(mce_params, H[idx], 
                                              data=data[idx], mfd_hc=mfd_hc, 
                                              trace=traces[idx])

    return np.concatenate(resid)


#%% Compute bath temperature
def bath_temp(list_of_arrays, rel_temp_var=2.5e-6, rel_temp_bound=1e-3, timeit=False):
    """
    list_of_arrays should be a list of arrays that must include at least a temperature array as its first element
    """
    tic = time.perf_counter()
    T = list_of_arrays[0]
    if len(T)<20:
        warn("Temperature array does not have enough values.")
        return # stop the function
    dT = np.diff(T)
    d = {}
    for idx, elmt in enumerate(list_of_arrays):
        d[idx] = {'m':[], 'b':[], 'bm':[], 'bath':[]}
        d[idx]['m'] = np.mean([elmt[1:], elmt[:-1]], 0)
        while len(d[idx]['b'])<min(len(T)/4, 20):
            d[idx]['b'] = d[idx]['m'][abs(dT/d[0]['m'])<rel_temp_var]
            rel_temp_var = rel_temp_var*2
            if rel_temp_var>1e-2:
                raise DataError("Low-field data is too noisy.")
        d[idx]['bm'] = np.mean(d[idx]['b'])
# Remove        d[idx]['bath'] = d[idx]['b'][abs(d[idx]['b']-d[idx]['bm'])/d[idx]['bm']<rel_temp_bound]
        while len(d[idx]['bath'])<len(d[idx]['b'])/2:
            d[idx]['bath'] = d[idx]['b'][abs(d[idx]['b']-d[idx]['bm'])/d[idx]['bm']<rel_temp_bound]
            rel_temp_bound = rel_temp_bound*2
            if rel_temp_var>5e-2:
                raise DataError("Can't find enough points to compute bath temperature.")
        return_label = 'bath'
    toc = time.perf_counter()
    if timeit is True:
        print(f'Runtime of bath_temp function: {toc - tic:0.4f} seconds')
#     return [d[i][return_label] for i in range(len(d))]# For test purposes
    return np.mean(d[0][return_label])


#%% For given field and temperature, find two closest mfd's
def closest_mfd_values(ptest, upmfd):
    # sort array of fields by distance to current value of field and keep the two closest values
    dpmfd = abs(upmfd-ptest)# 
    closest2p = upmfd[np.argsort(dpmfd)][:2]
    pweights = 1 - dpmfd[np.argsort(dpmfd)][:2]/abs(np.diff(closest2p)[0])

    # if the value of field is below the lowest value in the array or above the highest, 
    # i.e. if the distance between the former and the second closest value in the array is larger than 
    # the distance between the two closest values in the array
    if dpmfd[np.argsort(dpmfd)][1]>=abs(np.diff(closest2p)[0]):
        # only keep the single closest value in the array (i.e. the lowest or the highest)
        closest2p = closest2p[:1]
        pweights = np.ones(1)
        # for instance: if H = 345 and the array is [1000, 2000, 3000], only keep 1000,
        # since abs(2000-345)>abs(2000-1000)
    
    return closest2p, pweights


#%% Compute traces to test
if __name__=='__main__':

    # Initialize dictionaries that will contain up- and downsweep solutions
    for key in ['z_up', 'z_down']:
        if key not in locals():
            exec(f'{key}={{}}')
    
    hmax = 2
    n_points = 2e3
    
    for T0x10 in [8, 12, 16]:
        prms = mce_parameters(Hc0=5e3, sweeprate=10, kappa=0.1, Tc0=2.2, Tbath=T0x10/10)
        Hc0 = prms['Hc0'].value
        Tc0 = prms['Tc0'].value
        H = Hc0*np.linspace(hmax/n_points, hmax, int(n_points))
        z_up[T0x10] = mce_residual(prms, H, data=None, trace='upsweep')
        z_down[T0x10] = mce_residual(prms, H, data=None, trace='downsweep')
    
    
    #%% Plot ODE solution using plotly
    layout = go.Layout(
        xaxis=dict(title='H (Oe)'),
        yaxis=dict(title='T(bath) + deltaT_MCE')
    )
    # if 'fig' not in locals():
    fig = go.Figure(layout=layout)
    for key in z_up.keys():
        fig.add_trace(go.Scatter(x=H, y=z_up[key][0], name=f'Upsweep Tb={key/10}K'))
        fig.add_trace(go.Scatter(x=H, y=z_down[key][0], name=f'Downsweep Tb={key/10}K'))
    
    # fig.update_layout(title='Average High and Low Temperatures in New York',
    #                    xaxis_title='Month',
    #                    yaxis_title='Temperature (degrees F)')
    
    fig.show(renderer="browser")
    
    #%% Plot ODE solution using matplotlib
    plt.figure()
    # hold on
    for key in z_up.keys():
        tbkey = f'$T_{{bath}}={key/10}\,$K'
        plt.plot(H, z_up[key][0], label=''.join(['Upsweep ', tbkey]))
        plt.plot(H, z_down[key][0], label=''.join(['Downsweep ', tbkey]))
    # plt.xlim([0, 1e4])
    plt.xlabel('$H$ (Oe)')
    plt.ylabel('$T_\mathrm{bath} + \Delta T_\mathrm{MCE}$')
    plt.legend(loc='upper left')
    plt.title(f"Simulated MCE traces at a sweeprate of {prms['sweeprate'].value} Oe")
    
    #%% Show available Plotly renderers
    # import plotly.io as pio
    # print(pio.renderers)
    
    
    #%% Test of solve_ivp
    def exponential_decay(t, y): return -0.5 * y
    
    sol = solve_ivp(exponential_decay, [0, 10], [2], dense_output=True)
    
    t = np.linspace(0,10)
    z = sol.sol(t) # continuous solution obtained from dense_output
    
    plt.figure(1)
    plt.plot(t, z.T)
    
    
