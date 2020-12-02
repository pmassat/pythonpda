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

# Official Python libraries
import numpy as np
from scipy.integrate import solve_ivp
from warnings import warn
from matplotlib import pyplot as plt
# import plotly.express as px
import plotly.graph_objects as go
# from plotly.offline import plot
from lmfit import minimize, Parameters

# Local modules
from tfim_functions import critical_field


#%% Define physical constants of the problem

# Hmax0 = 1e4 # Max field, in Oe

def mce_parameters(Hc0=5e3, sweeprate=10, kappa=0.1, Tc0=2.2, Tbath=0.8):
    R = 8.314 # gas constant, in J/K/mol
    mce_prms = Parameters()
    mce_prms.add('Hc0', value=Hc0, min=0)# critical field at T=0, in Oersted
    mce_prms.add('sweeprate', value=sweeprate, vary=False)# Sweeprate, in Oe/s
    mce_prms.add('thermal_conductivity', value=kappa, min=0)# thermal conductivity, in W/K/mol
    mce_prms.add('ODE_prefactor', value=Hc0*kappa/(sweeprate*R))# ODE prefactor
    mce_prms.add('Tc0', value=Tc0, vary=False)# transition temperature at zero field, in Kelvin
    mce_prms.add('Normalized_bath_temperature', value=Tbath/Tc0)# Normalized bath temperature
    return mce_prms


#%% Residual function to be minimized
def mce_residual(mce_params, H, data=None, trace='upsweep'):
    # unpack parameters: extract .value attribute for each parameter
    parvals = mce_params.valuesdict()
    Hc0 = parvals['Hc0']
    Tc0 = parvals['Tc0']
    # dtH0 = parvals['sweeprate']
    k1 = parvals['ODE_prefactor']
    tbath = parvals['Normalized_bath_temperature']
    hc = critical_field(tbath) # corresponding critical field, in the mean-field approximation
    
    # Function for ODE solving
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
            ODE parameter. Prefactor of the cosh. Corresponds to the physical quantity k1 = Hc0*kappa / (dtH0*R).
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
        
    # Solving of ODE    
    # ODE solving parameters
    t0 = np.array([tbath]) # value of reduced temperature at initial field (min(h) for upsweep, max(h) for downsweep)
    rtom = 6 # Exponent of the relative tolerance for solving the ODE
    rel_tol = 10**(-rtom) # relative tolerance for solving the ODE
    h = H/Hc0 # reduced magnetic field data    
    
    if trace=='upsweep':
        h_span_up = (min(h), max(h)) # interval of integration of the ODE
        solup = solve_ivp(ode_rhs, h_span_up, t0, args=(k1,tbath,hc), dense_output=True, rtol=rel_tol)
        # Compute arrays of ODE solution
        z_up = solup.sol(h) # continuous solution obtained from dense_output
        if data is None:
            return z_up*Tc0
        else:
            return z_up*Tc0 - data
    elif trace=='downsweep':
        h_span_down = (max(h), min(h)) # interval of integration of the ODE
        soldown = solve_ivp(ode_rhs, h_span_down, t0, args=(-k1,tbath,hc), dense_output=True, rtol=rel_tol)
        z_down = soldown.sol(h) # continuous solution obtained from dense_output
        if data is None:
            return z_down*Tc0
        else:
            return z_down*Tc0 - data
    else:
        warn('Unrecognized trace type, no MCE residual output.')

#%% Compute traces
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


