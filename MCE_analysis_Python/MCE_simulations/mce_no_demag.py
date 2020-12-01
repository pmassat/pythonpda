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
import numpy as np
from scipy.integrate import solve_ivp
from matplotlib import pyplot as plt
# import plotly.express as px
import plotly.graph_objects as go
from plotly.offline import plot

from tfim_functions import critical_field


#%% Define physical constants of the problem

R = 8.314 # gas constant, in J/K/mol

Hc0 = 5e3 # critical field at T=0, in Oersted
Hmax0 = 1e4 # Max field, in Oe
dtH0 = 10# Sweeprate, in Oe/s
Tc0 = 2.2 # transition temperature at zero field, in Kelvin
kappa = 0.1 # thermal conductivity, in W/K/mol
k1 = Hc0*kappa / (dtH0*R) # equation prefactor

tbath = .8/Tc0 # ratio of bath temperature to Tq0, the transition temperature at zero field
hc = critical_field(tbath) # corresponding critical field, in the mean-field approximation


#%% Function for ODE

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


#%% Solving of ODE

# ODE solving parameters
t0 = np.array([tbath]) # value of reduced temperature at initial field (h = hc for upsweep, h = hmax for downsweep)
hmax = 2
n_points = 1e3
h = np.linspace(hmax/n_points, hmax, int(n_points))

rtom = 7
rel_tol = 10**(-rtom)

# Initialize dictionaries that will contain up- and downsweep solutions
for key in ['solup', 'soldown', 'z_up', 'z_down']:
    if key not in locals():
        exec(f'{key}={{}}')

# Upsweep
h_span_up = (0.1, 2) # interval of integration of the ODE
solup[rtom] = solve_ivp(ode_rhs, h_span_up, t0, args=(k1,tbath,hc), dense_output=True, rtol=rel_tol)


# Downsweep
h_span_down = (2, .1) # interval of integration of the ODE
soldown[rtom] = solve_ivp(ode_rhs, h_span_down, t0, args=(-k1,tbath,hc), dense_output=True, rtol=rel_tol)

# Compute arrays of ODE solution
z_up[rtom] = solup[rtom].sol(h) # continuous solution obtained from dense_output
z_down[rtom] = soldown[rtom].sol(h) # continuous solution obtained from dense_output

# Compute arrays of T-H trace below Hc

# h_low = np.linspace(0, float(hc), 200)
# z_low = tbath*np.ones(np.shape(h_low))

# Compute arrays of full T-H traces

# h = np.concatenate((h_low, h_high))
# z_up = np.concatenate((z_low, zhup[0]))
# z_down = np.concatenate((z_low, zhdn[0]))

#%% Plot ODE solution using plotly
if 'fig' not in locals():
    fig = go.Figure()
fig.add_trace(go.Scatter(x=h*Hc0, y=z_up[rtom][0]*Tc0, name=f'Upsweep rtol={rel_tol:.0e}'))
fig.add_trace(go.Scatter(x=h*Hc0, y=z_down[rtom][0]*Tc0, name=f'Downsweep rtol={rel_tol:.0e}'))
fig.show(renderer="browser")

#%% Plot ODE solution using matplotlib

plt.figure(1)
plt.plot(h*Hc0, z_up[rtom][0]*Tc0)
# plt.plot(soldown.t, np.transpose(soldown.y))

# plt.figure(1)
plt.plot(h*Hc0, z_down[rtom][0]*Tc0)
# plt.plot(soldown.t, np.transpose(soldown.y))

plt.xlim([0, 1e4])


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


