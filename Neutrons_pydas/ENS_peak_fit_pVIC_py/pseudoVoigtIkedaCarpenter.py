# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 18:15:56 2020

@author: Pierre Massat <pierre.massat@crans.org>
"""

import numpy as np
from scipy.special import erfc, exp1


def pVIC_residual(x, params, data, eps=None):
    """
    Note: this function is unnecessary since a 'Model' object as defined in the lmfit 
    module can be created directly from the 'pVIC' fit function and will be 
    much more convenient and powerful for the fitting procedure.
    
    Residual function corresponding to the 'pVIC' function.

    Parameters
    ----------
    x : Numerical value
        Independent 'pVIC' function variable, which is the position on the x axis.
    params : Parameter object, as defined in the lmfit module
        List of parameter
    data : 1D numerical array 
        Data to fit
    eps : 1D numerical array , optional
        Uncertainties on the data. The default is None.

    Returns
    -------
    1D numerical array
        Residual, which is the difference between the model function and the 
        data, divided by the uncertainties, if they are provided.

    """
    # unpack parameters: extract .value attribute for each parameter
    parvals = params.valuesdict()# params should be an object of the 
    A = parvals['A']# amplitude of the peak
    alpha = parvals['alpha']# IC fast decay parameter
    beta = parvals['beta']# IC slow decay parameter
    R = parvals['R']# IC weight ratio between fast and slow decays
    gamma = parvals['gamma']# pV Lorentzian width
    sigma = parvals['sigma']# pV Gaussian width
    k = parvals['k']# IC "approximation" parameter that is set to 0.05 (not sure what it does)
    xp = parvals['peakPosition']# position of peak, not max 
    # more precisely, xp is roughly the position of upturn (max of second derivative) 
    # of the lefthand side of the peak

    model = pVIC(x, A, alpha, beta, R, gamma, sigma, k, xp)
    if eps is None:
        return model - data
    return (model-data) / eps


def pVIC(x, A, alpha, beta, R, gamma, sigma, k, xp):
    """
    The pseudo-Voigt-Ikeda-Carpenter function is the convolution of a pseudo-Voigt
    profile with the Ikeda-Carpenter function. 
    See Nuclear Instruments and Methods in Physics Research, A239 (1985) 536-544
    and the "time of flight" computation behind the FullProf software, explained
    at http://www.ccp14.ac.uk/ccp/web-mirrors/plotr/Tutorials&Documents/TOF_FullProf.pdf

    Parameters
    ----------
    x : Numerical value (float)
        Independent 'pVIC' function variable, which is the position on the x axis.
    A : Float
        Amplitude of the pseudo-Voigt-Ikeda-Carpenter function.
    alpha : Float
        IC (Ikeda-Carpenter) fast decay parameter.
    beta : Float
        IC slow decay parameter.
    R : Float
        IC weight ratio between fast and slow decays.
    gamma : Float
        pV (pseudo-Voigt) Lorentzian width.
    sigma : Float
        pV Gaussian width.
    k : Float
        IC "approximation" parameter (not sure what it does).
        Its value is set to 0.05 by default.
    xp : Float
        Position of peak (not max).

    Returns
    -------
    TYPE: float
        Value of the pseudo-Voigt-Ikeda-Carpenter function at position x in reciprocal space.

    """
    xr = x - xp# position of current datapoint in reciprocal space, relative to peak position
    
    def pseudoVoigtFWHM(gamma, sigma):
        fL = 2*gamma 
        fG = 2*sigma*np.sqrt(2*np.log(2))
        Gamma = (fG**5 + 2.69269*fG**4*fL + 2.42843*fG**3*fL**2 \
            + 4.47163*fG**2*fL**3 + 0.07842*fG*fL**4 + fL**5)**(1/5)
        return Gamma
        
    def pseudoVoigtEta(gamma, sigma):
        fL = 2*gamma
        f = pseudoVoigtFWHM(gamma, sigma)
        fLbyf = fL/f # note: because of this line, one cannot have both gamma and sigma set to zero
        eta = 1.36603*fLbyf - 0.47719*fLbyf**2 + 0.11116*fLbyf**3
        return eta
    
    Gamma = pseudoVoigtFWHM(gamma, sigma)
    eta = pseudoVoigtEta(gamma, sigma)
    
    sigmaSq = Gamma**2/(8*np.log(2))
    gWidth = np.sqrt(2*sigmaSq)
    
    am = alpha*(1 - k)
    ap = alpha*(1 + k)
    
    x = am - beta
    y = alpha - beta
    z = ap - beta
    
    zs = -alpha*xr + 1j*alpha*Gamma/2
    zu = (1 - k)*zs
    zv = (1 + k)*zs
    zr = -beta*xr + 1j*beta*Gamma/2
    
    u = am*(am*sigmaSq - 2*xr)/2
    v = ap*(ap*sigmaSq - 2*xr)/2
    s = alpha*(alpha*sigmaSq - 2*xr)/2
    r = beta*(beta*sigmaSq - 2*xr)/2
    
    n = (1/4)*alpha*(1 - k**2)/k**2
    nu = 1 - R*am/x
    nv = 1 - R*ap/z
    ns = -2*(1 - R*alpha/y)
    nr = 2*R*alpha**2*beta*k**2/(x*y*z)
    
    yu = (am*sigmaSq - xr)/gWidth
    yv = (ap*sigmaSq - xr)/gWidth
    ys = (alpha*sigmaSq - xr)/gWidth
    yr = (beta*sigmaSq - xr)/gWidth
    
    val = A*n*((1 - eta)*(nu*np.exp(u)*erfc(yu) + nv*np.exp(v)*erfc(yv)\
        + ns*np.exp(s)*erfc(ys) + nr*np.exp(r)*erfc(yr))\
        - eta*2/np.pi*(np.imag(nu*exp1(zu)*np.exp(zu)) + np.imag(nv*exp1(zv)*np.exp(zv))\
        + np.imag(ns*exp1(zs)*np.exp(zs)) + np.imag(nr*exp1(zr)*np.exp(zr))))

    return val