# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 18:15:56 2020

@author: Pierre Massat <pierre.massat@crans.org>
"""

from numpy import sqrt, exp, log, imag, pi
from scipy.special import erfc, exp1

def pVIC_residual(x, params, data, eps=None):
    # unpack parameters: extract .value attribute for each parameter
    parvals = params.valuesdict()# params should be an object of the Parameter 
    # class, as defined in the lmfit module
    A = parvals['A'];# amplitude of the peak
    alpha = parvals['alpha'];# IC fast decay parameter
    beta = parvals['beta'];# IC slow decay parameter
    R = parvals['R'];# IC weight ratio between fast and slow decays
    gamma = parvals['gamma'];# pV Lorentzian width
    sigma = parvals['sigma'];# pV Gaussian width
    k = parvals['k'];# IC "approximation" parameter that is set to 0.05 (not sure what it does)
    x0 = parvals['peakPosition'];# position of peak, not max; 
    # more precisely, x0 is roughly the position of upturn (max of second derivative) 
    # of the lefthand side of the peak

    model = pVIC(x, A, alpha, beta, R, gamma, sigma, k, x0)
    if eps is None:
        return model - data
    return (model-data) / eps

def pVIC(x, A, alpha, beta, R, gamma, sigma, k, x0):
    """
    The pseudo-Voigt-Ikeda-Carpenter function is the convolution of a pseudo-Voigt
    profile with the Ikeda-Carpenter function. 
    See Nuclear Instruments and Methods in Physics Research, A239 (1985) 536-544
    and the "time of flight" computation behind the FullProf software, explained
    at http://www.ccp14.ac.uk/ccp/web-mirrors/plotr/Tutorials&Documents/TOF_FullProf.pdf
    """
    xr = x - x0;# position of current datapoint in reciprocal space, relative to peak position
    
    def pseudoVoigtFWHM(gamma, sigma):
        fL = 2*gamma; 
        fG = 2*sigma*sqrt(2*log(2));
        Gamma = (fG^5 + 2.69269*fG^4*fL + 2.42843*fG^3*fL^2 + \
            4.47163*fG^2*fL^3 + 0.07842*fG*fL^4 + fL^5)^(1/5);
        return Gamma
        
    def pseudoVoigtEta(gamma, sigma):
        fL = 2*gamma;
        f = pseudoVoigtFWHM(gamma, sigma);
        fLbyf = fL/f;
        eta = 1.36603*fLbyf - 0.47719*fLbyf^2 + 0.11116*fLbyf^3;
        return eta
    
    Gamma = pseudoVoigtFWHM(gamma, sigma);
    eta = pseudoVoigtEta(gamma, sigma);
    
    sigmaSq = Gamma**2/(8*log(2));
    gWidth = sqrt(2*sigmaSq);
    
    am = alpha*(1 - k);
    ap = alpha*(1 + k);
    
    x = am - beta;
    y = alpha - beta;
    z = ap - beta;
    
    zs = -alpha*xr + 1j*alpha*Gamma/2;
    zu = (1 - k)*zs;
    zv = (1 + k)*zs;
    zr = -beta*xr + 1j*beta*Gamma/2;
    
    u = am*(am*sigmaSq - 2*xr)/2;
    v = ap*(ap*sigmaSq - 2*xr)/2;
    s = alpha*(alpha*sigmaSq - 2*xr)/2;
    r = beta*(beta*sigmaSq - 2*xr)/2;
    
    n = (1/4)*alpha*(1 - k**2)/k**2;
    nu = 1 - R*am/x;
    nv = 1 - R*ap/z;
    ns = -2*(1 - R*alpha/y);
    nr = 2*R*alpha**2*beta*k**2/(x*y*z);
    
    yu = (am*sigmaSq - xr)/gWidth;
    yv = (ap*sigmaSq - xr)/gWidth;
    ys = (alpha*sigmaSq - xr)/gWidth;
    yr = (beta*sigmaSq - xr)/gWidth;
    
    val = A*n*((1 - eta)*(nu*exp(u)*erfc(yu) + nv*exp(v)*erfc(yv)\
        + ns*exp(s)*erfc(ys) + nr*exp(r)*erfc(yr))\
        - eta*2/pi*(imag(nu*exp1(zu)*exp(zu)) + imag(nv*exp1(zv)*exp(zv))\
        + imag(ns*exp1(zs)*exp(zs)) + imag(nr*exp1(zr)*exp(zr))));

    return val