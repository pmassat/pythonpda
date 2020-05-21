# -*- coding: utf-8 -*-
"""
Created on Wed May 20 15:38:58 2020

@author: Pierre Massat <pmassat@stanford.edu>
"""

import numpy as np
import pandas as pd

np.seterr('raise')

def averageCpwithH2(relTsep, Temperature, Cp, CpErr, H):
    """
    Average datapoints taken at any given temperature and field setpoint
    We want to compute the average of data points that are repetitions of the
    same measurement, i.e. with same temperature and field setpoints
    relTsep; temperature stability reported in the PPMS manual is <0.2\%
    (should be called temperature instability). Hence data points taken 
    within a relative interval of relTsep are considered to be measured at the same temperature 

    Parameters
    ----------
    relTsep : float
        Relative temperature separation between two datapoints measured at two
        different temperature setpoints. Should be relTsep <= 0.002 according 
        to the PPMS manual.
    T : NumPy array
        Temperature data
    Cp : NumPy array
        Heat capacity data. Should have same size as T.
    CpErr : NumPy array
        Heat capacity uncertainty data. Should have same size as T.
    H : NumPy array
        Magnetic field data. Should have same size as T.

    Returns
    -------
    Pandas DataFrame of Cp data where all datapoints taken at the same temperature
    setpoint have been averaged.

    """

    T = np.copy(Temperature.to_numpy())
    # print(T.shape)
    Tm = np.zeros(T.shape)
    Tsd = np.zeros(T.shape)
    Cpm = np.zeros(Cp.shape)
    Cpsd = np.zeros(Cp.shape)
    CpmErr = np.zeros(CpErr.shape)

    for k in range(len(T)):
        # print(k)
        if T[k]==0:
            continue
        elif len(T[np.abs(T-T[k])/T[k] < relTsep])>3:
    # T[full_selector] is the subset of temperatures which have
    # a relative separation of relTsep from the temperature of datapoint #k
    # If this subset contains more than 3 data points...
            halfrelTsep = relTsep/2# ... reduce the temperature interval
    #             T(np.abs(T-T[k])<Tsep2)%print out values of temperature which
    #             verify the if statement
            half_selector = np.abs(T-T[k])/T[k] < halfrelTsep
            Tm[k] = np.mean(T[half_selector])
            Tsd[k] = np.std(T[half_selector])
            Cpm[k] = np.mean(Cp[half_selector])
            Cpsd[k] = np.std(Cp[half_selector])
            CpmErr[k] = np.sum(CpErr[half_selector]) / np.sqrt(len(CpErr[half_selector]))
            T[half_selector]=0
        else:
            full_selector = np.abs(T-T[k])/T[k] < relTsep
            Tm[k] = np.mean(T[full_selector])
            Tsd[k] = np.std(T[full_selector])
            Cpm[k] = np.mean(Cp[full_selector])
            Cpsd[k] = np.std(Cp[full_selector])
            CpmErr[k] = np.sum(CpErr[full_selector])/ np.sqrt(len(CpErr[full_selector]))
            T[full_selector]=0

    S = pd.DataFrame()
    S['H'] = H[Tm>0]
    S['T'] = Tm[Tm>0]
    S['Tsd'] = Tsd[Tm>0]
    S['Cp'] = Cpm[Tm>0]
    S['CpFullErr'] = Cpsd[Tm>0] + CpmErr[Tm>0]
    return S
    
    
    
    
    
    
    
      