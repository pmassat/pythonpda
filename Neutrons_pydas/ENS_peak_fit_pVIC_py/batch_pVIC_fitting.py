# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 16:09:08 2020

@author: Pierre Massat <pmassat@stanford.edu>

Define batch fitting functions used in 'ENS_peak_fit_pVIC_2019-02-14.ipynb'

"""

import copy as cp, numpy as np
from matplotlib import pyplot as plt
from lmfit import Parameters
from ENS_peak_fit_pVIC_py.pseudoVoigtIkedaCarpenter import pVIC, xnpVIC_init_prm
# from ENS_peak_fit_pVIC_py.xpvic_fit_cls import xpvic_fit


def fixSharedParams(freeSharedParams, fitResult):
    fixedParams = {}

    # Find all parameters with a relative error of more than 100% in the last fit result, if any
    for prmKey in reversed(freeSharedParams):
        prmRelErr = fitResult.params[prmKey].stderr / fitResult.params[prmKey].value
        if prmRelErr > 1:
            fixedParams[prmKey] = 1e-10
            freeSharedParams.remove(prmKey)
    
    # If all shared parameters have a relative error of less than 100%
    # Store them in fixParams and empty freeSharedParams
    if len(fixedParams)==0:
        for prmKey in freeSharedParams:
            fixedParams[prmKey] = None
        freeSharedParams = []
    
    return freeSharedParams, fixedParams


# The following three functions are kept for records but have been included  in other modules

# Included in the xnpVIC_init_prm function of the 'pseudoVoigtIkedaCarpenter' module
def npvicResidual(params, x, data, data_range, nFunc):
#     model = np.sum([pVIC(x, *list(params.valuesdict().values())[idx*8:(idx+1)*8]) for idx in range(n)], axis=0)
    model = np.sum([pVIC(x, *xnpVIC_init_prm(params, fun_idx, dat_idx)) 
                    for fun_idx in range(nFunc) for dat_idx in data_range], axis=0)
    return data - model


# Included in the initParams method of the xpvic_fit class of the 'xpvic_fit_cls' module
def npvicInitParams(refParams, spec_idx, nFunc, xp=None, A=None):
    if xp is None:
        xp = np.array([refParams['xp'].value+0.015*(nFunc-1-2*idx) for idx in range(nFunc)])
    if A is None:
        A = np.array([refParams['A'].value/np.sqrt(nFunc) for _ in range(nFunc)])

    params = Parameters()
    for idx in range(nFunc):
        for key in refParams.keys():
            if key in ['A', 'xp']:
                try:
                    params.add(f"{key}{idx}_{spec_idx}", value=eval(key)[idx], vary=True)
                except:
                    raise TypeError(f"{key} must be an iterable object of length {nFunc}")
            else:
                params[key] = cp.copy(refParams[key])
    
    return params



# Functions below this point are deprecated and have been replaced by the 
# xpvic_fit class of the 'xpvic_fit_cls' module

def xyBatchFitNData(nData, data_range, data_select=None):
    """
    Create x- and y-axis arrays of data for batch fitting.
    
    Parameters
    ----------
    nData : Pandas dataframe
        Dataframe where each row of the column called "spectra" contains a 
        dataframe with ENS data, where hh0 is the x-axis data and Inorm is the y-axis data
    data_range : Range
        Range of indices of nData datasets to include in fit.
    data_select : Boolean NumPy array, optional
        Array with same size as nData.spectra[row_index].hh0, with ones where 
        data should be used for fitting, and zeros elsewhere.
        The default is None.

    Returns
    -------
    X : NumPy array
        Array of x-axis arrays for batch fitting.
    Y : NumPy array
        Array of y-axis arrays for batch fitting.

    """
    
    if data_select is None:
        X = np.stack([nData.spectra[idx].hh0 for idx in data_range])
        Y = np.stack([nData.spectra[idx].Inorm for idx in data_range])
    else:
        X = np.stack([nData.spectra[idx].hh0[data_select] for idx in data_range])
        Y = np.stack([nData.spectra[idx].Inorm[data_select] for idx in data_range])
    return X, Y


def xFitInitParams(fitParams, refParams, data_range, resultParams=None):
    """
    Initialize parameters for the next fitting iteration using the results of the previous fit
    and, if necessary, the default values of a reference set of parameters    

    Parameters
    ----------
    fitParams : lmfit Parameters object
        Fit parameters to initialize.
    refParams : lmfit Parameters object
        Reference Parameters object, containing default values and a fixed 
        number of Parameter objects, as defined from the model fit function.
    data_range : Range
        Range of indices of nData datasets to include in fit.
    resultParams : lmfit Parameters object, optional
        Parameters object yielded by the previously performed fit, if any.
        The default is None.

    Returns
    -------
    fitParams : TYPE
        DESCRIPTION.

    """
    # For those parameters that have been computed in the last run, 
    # use as initial values for the next run the best fit values obtained in the last
    if resultParams is not None:
        for key in fitParams.keys():
            try:
                fitParams[key].value = resultParams[key].value
            except KeyError: # in case fitParams has been modified since last fitting run
                continue

    # Create additional fit parameters, e.g. if the number of datasets has been extended
    for spec_idx in data_range:
    # loop over indices of datasets, in order to create fit parameters for each of them
        for k, _ in refParams.items():
            if k in ['A', 'xp']:
            # fit parameters that are different for each dataset are assigned individual names
                par_key = f'{k}{spec_idx}'
                if par_key not in fitParams.keys():
                    fitParams.add( par_key, value=refParams[k].value, 
                                   min=refParams[k].min, vary=refParams[k].vary )
            elif resultParams is None: # if there are no shared fit parameters, because no fit has been performed yet
            # all other parameters are shared by all datasets and are assigned the "generic" name from refParams
                fitParams[k] = cp.copy(refParams[k])
    return fitParams


def xFitInitParams2(refParams, data_range, resultParams=None):
    """
    Initialize parameters for the next fitting iteration using the results of the previous fit
    and, if necessary, the default values of a reference set of parameters    

    Parameters
    ----------
    fitParams : lmfit Parameters object
        Fit parameters to initialize.
    refParams : lmfit Parameters object
        Reference Parameters object, containing default values and a fixed 
        number of Parameter objects, as defined from the model fit function.
    data_range : Range
        Range of indices of nData datasets to include in fit.
    resultParams : lmfit Parameters object, optional
        Parameters object yielded by the previously performed fit, if any.
        The default is None.

    Returns
    -------
    fitParams : TYPE
        DESCRIPTION.

    """
    # Initialize lmfit Parameters object
    fitParams = Parameters()
    # For those parameters that have been computed in the last run, 
    # use as initial values for the next run the best fit values obtained in the last
    if resultParams is not None:
        for key in resultParams.keys():
            # try:
            fitParams[key] = cp.copy(resultParams[key])
            # except KeyError: # in case fitParams has been modified since last fitting run
            #     continue

    # Create additional fit parameters, e.g. if the number of datasets has been extended
    for spec_idx in data_range:
    # loop over indices of datasets, in order to create fit parameters for each of them
        for k in refParams.keys():
            if k in ['A', 'xp']:
            # fit parameters that are different for each dataset are assigned individual names
                par_key = f'{k}{spec_idx}'
                if par_key not in fitParams.keys():
                    fitParams.add( par_key, value=refParams[k].value, 
                                   min=refParams[k].min, vary=refParams[k].vary )
            elif resultParams is None: # if there are no shared fit parameters, because no fit has been performed yet
            # all other parameters are shared by all datasets and are assigned the "generic" name from refParams
                fitParams[k] = cp.copy(refParams[k])
    return fitParams


def bestFitParams(data_range, refParams, fitResultParams):
    """
    Create NumPy array of best fit parameter values for each fitted curve.

    Parameters
    ----------
    data_range : Range
        Range of indices of nData datasets to include in fit.
    refParams : lmfit Parameters object
        Reference Parameters object, containing default values and a fixed 
        number of Parameter objects, as defined from the model fit function.
    fitResultParams : lmfit Parameters object, optional
        Parameters object yielded by a previously performed fit.

    Returns
    -------
    bestparams : NumPy array
        Array of arrays of best fit parameter values for each fitted curve.

    """
    num_spec = len(data_range)
    bestparams = np.zeros((num_spec,len(refParams)))
    # bestparams.shape = # of datasets x # of parameters in fit function (pVIC)
    for spec_idx in range(num_spec):
        for par_idx, refKey in enumerate(refParams.keys()):
            par_key = f'{refKey}{data_range[spec_idx]}' 
            # parameter name is a concatenation of the generic parameter name,
            # as defined in the refParams function, and the spectrum index
            try:
                bestparams[spec_idx][par_idx] = fitResultParams[par_key].value
            except KeyError:
                bestparams[spec_idx][par_idx] = fitResultParams[refKey].value
    return bestparams


def plotMultipleFit(data_range, xfitdata, yfitdata, fitParams, bestFitParams, fieldLabel ):
    """
    Plot multiple datasets with the corresponding fits.

    Parameters
    ----------
    data_range : Range
        Range of indices of nData datasets to include in fit.
    xfitdata : NumPy array
        Array of arrays of x-axis data used for the fit.
    yfitdata : NumPy array
        Array of arrays of y-axis data used for the fit.
    fitParams : lmfit Parameters object
        Parameters used in fit, with information on which were free to vary, 
        which will be used to determine the number of free parameters in the title of the plot.
    bestFitParams : NumPy array
        Array of arrays of best fit parameter values for each fitted curve, 
        as calculated by the bestFitParams function
    fieldLabel : Pandas series / dataframe column
        Each row contains a numerical string indicating the value of magnetic 
        field at which the corresponding dataset was measured (see nData dataframe for more info).

    Returns
    -------
    None.

    """
    plt.figure()
    for idx, spec_idx in enumerate(data_range):
        # if idx % (len(data_range)//5) == 0: # this can be done by slicing data_range directly when calling the function
        bestfit = pVIC(xfitdata[idx], *bestFitParams[idx][:8])
        p = plt.plot(xfitdata[idx], yfitdata[idx], 'o', label=f"expt {fieldLabel[spec_idx]:.3g}T")
        plt.plot(xfitdata[idx], bestfit, '-', color=p[-1].get_color(), label=f"fit {fieldLabel[spec_idx]:.3g}T")
        plt.legend(loc='best')
    plt.show()
    freeParams = [k for k in list(fitParams.keys()) if fitParams[k].vary==True]
    plt.title(f"TmVO$_4$ neutrons {len(freeParams)} free parameters")
    

