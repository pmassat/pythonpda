# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 16:09:08 2020

@author: Pierre Massat <pmassat@stanford.edu>

Define batch fitting functions used in 'ENS_peak_fit_pVIC_2019-02-14.ipynb'

"""

import copy as cp, numpy as np, warnings
from matplotlib import pyplot as plt
from lmfit import Parameters, minimize
from ENS_peak_fit_pVIC_py.pseudoVoigtIkedaCarpenter import pVIC, xpVIC_residual

np.seterr(divide='warn')

class xpvic_fit:
    """
    Class of elements used when batch fitting neutrons diffraction data with
    the pseudo-Voigt Ikeda-Carpenter function.
    """
    
    def __init__(self, h, data, refParams):
        """
        Initialize attributes of the class
        
        Parameters
        ----------
        data : Pandas dataframe
            Dataframe where each row of the column called "spectra" contains a 
            dataframe with ENS data, where hh0 is the x-axis data and Inorm is 
            the y-axis data. See the xyData() method.
        data_range : Range
            Range of indices of nData datasets to include in fit.
        
        refParams : lmfit Parameters object
            Reference Parameters object, containing default values and a fixed 
            number of Parameter objects, as defined from the model fit function.
        """

        # Metadata of the fit: peak position in (h h 0) reciprocal space
        self.h = h
        self.hkl = f"({h} {h} 0)"
        
        # Data to fit
        self.data = data
        
        # Create range of data to fit and plot
        self.data_range = range(len(self.data)) # by default, use the full range of data
        self.plot_range = self.data_range # same for plotting
        
        # Set of reference parameters to use for the fit
        self.refParams = refParams


    def makeData(self, data_select=None): # formerly named xyBatchFitNData
        """
        Create x- and y-axis arrays of data for batch fitting.
        
        Parameters
        ----------
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
        # Initialize arrays with zero line and as many columns as there are data in each spectrum
        # This is only useful if weigths should be calculated differently for different spectra
        # self.X = np.empty((0,len(self.data.spectra[self.data_range[0]].hh0)))
        # self.Y = np.empty(self.X.shape)
        # self.dY = np.empty(self.X.shape)
        
        # If no data selection filter is applied, select all the data
        if data_select is None:
            data_select = np.ones(self.data.spectra[self.data_range[0]].hh0.shape, dtype=bool)
        
        # Create x, y and dy data arrays
        self.X = np.stack([self.data.spectra[idx].hh0[data_select] for idx in self.data_range])
        self.Y = np.stack([self.data.spectra[idx].Inorm[data_select] for idx in self.data_range])
        self.dY = np.stack([self.data.spectra[idx].dInorm[data_select] for idx in self.data_range])

        # Compute weights from data errors
        self.weights = 1/(self.dY)**2
        # Set all np.inf values in self.weights to zero
        if np.any(self.weights==np.inf):
            self.weights[self.weights==np.inf] = 0
            # print(np.argwhere(self.weights==np.inf))
            warnings.warn(f"np.inf values were found in 'weights'. They were reset to zero.")

        for idx in self.data_range:
            if not np.any(self.dY):
                warnings.warn(f"All errors are zero in spectrum with index {idx}; \
                              using all ones as weights.")
                self.weights[idx] = np.ones(self.weights[idx].shape)
        
    
    def initParams(self, resultParams=None): # formerly named xFitInitParams2
        """
        Initialize parameters for the next fitting iteration using the results of the previous fit
        and, if necessary, the default values of a reference set of parameters    
    
        Parameters
        ----------
        resultParams : lmfit Parameters object, optional
            Parameters object yielded by the previously performed fit, if any.
            The default is None.
    
        Returns
        -------
        self.init_params : lmfit Parameters object
            Parameters to be used in the fit of all curves.
        """
        # Initialize lmfit Parameters object
        self.init_params = Parameters()
        # For those parameters that have been computed in the last run, 
        # use as initial values for the next run the best fit values obtained in the last
        if resultParams is not None:
            for key in resultParams.keys():
                # try:
                self.init_params[key] = cp.copy(resultParams[key])
                # except KeyError: # in case self.init_params has been modified since last fitting run
                #     continue
    
        # Create additional fit parameters, e.g. if the number of datasets has been extended
        for spec_idx in self.data_range:
        # loop over indices of datasets, in order to create fit parameters for each of them
            for k in self.refParams.keys():
                if k in ['A', 'xp']:
                # fit parameters that are different for each dataset are assigned individual names
                    par_key = f'{k}{spec_idx}'
                    if par_key not in self.init_params.keys():
                        self.init_params.add( par_key, value=self.refParams[k].value, 
                                       min=self.refParams[k].min, vary=self.refParams[k].vary )
                elif resultParams is None: # if there are no shared fit parameters, because no fit has been performed yet
                # all other parameters are shared by all datasets and are assigned the "generic" name from self.refParams
                    self.init_params[k] = cp.copy(self.refParams[k])


    def performFit(self, weights=None):
        """
        Perfor fit using the xpVIC_residual function, which is the residual 
        function for fitting multiple curves using the pseudo-Voigt-Ikeda-Carpenter
        functional form.

        Returns
        -------
        self.result: Minimizer.Result object from the lmfit module
            Result of the minimize function, containing the fit results.

        """
        self.result = minimize(xpVIC_residual, self.init_params, 
                               args=(self.X, self.Y, self.data_range, weights))

    
    def bestFitParams(self):
        """
        Create NumPy array of best fit parameter values for each fitted curve.
    
        Returns
        -------
        self.bestparams : NumPy array
            Array of arrays of best fit parameter values for each fitted curve.
    
        """
        num_spec = len(self.data_range)
        self.bestparams = np.zeros((num_spec,len(self.refParams)))
        # self.bestparams.shape = # of datasets x # of parameters in fit function (pVIC)
        for spec_idx in range(num_spec):
            for par_idx, refKey in enumerate(self.refParams.keys()):
                par_key = f'{refKey}{self.data_range[spec_idx]}' 
                # parameter name is a concatenation of the generic parameter name,
                # as defined in the self.refParams function, and the spectrum index
                try:
                    self.bestparams[spec_idx][par_idx] = self.result.params[par_key].value
                except KeyError:
                    self.bestparams[spec_idx][par_idx] = self.result.params[refKey].value
    
    
    def plotMultipleFits(self, title=None):
        """
        Plot multiple datasets with the corresponding fits.    
        """
        
        if not hasattr(self, 'bestparams'):
            self.bestFitParams()
        
        plt.figure()
        for spec_idx in self.plot_range: # wrong index is displayed when plot_range has a step greater than 1
            idx = list(self.data_range).index(spec_idx) # this might work better; needs testing as of 2020-04-26
            bestfit = pVIC(self.X[idx], *self.bestparams[idx])
            p = plt.errorbar(self.X[idx], self.Y[idx], self.dY[idx], marker='o',
                             linewidth=0, label=f"expt {self.data['H (T)'][spec_idx]:.3g}T")
            plt.plot(self.X[idx], bestfit, '-', color=p[-1][0].get_color()[0,:3], 
                     label=f"fit {self.data['H (T)'][spec_idx]:.3g}T")
            plt.legend(loc='best')
        plt.show()
        
        freeParams = [k for k in list(self.init_params.keys()) if self.init_params[k].vary==True]
        if title is None:
            plt.title(f"TmVO$_4$ neutrons {len(freeParams)} free parameters")
        else:
            plt.title(title)




    # The following function is deprecated as of 2020-04-24; could be deleted

    # def xFitInitParams(self, fitParams, refParams, resultParams=None):
    #     """
    #     Initialize parameters for the next fitting iteration using the results of the previous fit
    #     and, if necessary, the default values of a reference set of parameters    
    
    #     Parameters
    #     ----------
    #     fitParams : lmfit Parameters object
    #         Fit parameters to initialize.
    #     refParams : lmfit Parameters object
    #         Reference Parameters object, containing default values and a fixed 
    #         number of Parameter objects, as defined from the model fit function.
    #     resultParams : lmfit Parameters object, optional
    #         Parameters object yielded by the previously performed fit, if any.
    #         The default is None.
    
    #     Returns
    #     -------
    #     fitParams : TYPE
    #         DESCRIPTION.
    
    #     """
    #     # For those parameters that have been computed in the last run, 
    #     # use as initial values for the next run the best fit values obtained in the last
    #     if resultParams is not None:
    #         for key in fitParams.keys():
    #             try:
    #                 fitParams[key].value = resultParams[key].value
    #             except KeyError: # in case fitParams has been modified since last fitting run
    #                 continue
    
    #     # Create additional fit parameters, e.g. if the number of datasets has been extended
    #     for spec_idx in self.data_range:
    #     # loop over indices of datasets, in order to create fit parameters for each of them
    #         for k, _ in refParams.items():
    #             if k in ['A', 'xp']:
    #             # fit parameters that are different for each dataset are assigned individual names
    #                 par_key = f'{k}{spec_idx}'
    #                 if par_key not in fitParams.keys():
    #                     fitParams.add( par_key, value=refParams[k].value, 
    #                                    min=refParams[k].min, vary=refParams[k].vary )
    #             elif resultParams is None: # if there are no shared fit parameters, because no fit has been performed yet
    #             # all other parameters are shared by all datasets and are assigned the "generic" name from refParams
    #                 fitParams[k] = cp.copy(refParams[k])
    #     return fitParams
