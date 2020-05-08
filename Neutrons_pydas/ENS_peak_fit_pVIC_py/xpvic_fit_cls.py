# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 16:09:08 2020

@author: Pierre Massat <pmassat@stanford.edu>

Define batch fitting functions used in 'ENS_peak_fit_pVIC_2019-02-14.ipynb'

"""

import copy as cp, numpy as np, warnings
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from lmfit import Parameters, minimize
from ENS_peak_fit_pVIC_py.pseudoVoigtIkedaCarpenter import pVIC, xnpVIC_residual


np.seterr(divide='warn')


class xpvic_fit:
    """
    Class of objects allowing to easily batch fit neutrons diffraction data with
    one or more pseudo-Voigt Ikeda-Carpenter functional forms.
    """
    
    def __init__(self, data, refParams, h, fit_interval=[-.1, .1], 
                 data_range=None, num_fit_func=1):
        """
        Initialize attributes of the class
        
        Parameters
        ----------
        data : Pandas dataframe
            Dataframe where each row of the column called "spectra" contains a 
            dataframe with ENS data, where hh0 is the x-axis data and Inorm is 
            the y-axis data. See the xyData() method.
        refParams : lmfit Parameters object
            Reference Parameters object, containing default values and a fixed 
            number of Parameter objects, as defined from the model fit function.
        h: positive integer
            peak position in (h h 0) reciprocal space
        data_range : Range
            Range of indices of nData datasets to include in fit.
            
        Returns
        -------
        self.xdata_selection : Boolean NumPy array, optional
            Array with same size as nData.spectra[row_index].hh0, with ones where 
            data should be used for fitting, and zeros elsewhere.
        self.data_range : Range
            Range of indices of nData datasets to include in fit.

        """

        # Metadata of the fit: peak position in (h h 0) reciprocal space
        self.h = h
        peak_position = - float(h)
        self.hkl = f"({self.h} {self.h} 0)"
        
        # Number of fitting function for each curve
        self.num_fit_func = num_fit_func

        # Create range of data to fit and plot
        if data_range is None:
            self.data_range = range(len(data)) # by default, use the full range of data
        else:
            self.data_range = data_range
        self.plot_range = self.data_range # use the same range for plotting

        # Number of spectra to use for the fit
        self.num_spec = len(self.data_range)

        # x-axis data selection
        dat_idx = data_range[-1]
        self.xdata_selection = np.logical_and(
            data.spectra[dat_idx].hh0 > peak_position + fit_interval[0], 
            data.spectra[dat_idx].hh0 < peak_position + fit_interval[1]
            )
        self.plot_lim = .15
        self.xdata_plot_selection = np.logical_or(
            np.logical_and(
            data.spectra[dat_idx].hh0 > peak_position - self.plot_lim, 
            data.spectra[dat_idx].hh0 < peak_position + fit_interval[0]
            ),
            np.logical_and(
            data.spectra[dat_idx].hh0 > peak_position + fit_interval[1], 
            data.spectra[dat_idx].hh0 < peak_position + self.plot_lim
            )
            )
        
        # Data to fit
        self.data = data
        
        # Set of reference parameters to use for the fit
        self.refParams = refParams
        
        # Number of shared free parameters in the fit
        self.freeSharedPrms = 0
        for key in refParams.keys():
            if refParams[key].vary is True:
                self.freeSharedPrms += 1

        # Create x- and y-axis arrays of data for batch fitting.
        self.makeData()
        

    def makeData(self): #
        """
        Create x- and y-axis arrays of data for batch fitting.
            
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
        
        # These lines can be deleted as of 2020-04-30 
        # # If no data selection filter is applied, select all the data
        # if data_select is None:
        #     data_select = np.ones(self.data.spectra[self.data_range[0]].hh0.shape, dtype=bool)
        
        # Create x, y and dy data arrays
        self.X = np.stack([self.data.spectra[idx].hh0[self.xdata_selection] 
                           for idx in self.data_range])
        self.Xplot = np.stack([self.data.spectra[idx].hh0[self.xdata_plot_selection] 
                           for idx in self.data_range])
        self.Y = np.stack([self.data.spectra[idx].Inorm[self.xdata_selection] 
                           for idx in self.data_range])
        self.Yplot = np.stack([self.data.spectra[idx].Inorm[self.xdata_plot_selection] 
                           for idx in self.data_range])
        self.dY = np.stack([self.data.spectra[idx].dInorm[self.xdata_selection] 
                            for idx in self.data_range])

        # Compute weights from data errors
        self.weights = 1/(self.dY)
        # Set all np.inf values in self.weights to zero
        if np.any(self.weights==np.inf):
            self.weights[self.weights==np.inf] = 0
            # print(np.argwhere(self.weights==np.inf))
            warnings.warn(f"Infinite values were encountered in 'weights', at positions \
                          {np.argwhere(self.weights==np.inf)}. They were reset to zero.")

        for idx in self.data_range:
            if not np.any(self.dY):
                warnings.warn(f"All errors are zero in spectrum with index {idx}; \
                              using all ones as weights.")
                self.weights[idx] = np.ones(self.weights[idx].shape)
        
    
    def initParams(self, resultParams=None, xp=None, A=None):
        """
        Initialize parameters for the next fitting iteration using the results of the previous fit
        and, if necessary, the default values of a reference set of parameters    
    
        Parameters
        ----------
        xp: NumPy array of length self.num_fit_func
            Initial values of the independent parameter xp, which contains the peak positions
        A: NumPy array of length self.num_fit_func
            Initial values of the independent parameter A, which contains the peak amplitudes
        resultParams : lmfit Parameters object, optional
            Parameters object yielded by the previously performed fit, if any.
            The default is None.
    
        Returns
        -------
        self.init_params : lmfit Parameters object
            Parameters to be used in the fit of all curves.
        """

        # Set default values of xp and A arrays when fitting with only one pVIC function: 
        # Default value of xp is np.array([self.refParams['xp'].value])
        if xp is None:
            xp = np.array([self.refParams['xp'].value+0.015*(self.num_fit_func-1-2*idx) 
                           for idx in range(self.num_fit_func)])
            
        # Default value of A is np.array([self.refParams['A'].value])
        if A is None:
            A = np.array([self.refParams['A'].value/np.sqrt(self.num_fit_func) 
                          for _ in range(self.num_fit_func)])

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
            for key in self.refParams.keys():
                if key in ['A', 'xp']:
                # fit parameters that are different for each dataset are assigned individual names
                    for fidx in range(self.num_fit_func):                      
                        par_key = f"{key}{spec_idx}_{fidx}"
                        if par_key not in self.init_params.keys():
                            try:
                                self.init_params.add(par_key, 
                                                     value=eval(key)[fidx], 
                                                     min=self.refParams[key].min, 
                                                     vary=self.refParams[key].vary
                                                     )
                            except:
                                raise TypeError(f"{key} must be an iterable \
                                                object of length {self.num_fit_func}")

                # For the shared fit parameters, if they have not been previously computed
                elif resultParams is None: 
                # They are assigned the "generic" name from self.refParams
                    self.init_params[key] = cp.copy(self.refParams[key])


    def performFit(self, with_weights=True):
        """
        Perfor fit using the xpVIC_residual function, which is the residual 
        function for fitting multiple curves using the pseudo-Voigt-Ikeda-Carpenter
        functional form.

        Returns
        -------
        self.result: Minimizer.Result object from the lmfit module
            Result of the minimize function, containing the fit results.

        """
        if with_weights is True:
            self.result = minimize(xnpVIC_residual, self.init_params, 
                                   args=(self.X, self.Y, self.data_range),
                                   kws={'weights':self.weights, 
                                        'nFunc':self.num_fit_func})
        else:
            self.result = minimize(xnpVIC_residual, self.init_params, 
                                   args=(self.X, self.Y, self.data_range),
                                   kws={'nFunc':self.num_fit_func})

    
    def bestFitParams(self):
        """
        Create NumPy array of best fit parameter values for each fitted curve.
    
        Returns
        -------
        self.bestparams : NumPy array
            Array of arrays of best fit parameter values for each fitted curve.
    
        """

        self.bestparams = np.zeros((self.num_spec,self.num_fit_func,len(self.refParams)))
        # self.bestparams.shape = # of datasets x # of parameters in fit function (pVIC)
        for spec_idx in range(self.num_spec):
            for fidx in range(self.num_fit_func):
                for par_idx, refKey in enumerate(self.refParams.keys()):
                    par_key = f"{refKey}{self.data_range[spec_idx]}_{fidx}"
                    # parameter name is a concatenation of the generic parameter name,
                    # as defined in the self.refParams function, and the spectrum index
                    try:
                        self.bestparams[spec_idx][fidx][par_idx] = self.result.params[par_key].value
                    except KeyError:
                        self.bestparams[spec_idx][fidx][par_idx] = self.result.params[refKey].value
    
    
    def plotMultipleFits(self, title=None, plotSubFits=False):
        """
        Plot multiple datasets with the corresponding fits.    
        """
        
        # (Re)compute best fit parameters if they don't exist or if something 
        # went wrong during the first computation and all computed parameters are zero
        if not hasattr(self, 'bestparams') or np.any(self.bestparams)==0:
            self.bestFitParams()
        
        fig, ax = plt.subplots()
        for spec_idx in self.plot_range: #
            dat_idx = list(self.data_range).index(spec_idx) #
            # print(f"data_range index = {self.data_range[dat_idx]}; \
            #       plot_range index = {spec_idx}") # Just to check that the right labels are displayed in the legend
            fieldlabel = f"{self.data['H (T)'][spec_idx]:.3g}T"
            p = plt.errorbar(self.X[dat_idx], self.Y[dat_idx], self.dY[dat_idx], 
                             marker='o', elinewidth=1, linewidth=0, label=f"expt {fieldlabel}")
            pcolor = p[-1][0].get_color()[0,:3]
            plt.plot(self.Xplot[dat_idx], self.Yplot[dat_idx], marker='x',
                     linewidth=0, color=pcolor, label=f"excluded")
            
            plot_center = -float(self.h)
            plot_lim = self.plot_lim
            fit_xrange = np.linspace(plot_center-plot_lim, plot_center+plot_lim, num=400)
            best_subfit = np.zeros((self.num_fit_func, fit_xrange.shape[0]))

            for fidx in range(self.num_fit_func):
                best_subfit[fidx] = pVIC(fit_xrange, *self.bestparams[dat_idx][fidx])
                if plotSubFits is True:
                    plt.plot(fit_xrange, best_subfit[fidx], '-', label=f"subfit {fieldlabel} {fidx}")
                    
            bestfit = np.sum(best_subfit, axis=0)
            plt.plot(fit_xrange, bestfit, '-', color=pcolor, label=f"fit {fieldlabel}")
            plt.legend(loc='best')
        plt.show()
        
        freeParams = [k for k in list(self.init_params.keys()) if self.init_params[k].vary==True]
        if title is None:
            plt.title(f"TmVO$_4$ neutrons {len(freeParams)} free parameters")
        else:
            plt.title(title)
        plt.xlabel("$h$ in ($h$ $h$ 0)")
        plt.ylabel("$I$ (a.u.)")
        
        # Set the format of y-axis tick labels
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2g'))




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
