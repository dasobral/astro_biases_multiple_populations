#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Jul 2023

@author: dasobral

"""

#Import
import numpy as np
from scipy.optimize import fsolve
from scipy.interpolate import interp1d
from cosmofuncs import CosmoFuncs
import pandas as pd

class GalaxyBias(object):
    """
    Represents a galaxy bias model.
    Args:
        b1 (float): The value of b1 parameter. Default is 0.554.
        b2 (float): The value of b2 parameter. Default is 0.783.
        to_list (bool): Whether to return the result as a list. Default is False.
    Methods:
        gbias_bright(x):
            Calculates the galaxy bias for bright galaxies.
            Args:
                x (array-like): Input values.
            Returns:
                array-like: The calculated galaxy bias for bright galaxies.
        gbias_faint(x):
            Calculates the galaxy bias for faint galaxies.
            Args:
                x (array-like): Input values.
            Returns:
                array-like: The calculated galaxy bias for faint galaxies.
        gbias_total(x):
            Calculates the galaxy bias of the full population.
            Args:
                x (array-like): Input values.
            Returns:
                array-like: The calculated total galaxy bias.
    """
    
    def __init__(self, b1 = 0.554, b2  = 0.783, to_list = False):
        self.b1 = b1
        self.b2 = b2
        self.to_list = to_list

    def gbias_bright(self, x):

        if self.to_list == True:
            return (self.b1 * np.exp(self.b2 * np.array(x)) + 0.5).tolist()
        else:
            return (self.b1 * np.exp(self.b2 * np.array(x)) + 0.5)

    def gbias_faint(self, x):

        if self.to_list == True:
            return (self.b1 * np.exp(self.b2 * np.array(x)) - 0.5).tolist()
        else:
            return (self.b1 * np.exp(self.b2 * np.array(x)) - 0.5)
    
    def gbias_total(self, x):
        if list == True:
            return (self.b1 * np.exp(self.b2 * np.array(x))).tolist()
        else:
            return (self.b1 * np.exp(self.b2 * np.array(x)))
        
class GalaxyBiasMultiSplit(object):
    """
    Represents a multi-split galaxy bias model.
    Args:
        b1 (float): The value of b1 parameter. Default is 0.554.
        b2 (float): The value of b2 parameter. Default is 0.783.
        n_split (float): The number of splits. Default is 2.0.
        delta (float): The value of delta parameter. Default is 1.0.
        to_list (bool): Whether to return the result as a list. Default is False.
    Methods:
        gbias_bright(x):
            Calculates the galaxy bias for bright galaxies.
            Args:
                x (array-like): Input values.
            Returns:
                array-like: The calculated galaxy bias for bright galaxies.
        gbias_faint(x):
            Calculates the galaxy bias for faint galaxies.
            Args:
                x (array-like): Input values.
            Returns:
                array-like: The calculated galaxy bias for faint galaxies.
        gbias_total(x):
            Calculates the galaxy bias of the full population.
            Args:
                x (array-like): Input values.
            Returns:
                array-like: The calculated total galaxy bias.
    """
    
    def __init__(self, b1=0.554, b2=0.783, n_split=2.0, delta=1.0, to_list=False):
        self.b1 = b1
        self.b2 = b2
        self.to_list = to_list
        self.deltab = delta
        self.n_split = n_split
        
    def gbias_bright(self, x):
        """
        Calculates the galaxy bias for bright galaxies.
        Args:
            x (array-like): Input values.
        Returns:
            array-like: The calculated galaxy bias for bright galaxies.
        """
        if self.to_list == True:
            return (self.b1 * np.exp(self.b2 * np.array(x)) + (self.n_split-1) * self.deltab / self.n_split).tolist()
        else:
            return (self.b1 * np.exp(self.b2 * np.array(x)) + (self.n_split-1) * self.deltab / self.n_split)

    def gbias_faint(self, x):
        """
        Calculates the galaxy bias for faint galaxies.
        Args:
            x (array-like): Input values.
        Returns:
            array-like: The calculated galaxy bias for faint galaxies.
        """
        if self.to_list == True:
            return (self.b1 * np.exp(self.b2 * np.array(x)) - self.deltab / self.n_split).tolist()
        else:
            return (self.b1 * np.exp(self.b2 * np.array(x)) - self.deltab / self.n_split)
    
    def gbias_total(self, x):
        """
        Calculates the galaxy bias of the full population.
        Args:
            x (array-like): Input values.
        Returns:
            array-like: The calculated total galaxy bias.
        """
        if self.to_list == True:
            return (self.b1 * np.exp(self.b2 * np.array(x))).tolist()
        else:
            return (self.b1 * np.exp(self.b2 * np.array(x)))

class FluxSolver(object):
    """
    A class that represents a FluxSolver object.
    
    Attributes:
        N_cut (float): The value of N_cut.
        n_split (int): The value of n_split.
        z_data (ndarray): An array of redshift values.
        sc_data (ndarray): An array of sc_data values.
    
    Methods:
        __init__(self, n_split, z_data, sc_data, N_cut):
            Initializes a FluxSolver object.
        fit_params(self):
            Fits the parameters using the data from 'fitting_params.csv'.
        Sc_fit(self, z):
            Calculates the Sc_fit value for a given redshift.
        LogN_g(self, z, Sc):
            Calculates the LogN_g value for a given redshift and Sc value.
        sc_equation(self, x, z, Sc):
            Calculates the sc_equation value for a given x, redshift, and Sc value.
        sc_solver(self, z, Sc):
            Solves the sc_equation using fsolve.
        LogN_z(self, z):
            Calculates the LogN_z value for a given redshift.
        LogNB_z(self, z):
            Calculates the LogNB_z value for a given redshift.
        LogNF_z(self, z):
            Calculates the LogNF_z value for a given redshift.
        LogFstar(self, z):
            Calculates the LogFstar value for a given redshift.
        LogFcut(self, z):
            Calculates the LogFcut value for a given redshift.
        dLogN(self, z):
            Calculates the derivative of LogN_z for a given redshift.
        dLogNB(self, z):
            Calculates the derivative of LogNB_z for a given redshift.
        dLogNF(self, z):
            Calculates the derivative of LogNF_z for a given redshift.
        dLogFstar(self, z):
            Calculates the derivative of LogFstar for a given redshift.
        dLogFcut(self, z):
            Calculates the derivative of LogFcut for a given redshift.
    """
    
    def __init__(self, n_split, 
                z_data = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]), 
                sc_data = np.array([6.24, 5.85, 5.54, 5.28, 5.08, 4.92, 4.79, 4.68, 4.61, 4.55, 4.51, 4.49, 4.48, 4.49, 4.5, 4.53, 4.56, 4.61, 4.66, 4.72]),
                N_cut=10.):
        """
        Initializes a FluxSolver object.
        
        Args:
            n_split (int): The number of splits.
            z_data (ndarray): An array of redshift values.
            sc_data (ndarray): An array of sc_data values.
            N_cut (float): The value of N_cut.
        """
        self.N_cut = N_cut 
        self.n_split = n_split
        self.z_data = z_data
        
        self.c1, self.c2, self.c3 = self.fit_params()
     
        self.Srms = interp1d(z_data, sc_data, kind='cubic', fill_value='extrapolate')
        self.Sc_data = self.Sc_fit(z_data)
        self.Sc_bright = self.sc_solver(z=self.z_data, Sc=self.Sc_data)
        
    def fit_params(self,):
        """
        Fits the parameters using the data from 'fitting_params.csv'.
        
        Returns:
            c1_int (interp1d): Interpolated function for c1.
            c2_int (interp1d): Interpolated function for c2.
            c3_int (interp1d): Interpolated function for c3.
        """
        fitting_params = pd.read_csv('Data/fitting_params.csv')

        Sc = fitting_params.iloc[:,0].values * self.N_cut/10. 

        c1 = fitting_params['c1'].values
        c2 = fitting_params['c2'].values
        c3 = fitting_params['c3'].values
        
        c1_int = interp1d(Sc, c1, kind='cubic', fill_value='extrapolate')
        c2_int = interp1d(Sc, c2, kind='cubic', fill_value='extrapolate')
        c3_int = interp1d(Sc, c3, kind='cubic', fill_value='extrapolate')
        
        return c1_int, c2_int, c3_int
    
    def Sc_fit(self, z):
        """
        Calculates the Sc_fit value for a given redshift.
        
        Args:
            z (array-like): Input redshift values.
        
        Returns:
            array-like: The calculated Sc_fit values.
        """
        return self.Srms(z) * self.N_cut/10.
    
    def LogN_g(self, z, Sc):
        """
        Calculates the LogN_g value for a given redshift and Sc value.
        
        Args:
            z (array-like): Input redshift values.
            Sc (array-like): Input Sc values.
        
        Returns:
            array-like: The calculated LogN_g values.
        """
        return np.log(10**self.c1(Sc) * z**self.c2(Sc) * np.exp(-self.c3(Sc)*z))
    
    def sc_equation(self, x, z, Sc):
        """
        Calculates the sc_equation value for a given x, redshift, and Sc value.
        
        Args:
            x (array-like): Input x values.
            z (array-like): Input redshift values.
            Sc (array-like): Input Sc values.
        
        Returns:
            array-like: The calculated sc_equation values.
        """
        return np.log(self.n_split) + self.LogN_g(z, x) - self.LogN_g(z, Sc)

    def sc_solver(self, z, Sc):
        """
        Solves the sc_equation using fsolve.
        
        Args:
            z (array-like): Input redshift values.
            Sc (array-like): Input Sc values.
        
        Returns:
            array-like: The solved sc_equation values.
        """
        guess = np.array([1.]*len(self.z_data))
        sol = fsolve(self.sc_equation, guess, args=(z, Sc))
        return sol
    
    def LogN_z(self, z):
        """
        Calculates the LogN_z value for a given redshift.
        
        Args:
            z (array-like): Input redshift values.
        
        Returns:
            array-like: The calculated LogN_z values.
        """
        return interp1d(self.z_data, self.LogN_g(self.z_data, self.Sc_data), kind='cubic', fill_value='extrapolate')(z)
    
    def LogNB_z(self, z):
        """
        Calculates the LogNB_z value for a given redshift.
        
        Args:
            z (array-like): Input redshift values.
        
        Returns:
            array-like: The calculated LogNB_z values.
        """
        return interp1d(self.z_data, self.LogN_g(self.z_data, self.Sc_bright), kind='cubic', fill_value='extrapolate')(z)
    
    def LogNF_z(self, z):
        """
        Calculates the LogNF_z value for a given redshift.
        
        Args:
            z (array-like): Input redshift values.
        
        Returns:
            array-like: The calculated LogNF_z values.
        """
        return  interp1d(self.z_data, np.log(np.exp(self.LogN_g(self.z_data, self.Sc_data)) - np.exp(self.LogN_g(self.z_data, self.Sc_bright))), kind='cubic', fill_value='extrapolate')(z)
    
    def LogFstar(self, z):
        """
        Calculates the LogFstar value for a given redshift.
        
        Args:
            z (array-like): Input redshift values.
        
        Returns:
            array-like: The calculated LogFstar values.
        """
        return interp1d(self.z_data, np.log(self.Sc_data), kind='cubic', fill_value='extrapolate')(z)
    
    def LogFcut(self, z):
        """
        Calculates the LogFcut value for a given redshift.
        
        Args:
            z (array-like): Input redshift values.
        
        Returns:
            array-like: The calculated LogFcut values.
        """
        return interp1d(self.z_data, np.log(self.Sc_bright), kind='cubic', fill_value='extrapolate')(z)
    
    def dLogN(self, z):
        """
        Calculates the derivative of LogN_z for a given redshift.
        
        Args:
            z (array-like): Input redshift values.
        
        Returns:
            array-like: The calculated derivative of LogN_z values.
        """
        return (1+z) * stencil_derivative(self.LogN_z, z)
    
    def dLogNB(self, z):
        """
        Calculates the derivative of LogNB_z for a given redshift.
        
        Args:
            z (array-like): Input redshift values.
        
        Returns:
            array-like: The calculated derivative of LogNB_z values.
        """
        return (1+z) * stencil_derivative(self.LogNB_z, z)
    
    def dLogNF(self, z):
        """
        Calculates the derivative of LogNF_z for a given redshift.
        
        Args:
            z (array-like): Input redshift values.
        
        Returns:
            array-like: The calculated derivative of LogNF_z values.
        """
        return (1+z) * stencil_derivative(self.LogNF_z, z)
    
    def dLogFstar(self, z):
        """
        Calculates the derivative of LogFstar for a given redshift.
        
        Args:
            z (array-like): Input redshift values.
        
        Returns:
            array-like: The calculated derivative of LogFstar values.
        """
        return (1+z) * stencil_derivative(self.LogFstar, z)
    
    def dLogFcut(self, z):
        """
        Calculates the derivative of LogFcut for a given redshift.
        
        Args:
            z (array-like): Input redshift values.
        
        Returns:
            array-like: The calculated derivative of LogFcut values.
        """
        return (1+z) * stencil_derivative(self.LogFcut, z)
        
    
class MagnificationBias(FluxSolver):
    """
    Represents a magnification bias model.
    Args:
        n_split (float): The number of splits. Default is 2.0.
        to_list (bool): Whether to return the result as a list. Default is False.
    Methods:
        Qmodel(z, Sc):
            Calculates the Qmodel value for a given redshift and Sc value.
            Args:
                z (array-like): Input redshift values.
                Sc (array-like): Input Sc values.
            Returns:
                array-like: The calculated Qmodel values.
        smodel_(z, Sc):
            Calculates the smodel_ value for a given redshift and Sc value.
            Args:
                z (array-like): Input redshift values.
                Sc (array-like): Input Sc values.
            Returns:
                array-like: The calculated smodel_ values.
        smodelB(z):
            Calculates the smodelB value for a given redshift.
            Args:
                z (array-like): Input redshift values.
            Returns:
                array-like: The calculated smodelB values.
        s_model(z):
            Calculates the s_model value for a given redshift.
            Args:
                z (array-like): Input redshift values.
            Returns:
                array-like: The calculated s_model values.
        s_bright(z):
            Calculates the s_bright value for a given redshift.
            Args:
                z (array-like): Input redshift values.
            Returns:
                array-like: The calculated s_bright values.
        s_faint(z):
            Calculates the s_faint value for a given redshift.
            Args:
                z (array-like): Input redshift values.
            Returns:
                array-like: The calculated s_faint values.
    """
    
    def __init__(self, n_split, to_list=False):
        super().__init__(n_split=n_split)
        self.to_list = to_list
        
    def Qmodel(self, z, Sc):
        """
        Calculates the Qmodel value for a given redshift and Sc value.
        Args:
            z (array-like): Input redshift values.
            Sc (array-like): Input Sc values.
        Returns:
            array-like: The calculated Qmodel values.
        """
        res = - Sc * ( stencil_derivative(self.c1, x=Sc)*np.log(10) + stencil_derivative(self.c2, x=Sc)*np.log(z) - stencil_derivative(self.c3, x=Sc)*z )
        return res
    
    def smodel_(self, z, Sc):
        """
        Calculates the smodel_ value for a given redshift and Sc value.
        Args:
            z (array-like): Input redshift values.
            Sc (array-like): Input Sc values.
        Returns:
            array-like: The calculated smodel_ values.
        """
        return 2/5 * self.Qmodel(z, Sc)
    
    def smodelB(self, z):
        """
        Calculates the smodelB value for a given redshift.
        Args:
            z (array-like): Input redshift values.
        Returns:
            array-like: The calculated smodelB values.
        """
        return interp1d(self.z_data, self.smodel_(self.z_data, self.Sc_bright), kind='cubic', fill_value='extrapolate', assume_sorted=False)(z)
    
    def s_model(self, z):
        """
        Calculates the s_model value for a given redshift.
        Args:
            z (array-like): Input redshift values.
        Returns:
            array-like: The calculated s_model values.
        """
        return interp1d(self.z_data, self.smodel_(self.z_data, self.Sc_data), kind='cubic', fill_value='extrapolate', assume_sorted=False)(z)
    
    def s_bright(self, z):
        """
        Calculates the s_bright value for a given redshift.
        Args:
            z (array-like): Input redshift values.
        Returns:
            array-like: The calculated s_bright values.
        """
        if self.to_list == True:
            return self.smodelB(z).tolist()
        else:
            return self.smodelB(z)

    def s_faint(self, z):
        """
        Calculates the s_faint value for a given redshift.
        Args:
            z (array-like): Input redshift values.
        Returns:
            array-like: The calculated s_faint values.
        """
        sfaint = (self.n_split/(self.n_split-1)) * self.s_model(z) - (1/(self.n_split - 1)) * self.smodelB(z)
        if self.to_list == True:
            return sfaint.tolist()
        else:
            return sfaint
        
class EvolutionBias(MagnificationBias):
    """
    A class representing the evolution bias for multiple populations.

    Args:
        n_split (int): The number of populations to split the sample into.
        CAMBsolver (optional): The CAMB solver object. Defaults to None.
        to_list (bool): Whether to convert the output to a list. Defaults to False.

    Attributes:
        to_list (bool): Whether the output is converted to a list.
        CAMBsolver: The CAMB solver object.
        cfuns: An instance of the CosmoFuncs class.

    Methods:
        dlogH(z):
            Calculates the derivative of the logarithm of the Hubble parameter with respect to redshift.
            Args:
                z (float or array-like): The redshift value(s).

            Returns:
                float or array-like: The calculated derivative value(s).

        rHterm(z):
            Calculates the ratio of the scale factor to the Hubble parameter.
            Args:
                z (float or array-like): The redshift value(s).

            Returns:
                float or array-like: The calculated ratio value(s).

        Q_model(z):
            Calculates the model bias for a given redshift.
            Args:
                z (float or array-like): The redshift value(s).

            Returns:
                float or array-like: The calculated model bias value(s).

        Q_bright(z):
            Calculates the bias for the bright population at a given redshift.
            Args:
                z (float or array-like): The redshift value(s).

            Returns:
                float or array-like: The calculated bias value(s).

        Q_faint(z):
            Calculates the bias for the faint population at a given redshift.
            Args:
                z (float or array-like): The redshift value(s).

            Returns:
                float or array-like: The calculated bias value(s).

        fevol_total(z):
            Calculates the total evolution bias for a given redshift.
            Args:
                z (float or array-like): The redshift value(s).

            Returns:
                float or array-like: The calculated total evolution bias value(s).

        fevol_bright(z):
            Calculates the evolution bias for the bright population at a given redshift.
            Args:
                z (float or array-like): The redshift value(s).

            Returns:
                float or array-like: The calculated evolution bias value(s).

        fevol_faint(z):
            Calculates the evolution bias for the faint population at a given redshift.
            Args:
                z (float or array-like): The redshift value(s).

            Returns:
                float or array-like: The calculated evolution bias value(s).
    """
    
    def __init__(self, n_split, CAMBsolver=None, to_list=False):
        super().__init__(n_split=n_split, to_list=to_list)
        self.to_list = to_list
        self.CAMBsolver = CAMBsolver
        self.cfuns = CosmoFuncs(CAMBsolver=self.CAMBsolver)
        
    def dlogH(self, z):
        """
        Calculates the derivative of the logarithm of the Hubble parameter with respect to redshift.

        Args:
            z (float or array-like): The redshift value(s).

        Returns:
            float or array-like: The calculated derivative value(s).
        """
        return (1+z) * self.cfuns.dlogH_f(z)
    
    def rHterm(self, z):
        """
        Calculates the ratio of the scale factor to the Hubble parameter.

        Args:
            z (float or array-like): The redshift value(s).

        Returns:
            float or array-like: The calculated ratio value(s).
        """
        return (1+z)/(self.cfuns.r(z) * self.cfuns.H(z))
    
    def Q_model(self, z):
        """
        Calculates the model bias for a given redshift.

        Args:
            z (float or array-like): The redshift value(s).

        Returns:
            float or array-like: The calculated model bias value(s).
        """
        return 2 * (1 + self.rHterm(z)) * 5/2 * self.s_model(z)
    
    def Q_bright(self, z):
        """
        Calculates the bias for the bright population at a given redshift.

        Args:
            z (float or array-like): The redshift value(s).

        Returns:
            float or array-like: The calculated bias value(s).
        """
        return 2 * (1 + self.rHterm(z)) * 5/2 * self.s_bright(z)

    def Q_faint(self, z):
        """
        Calculates the bias for the faint population at a given redshift.

        Args:
            z (float or array-like): The redshift value(s).

        Returns:
            float or array-like: The calculated bias value(s).
        """
        return 2 * (1 + self.rHterm(z)) * 5/2 * self.s_faint(z)
    
    def fevol_total(self, z):
        """
        Calculates the total evolution bias for a given redshift.

        Args:
            z (float or array-like): The redshift value(s).

        Returns:
            float or array-like: The calculated total evolution bias value(s).
        """
        return - self.dLogN(z) - self.dlogH(z) + (2 - 5*self.s_model(z))*self.rHterm(z) - 5*self.s_model(z) - 5/2*self.s_model(z)*self.dLogFstar(z)
    
    def fevol_bright(self, z): 
        """
        Calculates the evolution bias for the bright population at a given redshift.

        Args:
            z (float or array-like): The redshift value(s).

        Returns:
            float or array-like: The calculated evolution bias value(s).
        """
        return - self.dLogNB(z) - self.dlogH(z) + (2 - 5*self.s_bright(z))*self.rHterm(z) - 5*self.s_bright(z) - 5/2*self.s_bright(z)*self.dLogFcut(z)
    
    def fevol_faint(self, z):
        """
        Calculates the evolution bias for the faint population at a given redshift.

        Args:
            z (float or array-like): The redshift value(s).

        Returns:
            float or array-like: The calculated evolution bias value(s).
        """
        return - self.dLogNF(z) - self.dlogH(z) + (2 - 5*self.s_faint(z))*self.rHterm(z) - 5*self.s_faint(z) + 5/2*((1/(self.n_split-1)) * self.s_bright(z)*self.dLogFcut(z) - (self.n_split/(self.n_split-1)) * self.s_model(z)*self.dLogFstar(z))
    
# ------------------------------------------------------------------------------------------------------------------ #
        
def stencil_derivative(fun, x, h=0.0001):
    """
    Calculates the stencil derivative of a given function at a specific point.

    Parameters:
    - fun: The function to calculate the derivative of.
    - x: The point at which to calculate the derivative.
    - h: The step size for the derivative calculation. Default is 0.0001.

    Returns:
    - d: The calculated derivative value.
    """
    if type(x) == list:
        x=np.array(x)    
    d = (-fun(x+2*h) + 8*fun(x+h) - 8*fun(x-h) + fun(x-2*h))/(12*h)
    return d        
    
