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
    
    def __init__(self, b1 = 0.554, b2  = 0.783, n_split = 2., delta = 1.0, to_list = False):
        self.b1 = b1
        self.b2 = b2
        self.to_list = to_list
        
        self.deltab = delta
        self.n_split = n_split
        
    def gbias_bright(self, x):

        if self.to_list == True:
            return (self.b1 * np.exp(self.b2 * np.array(x)) + (self.n_split-1) * self.deltab / self.n_split).tolist()
        else:
            return (self.b1 * np.exp(self.b2 * np.array(x)) + (self.n_split-1) * self.deltab / self.n_split)

    def gbias_faint(self, x):

        if self.to_list == True:
            return (self.b1 * np.exp(self.b2 * np.array(x)) - self.deltab / self.n_split).tolist()
        else:
            return (self.b1 * np.exp(self.b2 * np.array(x)) - self.deltab / self.n_split)
    
    def gbias_total(self, x):
        if self.to_list == True:
            return ( self.b1 * np.exp(self.b2 * np.array(x)) ).tolist()
        else:
            return ( self.b1 * np.exp(self.b2 * np.array(x)) )

class FluxSolver(object):
    
    def __init__(self, n_split, 
                z_data = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]), 
                sc_data = np.array([6.24, 5.85, 5.54, 5.28, 5.08, 4.92, 4.79, 4.68, 4.61, 4.55, 4.51, 4.49, 4.48, 4.49, 4.5, 4.53, 4.56, 4.61, 4.66, 4.72]),
                N_cut=10.):
        
        self.N_cut = N_cut 
        self.n_split = n_split
        self.z_data = z_data
        
        self.c1, self.c2, self.c3 = self.fit_params()
     
        self.Srms = interp1d(z_data, sc_data, kind='cubic', fill_value='extrapolate')
        self.Sc_data = self.Sc_fit(z_data)
        self.Sc_bright = self.sc_solver(z=self.z_data, Sc=self.Sc_data)
        
    def fit_params(self,):
        
        fitting_params = pd.read_csv('fitting_params.csv')

        Sc = fitting_params.iloc[:,0].values * self.N_cut/10. 

        c1 = fitting_params['c1'].values
        c2 = fitting_params['c2'].values
        c3 = fitting_params['c3'].values
        
        c1_int = interp1d(Sc, c1, kind='cubic', fill_value='extrapolate')
        c2_int = interp1d(Sc, c2, kind='cubic', fill_value='extrapolate')
        c3_int = interp1d(Sc, c3, kind='cubic', fill_value='extrapolate')
        
        return c1_int, c2_int, c3_int
    
    def Sc_fit(self, z):
        return self.Srms(z) * self.N_cut/10.
    
    def LogN_g(self, z, Sc):
        return np.log(10**self.c1(Sc) * z**self.c2(Sc) * np.exp(-self.c3(Sc)*z))
    
    def sc_equation(self, x, z, Sc):
        return np.log(self.n_split) + self.LogN_g(z, x) - self.LogN_g(z, Sc)

    def sc_solver(self, z, Sc):
        guess = np.array([1.]*len(self.z_data))
        sol = fsolve(self.sc_equation, guess, args=(z, Sc))
        return sol
    
    def LogN_z(self, z):
        return interp1d(self.z_data, self.LogN_g(self.z_data, self.Sc_data), kind='cubic', fill_value='extrapolate')(z)
    
    def LogNB_z(self, z):
        return interp1d(self.z_data, self.LogN_g(self.z_data, self.Sc_bright), kind='cubic', fill_value='extrapolate')(z)
    
    def LogNF_z(self, z):
        return  interp1d(self.z_data, np.log(np.exp(self.LogN_g(self.z_data, self.Sc_data)) - np.exp(self.LogN_g(self.z_data, self.Sc_bright))), kind='cubic', fill_value='extrapolate')(z)
    
    def LogFstar(self, z):
        return interp1d(self.z_data, np.log(self.Sc_data), kind='cubic', fill_value='extrapolate')(z)
    
    def LogFcut(self, z):
        return interp1d(self.z_data, np.log(self.Sc_bright), kind='cubic', fill_value='extrapolate')(z)
    
    def dLogN(self, z):
        return (1+z) * stencil_derivative(self.LogN_z, z)
    
    def dLogNB(self, z):
        return (1+z) * stencil_derivative(self.LogNB_z, z)
    
    def dLogNF(self, z):
        return (1+z) * stencil_derivative(self.LogNF_z, z)
    
    def dLogFstar(self, z):
        return (1+z) * stencil_derivative(self.LogFstar, z)
    
    def dLogFcut(self, z):
        return (1+z) * stencil_derivative(self.LogFcut, z)
        
    
class MagnificationBias(FluxSolver):
    
    def __init__(self, n_split, to_list=False):
        super().__init__(n_split=n_split)
        self.to_list = to_list
        
    def Qmodel(self, z, Sc):
        res = - Sc * ( stencil_derivative(self.c1, x=Sc)*np.log(10) + stencil_derivative(self.c2, x=Sc)*np.log(z) - stencil_derivative(self.c3, x=Sc)*z )
        return res
    
    def smodel_(self, z, Sc):
        return 2/5 * self.Qmodel(z, Sc)
    
    def smodelB(self, z):
        return interp1d(self.z_data, self.smodel_(self.z_data, self.Sc_bright), kind='cubic', fill_value='extrapolate', assume_sorted=False)(z)
    
    def s_model(self, z):
        return interp1d(self.z_data, self.smodel_(self.z_data, self.Sc_data), kind='cubic', fill_value='extrapolate', assume_sorted=False)(z)
    
    def s_bright(self, z):
        if self.to_list == True:
            return self.smodelB(z).tolist()
        else:
            return self.smodelB(z)

    def s_faint(self, z):
        sfaint = (self.n_split/(self.n_split-1)) * self.s_model(z) - (1/(self.n_split - 1)) * self.smodelB(z)
        if self.to_list == True:
            return sfaint.tolist()
        else:
            return sfaint
        
class EvolutionBias(MagnificationBias):
    
    def __init__(self, n_split, CAMBsolver = None, to_list=False):
        
        super().__init__(n_split=n_split, to_list=to_list)
        self.to_list = to_list
        self.CAMBsolver = CAMBsolver
        
        self.cfuns = CosmoFuncs(CAMBsolver=self.CAMBsolver)
        
    def dlogH(self, z):
        return (1+z) * self.cfuns.dlogH_f(z)
    
    def rHterm(self, z):
        return (1+z)/(self.cfuns.r(z) * self.cfuns.H(z))
    
    def Q_model(self, z):
        return 2 * (1 + self.rHterm(z)) * 5/2 * self.s_model(z)
    
    def Q_bright(self, z):
        return 2 * (1 + self.rHterm(z)) * 5/2 * self.s_bright(z)

    def Q_faint(self, z):
        return 2 * (1 + self.rHterm(z)) * 5/2 * self.s_faint(z)
    
    def fevol_total(self, z):
        return - self.dLogN(z) - self.dlogH(z) + (2 - 5*self.s_model(z))*self.rHterm(z) - 5*self.s_model(z) - 5/2*self.s_model(z)*self.dLogFstar(z)
    
    def fevol_bright(self, z): 
        return - self.dLogNB(z) - self.dlogH(z) + (2 - 5*self.s_bright(z))*self.rHterm(z) - 5*self.s_bright(z) - 5/2*self.s_bright(z)*self.dLogFcut(z)
    
    def fevol_faint(self, z):
        return - self.dLogNF(z) - self.dlogH(z) + (2 - 5*self.s_faint(z))*self.rHterm(z) - 5*self.s_faint(z) + 5/2*((1/(self.n_split-1)) * self.s_bright(z)*self.dLogFcut(z) - (self.n_split/(self.n_split-1)) * self.s_model(z)*self.dLogFstar(z))     
    
# ------------------------------------------------------------------------------------------------------------------ #
        
def stencil_derivative(fun, x, h=0.0001):
    if type(x) == list:
        x=np.array(x)    
    d = (-fun(x+2*h) + 8*fun(x+h) - 8*fun(x-h) + fun(x-2*h))/(12*h)
    return d        
    
