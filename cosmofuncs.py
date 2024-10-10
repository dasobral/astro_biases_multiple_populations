#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Oct 2022

@author: dasobral

"""

# Import

import numpy as np
import scipy.integrate as integrate
import scipy as sc
from scipy.interpolate import interp1d

################################################################################################

class CosmoFuncs(object):
    """
    A class that provides cosmological functions and calculations.

    Parameters:
    -----------
    CAMBsolver : object, optional
        An instance of the CAMBsolver class. If provided, the cosmological parameters will be extracted from the CAMBsolver object.
    Om : float, optional
        The matter density parameter. Default is 0.3111.
    h : float, optional
        The dimensionless Hubble constant. Default is 0.6766.
    H0 : float, optional
        The Hubble constant in units of km/s/Mpc. Default is 1/2997.9.
    c : float, optional
        The speed of light in km/s. Default is 299792.458.

    Attributes:
    -----------
    solver : object
        An instance of the CAMBsolver class.
    c : float
        The speed of light in km/s.
    h : float
        The dimensionless Hubble constant.
    H0 : float
        The Hubble constant in units of km/s/Mpc.
    Om : float
        The matter density parameter.
    Oml : float
        The dark energy density parameter.

    Methods:
    --------
    rH0_novec(z)
        Calculate the comoving distance at redshift z.
    HH(z)
        Calculate the conformal Hubble factor at redshift z.
    HH_dz(z)
        Calculate the derivative of the conformal Hubble factor with respect to redshift z.
    D1_novec(z)
        Calculate the growth factor at redshift z.
    D1_dz_novec(z)
        Calculate the derivative of the growth factor with respect to redshift z.
    f(z)
        Calculate the growth function at redshift z.
    f_dz(z)
        Calculate the derivative of the growth function with respect to redshift z.
    Omega_m(z)
        Calculate the matter density parameter as a function of redshift z.
    r_(z)
        Calculate the comoving distance at redshift z.
    H(z)
        Calculate the Hubble parameter at redshift z.
    dlogH_f(z)
        Calculate the derivative of the logarithm of the Hubble parameter with respect to redshift z.
    survey_vol(z, delta=0.05, factor=30000/41253)
        Calculate the survey volume at redshift z.
    Nbarz_f(z, Nbarzh3)
        Calculate the number density per redshift z.
    Nbar_pop(z, npop1, npop2)
        Calculate the number density for two populations at redshift z.
    """

    def __init__(self, CAMBsolver=None, Om=0.3111, h=0.6766, H0=1/2997.9, c=299792.458):
        """
        Initialize the CosmoFuncs object.

        Parameters:
        -----------
        CAMBsolver : object, optional
            An instance of the CAMBsolver class. If provided, the cosmological parameters will be extracted from the CAMBsolver object.
        Om : float, optional
            The matter density parameter. Default is 0.3111.
        h : float, optional
            The dimensionless Hubble constant. Default is 0.6766.
        H0 : float, optional
            The Hubble constant in units of km/s/Mpc. Default is 1/2997.9.
        c : float, optional
            The speed of light in km/s. Default is 299792.458.
        """

        self.solver = CAMBsolver

        if CAMBsolver is not None:
            self.c = c
            self.h = CAMBsolver.h
            self.H0 = CAMBsolver.H0/(c * CAMBsolver.h) # Dimensionless units 1/h
            self.Om = (CAMBsolver.ombh2 + CAMBsolver.omch2) / CAMBsolver.h**2
            self.Oml = 1 - self.Om
        else:
            self.c = c
            self.h = h
            self.H0 = H0 # Dimensionless units 1/h
            self.Om = Om
            self.Oml = 1 - self.Om

        self.params = {
            'c' : self.c,
            'h' : self.h,
            'H0' : self.H0,
            'Om' : self.Om,
            'Oml' : self.Oml
        }

        self.rH0 = np.vectorize(self.rH0_novec)
        self.D1 = np.vectorize(self.D1_novec)
        self.D1_dz = np.vectorize(self.D1_dz_novec)
        self.r = np.vectorize(self.r_)

    def rH0_novec(self, z):
        """
        Calculate the comoving distance at redshift z.

        Parameters:
        -----------
        z : float or array-like
            The redshift value(s) at which to calculate the comoving distance.

        Returns:
        --------
        value : float or array-like
            The comoving distance at redshift z.
        """

        if type(z) == list:
            z = np.array(z)
        
        result = integrate.quad(lambda x: 1/np.sqrt(self.Om * x**3 + self.Oml), 1, 1+z)
        value = np.array(result[0])

        return value
    
    def HH(self, z):
        """
        Calculate the conformal Hubble factor at redshift z.

        Parameters:
        -----------
        z : float or array-like
            The redshift value(s) at which to calculate the conformal Hubble factor.

        Returns:
        --------
        result : float or array-like
            The conformal Hubble factor at redshift z.
        """

        if type(z) == list:
            z = np.array(z)

        result =  np.sqrt(self.Om*(1+z) + self.Oml / (1+z)**2)
        return result

    def HH_dz(self, z):
        """
        Calculate the derivative of the conformal Hubble factor with respect to redshift z.

        Parameters:
        -----------
        z : float or array-like
            The redshift value(s) at which to calculate the derivative.

        Returns:
        --------
        result : float or array-like
            The derivative of the conformal Hubble factor with respect to redshift z.
        """

        if type(z) == list:
            z = np.array(z)

        result = - (1/2) * (self.Om*(1+z) - 2/(1+z)**2*(1-self.Om))

        return result

    def D1_novec(self, z):
        """
        Calculate the growth factor at redshift z.

        Parameters:
        -----------
        z : float or array-like
            The redshift value(s) at which to calculate the growth factor.

        Returns:
        --------
        value : float or array-like
            The growth factor at redshift z.
        """

        if type(z) == list:
            z = np.array(z)

        result =  integrate.quad(lambda x: 1/(self.Om/x + self.Oml*x**2)**(3/2), 0, 1/(1+z))
        value = 5*self.Om/2 * (self.Om*(1+z)**3 + self.Oml)**(1/2) * result[0]

        return value

    def D1_dz_novec(self, z):
        """
        Calculate the derivative of the growth factor with respect to redshift z.

        Parameters:
        -----------
        z : float or array-like
            The redshift value(s) at which to calculate the derivative.

        Returns:
        --------
        value : float or array-like
            The derivative of the growth factor with respect to redshift z.
        """

        if type(z) == list:
            z = np.array(z)
        
        result =  integrate.quad(lambda x: (1+x)/np.sqrt(self.Om*(1+x)**3 + self.Oml)**3, z, np.inf) 
        value = (5*self.Om/2) * ((3*self.Om*(1+z)**2/(2*np.sqrt(self.Om*(1+z)**3 + self.Oml))) * result[0] - (1+z)/(self.Om*(1+z)**3 + self.Oml))

        return value


    def f(self, z):
        """
        Calculate the growth function at redshift z.

        Parameters:
        -----------
        z : float or array-like
            The redshift value(s) at which to calculate the growth function.

        Returns:
        --------
        value : float or array-like
            The growth function at redshift z.
        """

        if type(z) == list:
            z = np.array(z)

        value = 1 / (self.Om + self.Oml*(1+z)**(-3)) * (-3*self.Om/2 + 5*self.Om/2/(1+z)/self.D1(z))

        return value

    def f_dz(self, z):
        """
        Calculate the derivative of the growth function with respect to redshift z.

        Parameters:
        -----------
        z : float or array-like
            The redshift value(s) at which to calculate the derivative.

        Returns:
        --------
        value : float or array-like
            The derivative of the growth function with respect to redshift z.
        """

        if type(z) == list:
            z = np.array(z)

        value = (3*self.Oml*(1+z)**2)/(self.Om*(1+z)**3 + self.Oml)**2 * (-3*self.Om/2 + 5*self.Om/2/((1+z)*self.D1(z))) - (5*self.Om/2)*((1+z)**3/(self.Om*(1+z)**3+self.Oml))*(self.D1(z) + (1+z)*self.D1_dz(z))/((1+z)*self.D1(z))**2
        
        return value

    def Omega_m(self, z):
        """
        Calculate the matter density parameter as a function of redshift z.

        Parameters:
        -----------
        z : float or array-like
            The redshift value(s) at which to calculate the matter density parameter.

        Returns:
        --------
        value : float or array-like
            The matter density parameter as a function of redshift z.
        """

        if type(z) == list:
            z = np.array(z)

        value = self.Om*(1+z)**3 / (self.Om*(1+z)**3 + self.Oml)

        return value
    
    def r_(self, z):
        """
        Calculate the comoving distance at redshift z.

        Parameters:
        -----------
        z : float or array-like
            The redshift value(s) at which to calculate the comoving distance.

        Returns:
        --------
        value : float or array-like
            The comoving distance at redshift z.
        """

        if type(z) == list:
            z = np.array(z)
        H0 = 100 * self.h / self.c # Mpc^{-1} units
        Oml = 1 - self.Om
        result = integrate.quad(lambda x: 1/(H0*np.sqrt(self.Om * (1+x)**3 + Oml)), 0, z)
        value = np.array(result[0])
        return value

    def H(self, z):
        """
        Calculate the Hubble parameter at redshift z.

        Parameters:
        -----------
        z : float or array-like
            The redshift value(s) at which to calculate the Hubble parameter.

        Returns:
        --------
        H : float or array-like
            The Hubble parameter at redshift z.
        """

        if type(z) == list:
            z = np.array(z)
        H0 = 100*self.h/self.c
        H = H0 * np.sqrt(self.Om*(1+np.array(z))**3 + (1-self.Om))
        return H
        
    def dlogH_f(self, z):
        """
        Calculate the derivative of the logarithm of the Hubble parameter with respect to redshift z.

        Parameters:
        -----------
        z : float or array-like
            The redshift value(s) at which to calculate the derivative.

        Returns:
        --------
        derivative : float or array-like
            The derivative of the logarithm of the Hubble parameter with respect to redshift z.
        """

        z = np.array(z)
        H0 = 100*self.h/self.c
        derivative = H0**2/2 * (3*self.Om*(1+z)**2) / (self.H(z)**2)
        return derivative

    def survey_vol(self, z, delta=0.05, factor=30000/41253):
        """
        Calculate the survey volume at redshift z.

        Parameters:
        -----------
        z : float or array-like
            The redshift value(s) at which to calculate the survey volume.
        delta : float, optional
            The redshift interval around z over which to calculate the survey volume. Default is 0.05.
        factor : float, optional
            A scaling factor for the survey volume calculation. Default is 30000/41253.

        Returns:
        --------
        vol : float or array-like
            The survey volume at redshift z.
        """

        if type(z) == list:
            z = np.array(z)

        vol = 4*np.pi/3 * (self.rH0(z+delta)**3 - self.rH0(z-delta)**3) / (self.H0**3) * factor

        return vol

    def Nbarz_f(self, z, Nbarzh3=np.array([6.2e-2, 3.63e-2, 2.16e-2, 1.31e-2, 8.07e-3, 5.11e-3, 3.27e-3, 2.11e-3, 1.36e-3, 8.7e-4, 5.56e-4, 3.53e-4, 2.22e-4, 1.39e-4, 8.55e-5, 5.2e-5, 3.12e-5, 1.83e-5, 1.05e-5])):
        """
        Calculate the number density per redshift z.

        Parameters:
        -----------
        z : float or array-like
            The redshift value(s) at which to calculate the number density.
        Nbarzh3 : array-like, optional
            The number density per redshift z in units of h^3/Mpc^3. Default is [6.2e-2, 3.63e-2, 2.16e-2, 1.31e-2, 8.07e-3, 5.11e-3, 3.27e-3, 2.11e-3, 1.36e-3, 8.7e-4, 5.56e-4, 3.53e-4, 2.22e-4, 1.39e-4, 8.55e-5, 5.2e-5, 3.12e-5, 1.83e-5, 1.05e-5].

        Returns:
        --------
        Nbarz_f : float or array-like
            The number density per redshift z.
        """

        if type(z) == list:
            z = np.array(z)

        Nbarz = Nbarzh3 / self.h**3
        zska = np.array([0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 1.05, 1.15, 1.25, 1.35, 1.45, 1.55, 1.65, 1.75, 1.85, 1.95])
        
        Nbarz_f = interp1d(zska, Nbarz, kind='cubic', fill_value= "extrapolate")
        
        return Nbarz_f(z)

    def Nbar_pop(self, z, npop1=0.5, npop2=0.5):
        """
        Calculate the number density for two populations at redshift z.

        Parameters:
        -----------
        z : float or array-like
            The redshift value(s) at which to calculate the number density.
        npop1 : float, optional
            The fraction of the first population. Default is 0.5.
        npop2 : float, optional
            The fraction of the second population. Default is 0.5.

        Returns:
        --------
        Nbar_pop1 : float or array-like
            The number density for the first population at redshift z.
        Nbar_pop2 : float or array-like
            The number density for the second population at redshift z.
        """

        if type(z) == list:
            z = np.array(z)

        Nbar_pop1 = npop1 * self.Nbarz_f(z)
        Nbar_pop2 = npop2 * self.Nbarz_f(z)

        return Nbar_pop1, Nbar_pop2
