#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Oct 2022

@author: dasobral

"""

# Import

import numpy as np
import itertools as it

from biasmodels import MagnificationBias, EvolutionBias, GalaxyBiasMultiSplit
from CAMBsolver import Solver
from cosmofuncs import CosmoFuncs

class Signal(object):
    """
    Class for the calculation of real-space multipole signals of multiple populations.
    
    Parameters:
    - solver: Solver object
        The solver object used to calculate cosmological quantities.
    - zin: float, optional
        The redshift at which the signal is calculated. Default is 10.
    - n_split: float, optional
        The number of population splits. Default is 2.
    - b1: float, optional
        The bias parameter for the first population. Default is 0.554.
    - b2: float, optional
        The bias parameter for the second population. Default is 0.783.
    - delta: float, optional
        The delta parameter. Default is 1.0.
    - which_multipoles: list of str, optional
        The list of multipoles to calculate. Default is ['monopole', 'dipole', 'quadrupole', 'hexadecapole'].
    - pop: list of str, optional
        The list of populations to consider. Default is ['b', 'f'].
    - wide_angle: bool, optional
        Flag indicating whether to consider wide-angle effects. Default is True.
    - evol_bias: bool, optional
        Flag indicating whether to consider evolution bias. Default is True.
    - tol: float, optional
        The tolerance for numerical calculations. Default is 1e-15.
    - return_lists: bool, optional
        Flag indicating whether to return the results as lists or numpy arrays. Default is True.
    """
        
    def __init__(self, solver, zin=10., n_split=2., b1=0.554, b2=0.783, delta=1.0, which_multipoles=['monopole', 'dipole', 'quadrupole', 'hexadecapole'], pop=['b', 'f'], wide_angle=True, evol_bias=True, tol=1e-15, return_lists=True):
        self.solver = solver
        self.multi_list = solver.interpolate_multi()
        self.tol = tol
        self.return_lists = return_lists
        
        self.n_split = n_split
        self.b1 = b1
        self.b2 = b2
        self.deltab = delta
        
        self.evol_bias_bool = evol_bias
        
        self.which_multipoles = which_multipoles
        self.pop = pop
        self.H0 = solver.H0 / 299792.485 / self.solver.h    
        
        self.mu0 = self.multi_list['mu0']
        self.mu2 = self.multi_list['mu2']
        self.mu4 = self.multi_list['mu4']
        self.nu1 = self.multi_list['nu1']
        self.nu3 = self.multi_list['nu3']

        self.GalaxyBias = GalaxyBiasMultiSplit(b1=self.b1, b2=self.b2, n_split=self.n_split, delta=self.deltab, to_list=self.return_lists)
        self.MagBias = MagnificationBias(n_split=self.n_split)
        self.EvolBias = EvolutionBias(n_split=self.n_split, CAMBsolver=self.solver)
        self.CosmoF = CosmoFuncs(self.solver)

        self.s8 = self.solver.Sigma8(zin)
        self.s80 = self.s8(0.0)
        self.D10 = self.CosmoF.D1(0.)

        if wide_angle:
            self._multipoles = {'monopole': self._monopole,
                                'dipole': self._dipole, 
                                'quadrupole': self._quadrupole,
                                'hexadecapole': self._hexadecapole, 
                                'octupole': self._octupole} # Specifies a list of functions (defined below)
        else:
            self._multipoles = {'monopole': self._monopole,
                                'dipole': self._dipole_nowide, 
                                'quadrupole': self._quadrupole,
                                'hexadecapole': self._hexadecapole, 
                                'octupole': self._octupole_nowide} 
        
    def calculate_signal(self, d, z):
        """
        Calculate the signal for the given separations and redshifts.
        
        Parameters:
        - d: float or array-like
            The separation(s) at which to calculate the signal.
        - z: float or array-like
            The redshift(s) at which to calculate the signal.
        
        Returns:
        - signal: array-like
            The calculated signal(s) for the given separations and redshifts.
        """
        for i, m in enumerate(self.which_multipoles): 
            if i == 0: 
                signal = self._multipoles[m](d, z)
            else:
                signal += self._multipoles[m](d, z)
        
        return signal
    
    def _monopole(self, d, z, which_comb='default'):
        """
        Calculate the monopole signal for the given separation and redshift.
        
        Parameters:
        - d: float or array-like
            The separation(s) at which to calculate the signal.
        - z: float or array-like
            The redshift(s) at which to calculate the signal.
        - which_comb: list of list of str, optional
            The combinations of populations to consider. Default is 'default'.
        
        Returns:
        - monopole: array-like
            The calculated monopole signal(s) for the given separation and redshift.
        """
        if which_comb == 'default':
            which_comb = list(it.combinations_with_replacement(self.pop, 2))
        
        mu0 = self.mu0(d)
        f = self.CosmoF.f(z) * self.s8(z)
        
        monopole = []
        for i, comb in enumerate(which_comb):
            bpop1 = self.galaxybias(comb[0], z)
            bpop2 = self.galaxybias(comb[1], z)
            zfactor = (bpop1 * bpop2 + 1/3 * (bpop1 + bpop2) * f + 1/5 * f**2) / self.s80**2
            monopole += [mu0 * np.transpose(np.array([zfactor,] * len(mu0)))]
        
        if not self.return_lists:
            monopole = np.array(monopole)
        
        return monopole
    
    def _dipole(self, d, z, which_comb='default'):
        """
        Calculate the dipole signal for the given separation and redshift.
        
        Parameters:
        - d: float or array-like
            The separation(s) at which to calculate the signal.
        - z: float or array-like
            The redshift(s) at which to calculate the signal.
        - which_comb: list of list of str, optional
            The combinations of populations to consider. Default is 'default'.
        
        Returns:
        - dipole: array-like
            The calculated dipole signal(s) for the given separation and redshift.
        """
        dipole_nowide = self._dipole_nowide(d, z, which_comb=which_comb)
        dipole_onlywide = self._dipole_onlywide(d, z, which_comb=which_comb)
        dipole_total = [x + y for x, y in zip(dipole_nowide, dipole_onlywide)]
        
        if not self.return_lists:
            dipole_total = np.array(dipole_total)
        
        return dipole_total
    
    def _dipole_nowide(self, d, z, which_comb='default', Euler=True):
        """
        Calculate the non-wide-angle dipole signal for the given separation and redshift.
        
        Parameters:
        - d: float or array-like
            The separation(s) at which to calculate the signal.
        - z: float or array-like
            The redshift(s) at which to calculate the signal.
        - which_comb: list of list of str, optional
            The combinations of populations to consider. Default is 'default'.
        - Euler: bool, optional
            Flag indicating whether the Euler equation is used or not. Default is True.
        
        Returns:
        - dipole_nowide: array-like
            The calculated non-wide-angle dipole signal(s) for the given separation and redshift.
        """
        if which_comb == 'default':
            which_comb = list(it.combinations(self.pop, 2))
        
        rH0 = self.CosmoF.rH0(z)
        Hz = self.CosmoF.HH(z)
        f = self.CosmoF.f(z) * self.s8(z)
        fdot = - (1 + np.array(z)) * self.H0 * Hz * (self.s80 / self.D10) * (self.CosmoF.D1(z) * self.CosmoF.f_dz(z) + self.CosmoF.D1_dz(z) * self.CosmoF.f(z))
        Ih = self.CosmoF.Omega_m(z) * self.s8(z)
        dHz = self.CosmoF.HH_dz(z)
        nu1 = d * self.H0 * self.nu1(d)
        
        dipole_nowide = []
        for i, comb in enumerate(which_comb):
            bpop1 = self.galaxybias(comb[0], z)
            bpop2 = self.galaxybias(comb[1], z)
            spop1 = self.magbias(comb[0], z)
            spop2 = self.magbias(comb[1], z)
            fpop1 = self.fevol(comb[0], z)
            fpop2 = self.fevol(comb[1], z)
            betapop1 = 5 * spop1 * (1 / (rH0 * Hz) - 1) + fpop1
            betapop2 = 5 * spop2 * (1 / (rH0 * Hz) - 1) + fpop2
            
            if Euler:
                dipole_nowideA = (3/5) * (betapop1 - betapop2) * f**2
                dipole_nowideB = - (bpop1 * betapop2 - bpop2 * betapop1) * f
                dipole_nowideC = (bpop1 - bpop2) * (2 / (rH0 * Hz) + dHz / Hz**2) * f
                zfactor = Hz * (dipole_nowideA + dipole_nowideB + dipole_nowideC) / self.s80**2
                dipole_nowide += [nu1 * np.transpose(np.array([zfactor,] * len(nu1)))]
            else:
                dipole_nowideA = (3/5) * (betapop1 - betapop2) * f**2
                dipole_nowideB = - (bpop1 * betapop2 - bpop2 * betapop1) * f - (bpop1 - bpop2) * (1 - 2 / (rH0 * Hz)) * f
                dipole_nowideC = (3/2) * (bpop1 - bpop2) * Ih
                dipole_nowideD = - (bpop1 - bpop2) * fdot / (self.H0 * Hz)
                zfactor = Hz * (dipole_nowideA + dipole_nowideB + dipole_nowideC + dipole_nowideD) / self.s80**2
                dipole_nowide += [nu1 * np.transpose(np.array([zfactor,] * len(nu1)))]
        
        if not self.return_lists:
            dipole_nowide = np.array(dipole_nowide)
        
        return dipole_nowide
    
    def _dipole_onlywide(self, d, z, which_comb='default'):
        """
        Calculate the wide-angle dipole signal for the given separation and redshift.
        
        Parameters:
        - d: float or array-like
            The separation(s) at which to calculate the signal.
        - z: float or array-like
            The redshift(s) at which to calculate the signal.
        - which_comb: list of list of str, optional
            The combinations of populations to consider. Default is 'default'.
        
        Returns:
        - dipole_onlywide: array-like
            The calculated wide-angle dipole signal(s) for the given separation and redshift.
        """
        if which_comb == 'default':
            which_comb = list(it.combinations(self.pop, 2))
        
        f = self.CosmoF.f(z) * self.s8(z)
        rH0 = self.CosmoF.rH0(z)
        mu2wide = d * self.mu2(d)
        
        dipole_onlywide = []
        for i, comb in enumerate(which_comb):
            bpop1 = self.galaxybias(comb[0], z)
            bpop2 = self.galaxybias(comb[1], z)
            zfactor = -2/5 * (bpop1 - bpop2) * f / (rH0 / self.H0) / self.s80**2
            dipole_onlywide += [mu2wide * np.transpose(np.array([zfactor,] * len(mu2wide)))]
        
        if not self.return_lists:
            dipole_onlywide = np.array(dipole_onlywide)
        
        return dipole_onlywide
    
    def _quadrupole(self, d, z, which_comb='default'):
        """
        Calculate the quadrupole signal for the given separation and redshift.
        
        Parameters:
        - d: float or array-like
            The separation(s) at which to calculate the signal.
        - z: float or array-like
            The redshift(s) at which to calculate the signal.
        - which_comb: list of list of str, optional
            The combinations of populations to consider. Default is 'default'.
        
        Returns:
        - quadrupole: array-like
            The calculated quadrupole signal(s) for the given separation and redshift.
        """
        if which_comb == 'default':
            which_comb = list(it.combinations_with_replacement(self.pop, 2))
        
        f = self.CosmoF.f(z) * self.s8(z)
        mu2 = self.mu2(d)
        
        quadrupole = []
        for i, comb in enumerate(which_comb):
            bpop1 = self.galaxybias(comb[0], z)
            bpop2 = self.galaxybias(comb[1], z)
            zfactor = -(2/3 * (bpop1 + bpop2) * f + 4/7 * f**2) / self.s80**2
            quadrupole += [mu2 * np.transpose(np.array([zfactor,] * len(mu2)))]
        
        if not self.return_lists:
            quadrupole = np.array(quadrupole)
        
        return quadrupole
    
    def _octupole(self, d, z, which_comb='default'):
        """
        Calculate the octupole signal for the given separation and redshift.
        
        Parameters:
        - d: float or array-like
            The separation(s) at which to calculate the signal.
        - z: float or array-like
            The redshift(s) at which to calculate the signal.
        - which_comb: list of list of str, optional
            The combinations of populations to consider. Default is 'default'.
        
        Returns:
        - octupole: array-like
            The calculated octupole signal(s) for the given separation and redshift.
        """
        oct_nowide = self._octupole_nowide(d, z, which_comb=which_comb)
        oct_onlywide = self._octupole_onlywide(d, z, which_comb=which_comb)
        oct_total = [x + y for x, y in zip(oct_nowide, oct_onlywide)]
        
        if not self.return_lists:
            oct_total = np.array(oct_total)
        
        return oct_total
    
    def _octupole_nowide(self, d, z, which_comb='default'):
        """
        Calculate the non-wide-angle octupole signal for the given separation and redshift.
        
        Parameters:
        - d: float or array-like
            The separation(s) at which to calculate the signal.
        - z: float or array-like
            The redshift(s) at which to calculate the signal.
        - which_comb: list of list of str, optional
            The combinations of populations to consider. Default is 'default'.
        
        Returns:
        - oct_nowide: array-like
            The calculated non-wide-angle octupole signal(s) for the given separation and redshift.
        """
        if which_comb == 'default':
            which_comb = list(it.combinations(self.pop, 2))
        
        rH0 = self.CosmoF.rH0(z)
        Hz = self.CosmoF.HH(z)
        f = self.CosmoF.f(z) * self.s8(z)
        nu3 = d * self.H0 * self.nu3(d)
        
        oct_nowide = []
        for i, comb in enumerate(which_comb):
            spop1 = self.magbias(comb[0], z)
            spop2 = self.magbias(comb[1], z)
            fpop1 = self.fevol(comb[0], z)
            fpop2 = self.fevol(comb[1], z)
            betapop1 = 5 * spop1 * (1 / (rH0 * Hz) - 1) + fpop1
            betapop2 = 5 * spop2 * (1 / (rH0 * Hz) - 1) + fpop2
            zfactor = 2/5 * Hz * (betapop1 - betapop2) * f**2 / self.s80**2
            oct_nowide += [nu3 * np.transpose(np.array([zfactor,] * len(nu3)))]
        
        if not self.return_lists:
            oct_nowide = np.array(oct_nowide)
        
        return oct_nowide
    
    def _octupole_onlywide(self, d, z, which_comb='default'):
        """
        Calculate the wide-angle octupole signal for the given separation and redshift.
        
        Parameters:
        - d: float or array-like
            The separation(s) at which to calculate the signal.
        - z: float or array-like
            The redshift(s) at which to calculate the signal.
        - which_comb: list of list of str, optional
            The combinations of populations to consider. Default is 'default'.
        
        Returns:
        - oct_onlywide: array-like
            The calculated wide-angle octupole signal(s) for the given separation and redshift.
        """
        if which_comb == 'default':
            which_comb = list(it.combinations(self.pop, 2))
            
    def galaxybias(self, pop, x): 
        """
        Compute the galaxy bias for a given population.

        This method evaluates the galaxy bias based on the specified population type
        (bright or faint) at the provided data point.

        Parameters:
        - pop: str
            The type of galaxy population for which to compute the bias. 
            Use 'b' for bright galaxies and any other value for faint galaxies.
        - x: float or array-like
            The input data point(s) at which to evaluate the galaxy bias.

        Returns:
        - float or array-like
            The computed galaxy bias for the specified population type at the given input.
        """

        if pop=='b':
                  return  self.GalaxyBias.gbias_bright(x)
        else:
                  return  self.GalaxyBias.gbias_faint(x)  
    
    
    def magbias(self, pop, x): 
        """
        Compute the magnification bias for a given population.

        This method evaluates the magnification bias based on the specified population type
        (bright or faint) at the provided redshift values.

        Parameters:
        - pop: str
            The type of galaxy population for which to compute the magnification bias. 
            Use 'b' for bright galaxies and any other value for faint galaxies.
        - x: float or array-like
            The input redshift value(s) at which to evaluate the magnification bias.

        Returns:
        - float or array-like
            The computed magnification bias for the specified population type at the given input redshift(s).
        """

        z = np.array(x)
        if pop=='b': 
            return self.MagBias.s_bright(z) 
        else:  
            return self.MagBias.s_faint(z)
        
    
    def fevol(self, pop, x):
        """
        Compute the evolution bias for a given population.

        This method calculates the evolution bias for either bright or faint galaxy populations
        at the provided redshift values.

        Parameters:
        - pop: str
            The type of galaxy population for which to compute the evolution bias.
            Use 'b' for bright galaxies and any other value for faint galaxies.
        - x: float or array-like
            The input redshift value(s) at which to evaluate the evolution bias.

        Returns:
        - float or array-like
            The computed evolution bias for the specified population type at the given input redshift(s).
        """

        if pop=='b':
            return self.EvolBias.fevol_bright(x)
        else:
            return self.EvolBias.fevol_faint(x)

        

class Derivatives(object):
    """
    A class to calculate derivatives related to cosmological parameters, galaxy bias, and magnification bias.

    Attributes:
    - params_dict0: dict
        A dictionary containing the names (as strings) of the parameters.
    - n_split: float
        The number of population splits.
    - b1: float
        The bias parameter for the first population.
    - b2: float
        The bias parameter for the second population.
    - delta: float
        The delta parameter.
    - wide_angle: bool
        Flag indicating whether to consider wide-angle effects.
    - dist_correction: bool
        Flag indicating whether to apply distance correction.
    - solverfid: Solver
        Solver instance initialized with z0 and params_dict0.
    - CosmoFfid: CosmoFuncs
        CosmoFuncs instance initialized with solverfid.
    - GalaxyBias: GalaxyBiasMultiSplit
        GalaxyBiasMultiSplit instance initialized with b1, b2, n_split, and delta.
    - MagBias: MagnificationBias
        MagnificationBias instance initialized with n_split.
    - EvolBias: EvolutionBias
        EvolutionBias instance initialized with n_split and CAMBsolver.
    - multi_listfid: dict
        Dictionary containing interpolated values from the solverfid.
    - mu0fid: float
        Interpolated mu0 value from multi_listfid.
    - mu2fid: float
        Interpolated mu2 value from multi_listfid.
    - mu4fid: float
        Interpolated mu4 value from multi_listfid.
    - nu1fid: float
        Interpolated nu1 value from multi_listfid.
    - nu3fid: float
        Interpolated nu3 value from multi_listfid.
    - H0: float
        Hubble constant in units of km/s/Mpc.
    """

    def __init__(self, params_dict0, z0=[0.0], n_split=2., b1=0.554, b2=0.783, delta=1.0, wide_angle=True, dist_correction=False):
       
        """
        Initialize the Derivatives class with specified parameters.
        
        Parameters:
        - params_dict0: dict
            A dictionary containing the names (as strings) of the parameters.
        - z0: list of float, optional
            The initial redshift(s). Default is [0.0].
        - n_split: float, optional
            The number of population splits. Default is 2.
        - b1: float, optional
            The bias parameter for the first population. Default is 0.554.
        - b2: float, optional
            The bias parameter for the second population. Default is 0.783.
        - delta: float, optional
            The delta parameter. Default is 1.0.
        - wide_angle: bool, optional
            Flag indicating whether to consider wide-angle effects. Default is True.
        - dist_correction: bool, optional
            Flag indicating whether to apply distance correction. Default is False.
        """

        self.params_dict0 = params_dict0 # This is a dict that contains names(str) of the parameters
        
        self.n_split = n_split
        self.b1 = b1
        self.b2 = b2
        self.deltab = delta
        
        self.solverfid = Solver(z0, **params_dict0)
        self.CosmoFfid = CosmoFuncs(self.solverfid)

        self.GalaxyBias = GalaxyBiasMultiSplit(b1=self.b1, b2=self.b2, n_split=self.n_split, delta=self.deltab)
        self.MagBias = MagnificationBias(n_split=self.n_split)
        self.EvolBias = EvolutionBias(n_split=self.n_split, CAMBsolver=self.solverfid)
        
        self.multi_listfid = self.solverfid.interpolate_multi()
        self.mu0fid = self.multi_listfid['mu0'];
        self.mu2fid = self.multi_listfid['mu2'];
        self.mu4fid = self.multi_listfid['mu4'];
        self.nu1fid = self.multi_listfid['nu1'];
        self.nu3fid = self.multi_listfid['nu3'];
        
        self.wide_angle = wide_angle
        self.dist_correction = dist_correction

        self.H0 = self.solverfid.H0 / 299792.485  / self.solverfid.h
        
    def five_pt_stencil(self, params, d, z, z0=[0.0], multipole=['monopole'], which_comb=[['b', 'b']], e=0.005, output_flat=False):
        
        """
        Compute the five-point stencil for the specified parameters to estimate derivatives. This is the method used
        to compute the derivatives with respect to the cosmic parameters.

        This method varies the specified parameters using a five-point stencil approach to calculate the
        derivative signals based on a given input dataset.

        Parameters:
        - params: list of str
            A list of parameter names to vary.
        - d: float or list of float
            The data points or values related to the parameters being varied.
        - z: float
            The redshift value at which to evaluate the signals.
        - z0: list of float, optional
            The initial redshift(s). Default is [0.0].
        - multipole: list of str, optional
            The multipole(s) to consider for the signals. Default is ['monopole'].
        - which_comb: list of list of str, optional
            Combinations of bias parameters for the calculations. Default is [['b', 'b']].
        - e: float, optional
            The perturbation size for varying the parameters. Default is 0.005.
        - output_flat: bool, optional
            Flag indicating whether to return the output in a flat format. Default is False.

        Returns:
        - expr_out: numpy.ndarray
            The computed derivative values based on the five-point stencil method,
            either in flat format or structured depending on the output_flat parameter.
        """

        epsilon = [-2*e, -e, e, 2*e]

        final_list = []
        for eps in epsilon:
            dict_list = []
            for i in params:
                params_dict = {}
                for j in self.params_dict0:
                    if i == j:
                        params_dict[j] = self.params_dict0[j]*(1.0 + eps)
                    else:
                        params_dict[j] = self.params_dict0[j]
                dict_list.append(params_dict)
            final_list.append(dict_list)

        
        if self.dist_correction == False:
            d  =  [d] * len(epsilon)
        else:
            #h0=self.params_dict0['h']
            h0 = 1.0
            if params == ['h']:
            # Remove the h0 here to get back units of [Mpc/h_fid] that match with the units in the covariance
                d = [d/(h0*(1.-2*e)), d/(h0*(1.-e)), d/(h0*(1.+e)), d/(h0*(1.+2*e))]
            else:
                d = [d] * len(epsilon)
                
        signals=[]
        
        k=0
        for i in epsilon:
            for j in np.arange(0, len(params)):
                solver = Solver(z0, **final_list[k][j])
                signals.append(Signal(solver, n_split=self.n_split, delta=self.deltab, which_multipoles=multipole, return_lists=False, wide_angle = self.wide_angle))
            k+=1
        
        if multipole==['hexadecapole']:
            sig2eps = signals[3]._multipoles[multipole[0]](d[3], z)
            sig1eps = 8*signals[2]._multipoles[multipole[0]](d[2], z)
            sign1eps = 8*signals[1]._multipoles[multipole[0]](d[1], z)
            sign2eps = signals[0]._multipoles[multipole[0]](d[0], z)
        else:
            sig2eps = signals[3]._multipoles[multipole[0]](d[3], z, which_comb=which_comb)
            sig1eps = 8*signals[2]._multipoles[multipole[0]](d[2], z, which_comb=which_comb)
            sign1eps = 8*signals[1]._multipoles[multipole[0]](d[1], z, which_comb=which_comb)
            sign2eps = signals[0]._multipoles[multipole[0]](d[0], z, which_comb=which_comb)
            
        delta = e * self.params_dict0[params[0]]

        expr = (- sig2eps + sig1eps - sign1eps + sign2eps)/(12 * delta)
        
        if output_flat == True:
            if multipole[0]=='dipole' or multipole[0]=='hexadecapole' or multipole[0]=='octupole':
                expr_out = expr[0]
            else:
                expr_out = np.concatenate([expr[0], expr[1], expr[2]], axis=1)
        else:
            expr_out = expr

        return expr_out
    
    def dsignal_cosmic_params(self, z, d, multipoles=['monopole', 'dipole', 'quadrupole', 'hexadecapole', 'octupole'], steps=[1e-3, 1e-1, 1e-4, 1e-2, 1e-2]):
    
        """
        Calculate the derivatives of signal with respect to cosmic parameters using the five-point stencil method.

        This method computes the derivatives of the signal based on variations in cosmic parameters for 
        a given set of multipoles. The results are organized into a structured array reflecting the 
        sensitivity of the signals to changes in parameters.

        Parameters:
        - z: float or list of float
            The redshift values at which to evaluate the signals.
        - d: float or list of float
            The data points or values related to the parameters being varied.
        - multipoles: list of str, optional
            A list of multipole terms to consider for the derivatives. Default includes 
            ['monopole', 'dipole', 'quadrupole', 'hexadecapole', 'octupole'].
        - steps: list of float, optional
            A list of perturbation sizes corresponding to each parameter. Default is 
            [1e-3, 1e-1, 1e-4, 1e-2, 1e-2].

        Returns:
        - dsignal_dtheta: numpy.ndarray
            An array containing the computed derivatives of the signals with respect to the cosmic parameters,
            structured by parameter, redshift, and data point.
        """
     
        keys = self.params_dict0.keys()
                
        derivatives = [[self.five_pt_stencil(params=[param], d=d, z=z, multipole=[multipole], which_comb='default', e=step, output_flat=True) for multipole in multipoles] for (param, step) in zip(keys, steps)]
    
        dsignal_dtheta = np.empty([len(keys), len(z), int(len(d) * 9)])
    
        for k,_ in enumerate(z):
            for i, key in enumerate(keys):
                if key == 'As':
                    dsignal_dtheta[i,k] = self.params_dict0[key] * np.concatenate([derivatives[i][0][k], derivatives[i][1][k], derivatives[i][2][k], derivatives[i][3][k], derivatives[i][4][k]])
                else:
                    dsignal_dtheta[i,k] = np.concatenate([derivatives[i][0][k], derivatives[i][1][k], derivatives[i][2][k], derivatives[i][3][k], derivatives[i][4][k]])
        
        return dsignal_dtheta

    def derivative_magbias(self, pop, param, d, z, zin=10., absolute=False):
        """
        Calculate the derivatives of the signal with respect to magnification bias parameters.

        This method computes the derivatives of the signal related to magnification bias parameters 
        based on the specified population and parameter type. The results provide insights into 
        how variations in the magnification bias parameters affect the signal.

        Parameters:
        - pop: str
            The population type for which the derivatives are computed. Options include 'sB' 
            or other categories.
        - param: str
            The specific magnification bias parameter to vary (e.g., 's1', 's2', 's3', 's0'). These 
            correspond with the parameters of the fitting model.
        - d: float or list of float
            The data points or values related to the parameters being varied.
        - z: float or list of float
            The redshift values at which to evaluate the derivatives.
        - zin: float, optional
            The initial redshift value for the calculations. Default is 10.0.
        - absolute: bool, optional
            If True, the method returns absolute values; if False, it returns relative changes. 
            Default is False.

        Returns:
        - numpy.ndarray
            An array containing the computed derivatives of the signals with respect to the 
            magnification bias parameters, organized by multipole and parameter.
        """

        nz = len(z)
        nd = len(d)

        rH0=self.CosmoFfid.rH0(z) #comoving distance to z, multiplied by H_0
        Hz=self.CosmoFfid.HH(z)
        s8 = self.solverfid.Sigma8(zin)
        f = self.CosmoFfid.f(z) * s8(z)

        sigma80 = s8(0.0)
        
        nu1= d * self.H0 * self.nu1fid(d)
        nu3= d * self.H0 * self.nu3fid(d)

        gamma = 5 * (1/(rH0 * Hz)-1)

        zeros = [0.] * nd

        monoBB = [zeros] * nz
        monoBF = monoBB
        monoFF = monoBB

        quadBB = monoBB
        quadBF = monoBF
        quadFF = monoFF

        hexa = monoBB
        
        if param == 's1':
            fac = z
        elif param == 's2':
            fac = np.log(z)
        elif param == 's3':
            fac = (np.log(z))**2
        elif param == 's0':
            fac = 1.0
        
        if absolute == False:
            if pop == 'sB':
                dipfactor = Hz * ( 3/5 * gamma * f**2 + (self.galaxybias('f',z)*s8(z)) * gamma * f) * fac
                octfactor = Hz * ( 2/5 * gamma * f**2 ) * fac
            else:
                dipfactor = Hz * (-3/5 * gamma * f**2 - (self.galaxybias('b',z)*s8(z)) * gamma * f) * fac
                octfactor = Hz * (-2/5 * gamma * f**2 ) * fac
        else:
            if pop == 'sB':
                dfevol = - (gamma + 10. + (5/2)*self.EvolBias.dLogFcut(z)) 
                dipfactor = Hz * ( (self.n_split/(self.n_split-1)) * 3/5 * (gamma + dfevol) * f**2 + (self.galaxybias('f',z) + (1/(self.n_split-1))*self.galaxybias('b',z))*s8(z) * (gamma + dfevol) * f) * fac
                octfactor = Hz * ( (self.n_split/(self.n_split-1)) * 2/5 * (gamma + dfevol) * f**2 ) * fac
            else:
                dfevol = - (gamma + 10. + (5/2)*self.EvolBias.dLogFstar(z))
                dipfactor = Hz * (self.n_split/(self.n_split-1)) * (-3/5 * (gamma + dfevol) * f**2 - self.galaxybias('b',z)*s8(z) * (gamma + dfevol) * f) * fac
                octfactor = Hz * (self.n_split/(self.n_split-1)) * (-2/5 * (gamma + dfevol) * f**2 ) * fac 
        
        dip =  nu1/sigma80**2 * np.transpose(np.array([dipfactor,] * nd))
        oct = nu3/sigma80**2 * np.transpose(np.array([octfactor,] * nd))

        return np.concatenate([monoBB, monoBF, monoFF, dip, quadBB, quadBF, quadFF, hexa, oct], axis=1)
    
    
    def dsignal_magbias_fit(self, d, z, pops=['sB', 'sF'], names_s_params=['s0', 's1', 's2', 's3'], absolute=True):
        """
        Compute the magnification bias derivatives for fitting parameters across specified populations.

        This method calculates the derivatives of the signal related to magnification bias 
        for different fitting parameters across specified populations and redshift values. 
        The results are structured to assist in the fitting process by providing insights into 
        how variations in the magnification bias parameters influence the signal.

        Parameters:
        - d: float or list of float
            The data points or values related to the parameters being varied.
        - z: float or list of float
            The redshift values at which to evaluate the derivatives.
        - pops: list of str, optional
            A list of population types for which the derivatives are computed. Default is 
            ['sB', 'sF'].
        - names_s_params: list of str, optional
            A list of names for the magnification bias parameters to differentiate in the 
            derivative calculations. Default is ['s0', 's1', 's2', 's3'].
        - absolute: bool, optional
            If True, the method returns absolute values; if False, it returns relative changes. 
            Default is True.

        Returns:
        - numpy.ndarray
            A concatenated array containing the computed magnification bias derivatives for 
            the specified populations and parameters.
        """

        dsignal_ds = np.array([[self.derivative_magbias(pop, param, d, z, absolute=absolute) for param in names_s_params] for pop in pops])
        
        return np.concatenate([dsignal_ds[0], dsignal_ds[1]])
    
    
    def derivative_gbias_fit(self, pop, param, d, z, zin=10.):
        """
        Compute the derivatives of the galaxy bias signal for a specified population and parameter.

        This method calculates the derivatives of the galaxy bias signal with respect to the
        specified parameter for a given population. The results can be used to analyze how
        variations in galaxy bias parameters influence the signal, aiding in cosmological
        modeling and parameter fitting.

        Parameters:
        - pop: str
            The population type for which the derivative is calculated. Can be 'B' for bright
            galaxies or 'F' for faint galaxies.
        - param: str
            The parameter of the galaxy bias to differentiate, such as 'b1'.
        - d: float or list of float
            The data points or values related to the parameters being varied.
        - z: float or list of float
            The redshift values at which to evaluate the derivatives.
        - zin: float, optional
            The initial redshift value used for some calculations. Default is 10.

        Returns:
        - numpy.ndarray
            A concatenated array containing the computed derivatives of the galaxy bias signal 
            for the specified population and parameter.
        """


        nz = len(z)
        nd = len(d)

        rH0 = self.CosmoFfid.rH0(z) #comoving distance to z, multiplied by H_0
        Hz = self.CosmoFfid.HH(z)
        s8 = self.solverfid.Sigma8(zin)
        sigma8 = s8(z)
        f = self.CosmoFfid.f(z) * sigma8
        dHz = -(1+np.array(z)) * Hz * self.CosmoFfid.HH_dz(z)

        sigma80 = s8(0.0)

        nu1 = d * self.H0 * self.nu1fid(d)
        nu3 = d * self.H0 * self.nu3fid(d)
        mu0 = self.mu0fid(d)
        mu2 = self.mu2fid(d)
        mu4 = self.mu4fid(d)

        gamma = 5 * (1/(rH0 * Hz)-1)
        betaB = self.magbias('b', z) * gamma + self.fevol('b', z)
        betaF = self.magbias('f', z) * gamma + self.fevol('f', z)

        if pop == 'B':
            if param == 'b1':
                fac = np.exp(self.b2 * z)
            else:
                fac = z * self.b1 * np.exp(self.b2 * z)
            monoBBfactor = fac * (2 * self.galaxybias('b',z) * sigma8 + 2/3 * f) * sigma8 
            monoBFfactor = fac * (self.galaxybias('f',z) * sigma8 + f/3) * sigma8
            monoFFfactor = [0.] * nz
            dipfactor = Hz * fac * (-betaF * f + (2/(rH0 * Hz) + dHz/Hz**2) * f) * sigma8
            quadBBfactor = - fac * 4 * f * sigma8 / 3 
            quadBFfactor = - fac * 2 * f * sigma8 / 3 
            quadFFfactor = [0.] * nz
            hexafactor = [0.] * nz
            octfactor =  [0.] * nz 
        else:
            if param == 'b1':
                fac = np.exp(self.b2 * z)
            else:
                fac = z * self.b1 * np.exp(self.b2 * z)
            monoBBfactor = [0.] * nz
            monoBFfactor = fac * (self.galaxybias('b',z) * sigma8 + f/3) * sigma8
            monoFFfactor = fac * (2 * self.galaxybias('f',z) * sigma8 + 2/3 * f) * sigma8
            dipfactor = Hz * fac * (betaB * f - (2/(rH0 * Hz) + dHz/Hz**2) * f) * sigma8
            quadBBfactor = [0.] * nz
            quadBFfactor = - fac * 2 * f * sigma8 / 3 
            quadFFfactor = - fac * 4 * f * sigma8 / 3 
            hexafactor = [0.] * nz
            octfactor =  [0.] * nz            

        monoBB = mu0/sigma80**2 * np.transpose(np.array([monoBBfactor,] * nd))
        monoBF = mu0/sigma80**2 * np.transpose(np.array([monoBFfactor,] * nd))
        monoFF = mu0/sigma80**2 * np.transpose(np.array([monoFFfactor,] * nd))
        dip = nu1/sigma80**2 * np.transpose(np.array([dipfactor,] * nd))
        quadBB = mu2/sigma80**2 * np.transpose(np.array([quadBBfactor,] * nd))
        quadBF = mu2/sigma80**2 * np.transpose(np.array([quadBFfactor,] * nd))
        quadFF = mu2/sigma80**2 * np.transpose(np.array([quadFFfactor,] * nd))
        hexa = mu4/sigma80**2 * np.transpose(np.array([hexafactor,] * nd))
        oct = nu3/sigma80**2 * np.transpose(np.array([octfactor,] * nd))

        return np.concatenate([monoBB, monoBF, monoFF, dip, quadBB, quadBF, quadFF, hexa, oct], axis=1)
    

    def dsignal_gbias_fit(self, d, z, pops=['B', 'F'], params=['b1', 'b2']):
        """
        Compute the galaxy bias signal derivatives with respect to specified parameters for different populations.

        This method calculates the derivatives of the galaxy bias signal based on the provided 
        populations and parameters. It combines the results from the derivatives for different
        populations and parameters into a single array, facilitating analysis of the impact of
        galaxy bias parameters on the signal.

        Parameters:
        - d: float or list of float
            The data points or values related to the parameters being varied.
        - z: float or list of float
            The redshift values at which to evaluate the derivatives.
        - pops: list of str, optional
            The populations for which to compute the derivatives. Default is ['B', 'F'].
        - params: list of str, optional
            The parameters of the galaxy bias to differentiate. Default is ['b1', 'b2'].

        Returns:
        - numpy.ndarray
            A concatenated array containing the computed derivatives of the galaxy bias signal 
            for the specified populations and parameters.
        """
        
        dsignal_dgbias_fit = np.array([[self.derivative_gbias_fit(pop, param, d, z, zin=10.) for param in params] for pop in pops])
    
        return np.concatenate([dsignal_dgbias_fit[0], dsignal_dgbias_fit[1]])


    def derivative_number_fit(self, param, d, z, zin=10.):
        """
        Compute the derivatives of the number density signal with respect to specified parameters.

        This method calculates the derivatives of the number density signal based on the 
        specified parameter. It incorporates the effects of redshift and galaxy bias to
        generate the necessary derivatives for further analysis.

        Parameters:
        - param: str
            The parameter for which to compute the derivative. Options include 'n0', 'n1', 'n2', and 'n3'.
        - d: float or list of float
            The data points or values related to the parameters being varied.
        - z: float or list of float
            The redshift values at which to evaluate the derivatives.
        - zin: float, optional
            A reference redshift value, default is 10.

        Returns:
        - numpy.ndarray
            A concatenated array containing the computed derivatives of the number density signal 
            for the specified parameter across the given data points and redshift values.
        """

        nz = len(z)
        nd = len(d)

        Hz=self.CosmoFfid.HH(z)
        s8 = self.solverfid.Sigma8(zin)
        f = self.CosmoFfid.f(z) * s8(z)

        s80 = s8(0.0)

        nu1 = d * self.H0 * self.nu1fid(d)
        mu0= self.mu0fid(d)

        monoBBfactor = [0.] * nz
        monoBB = mu0/s80**2 * np.transpose(np.array([monoBBfactor,] * nd))
        
        if param == 'n1':
            fac = z
        elif param == 'n2':
            fac = 1/z
        elif param == 'n3':
            fac = np.exp(-z)
        elif param == 'n0':
            fac = 1.0

        dipfactor = Hz * (self.galaxybias('b', z) - self.galaxybias('f', z)) * s8(z) * f * fac

        dip = nu1/s80**2 * np.transpose(np.array([dipfactor,] * nd))
        oct = monoBB

        return np.concatenate([monoBB, monoBB, monoBB, dip, monoBB, monoBB, monoBB, monoBB, oct], axis=1)
    
    
    def dsignal_number_logfit(self, d, z, params=['n0', 'n1', 'n2', 'n3']):
        """
        Compute the derivatives of the number density signal for a set of specified parameters.

        This method evaluates the derivatives of the number density signal for a given set
        of parameters at specified data points and redshift values, using the `derivative_number_fit`
        method for calculations.

        Parameters:
        - d: float or list of float
            The data points or values related to the parameters being varied.
        - z: float or list of float
            The redshift values at which to evaluate the derivatives.
        - params: list of str, optional
            The list of parameters for which to compute the derivatives. Default is ['n0', 'n1', 'n2', 'n3'].

        Returns:
        - numpy.ndarray
            An array containing the computed derivatives of the number density signal for 
            the specified parameters across the given data points and redshift values.
        """
        
        dsignal_dnumber_fit = np.array([self.derivative_number_fit(param, d, z, zin=10.) for param in params])

        return dsignal_dnumber_fit


    def galaxybias(self, pop, x): 
        """
        Compute the galaxy bias for a given population.

        This method evaluates the galaxy bias based on the specified population type
        (bright or faint) at the provided data point.

        Parameters:
        - pop: str
            The type of galaxy population for which to compute the bias. 
            Use 'b' for bright galaxies and any other value for faint galaxies.
        - x: float or array-like
            The input data point(s) at which to evaluate the galaxy bias.

        Returns:
        - float or array-like
            The computed galaxy bias for the specified population type at the given input.
        """

        if pop=='b':
                  return  self.GalaxyBias.gbias_bright(x)
        else:
                  return  self.GalaxyBias.gbias_faint(x)  
    
    
    def magbias(self, pop, x): 
        """
        Compute the magnification bias for a given population.

        This method evaluates the magnification bias based on the specified population type
        (bright or faint) at the provided redshift values.

        Parameters:
        - pop: str
            The type of galaxy population for which to compute the magnification bias. 
            Use 'b' for bright galaxies and any other value for faint galaxies.
        - x: float or array-like
            The input redshift value(s) at which to evaluate the magnification bias.

        Returns:
        - float or array-like
            The computed magnification bias for the specified population type at the given input redshift(s).
        """

        z = np.array(x)
        if pop=='b': 
            return self.MagBias.s_bright(z) 
        else:  
            return self.MagBias.s_faint(z)
        
    
    def fevol(self, pop, x):
        """
        Compute the evolution bias for a given population.

        This method calculates the evolution bias for either bright or faint galaxy populations
        at the provided redshift values.

        Parameters:
        - pop: str
            The type of galaxy population for which to compute the evolution bias.
            Use 'b' for bright galaxies and any other value for faint galaxies.
        - x: float or array-like
            The input redshift value(s) at which to evaluate the evolution bias.

        Returns:
        - float or array-like
            The computed evolution bias for the specified population type at the given input redshift(s).
        """

        if pop=='b':
            return self.EvolBias.fevol_bright(x)
        else:
            return self.EvolBias.fevol_faint(x)
