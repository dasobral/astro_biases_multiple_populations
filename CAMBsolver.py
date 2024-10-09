#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Oct 2022

@author: dasobral

"""

import numpy as np
from scipy.interpolate import interp1d
import fftlog
import camb
from camb import model, initialpower 


class Solver(object):
    
    """
    Class to calculate cosmological power spectrum.

    Parameters
    ----------
    zs : list
        The list of redshifts at which to calculate the power spectrum.
    h : float, default=0.677
        The dimensionless Hubble parameter.
    Om : float, default=0.3111
        The density parameter for matter.
    Ob : float, default=0.0490
        The density parameter for baryons.
    As : float, default=2.05e-9
        The amplitude of the primordial power spectrum.
    ns : float, default=0.9665
        The spectral index of the primordial power spectrum.
    NonLin : boolean, default=False
        Whether to use non-linear spectra.
    silent : boolean, default=True
        Whether to run with silent mode or not.
    """

    def __init__(self, zs, h = 0.677, Om = 0.3111, Ob=0.0490, As = 2.05e-9, ns=0.9665, NonLin=False, silent=True):

        self.h=h
        self.H0 = 100 * self.h
        self.ombh2 = Ob * self.h**2
        self.omch2 = (Om-Ob) * self.h**2
        self.As = As
        self.ns = ns
        self.NonLin=NonLin
        self.silent=silent

        if isinstance(zs, list):
            self.zs = zs
        else:
            raise TypeError('CAMB redshifts input must be a list')

        self.params = camb.CAMBparams()
        self.params.set_cosmology(H0 = self.H0, ombh2 = self.ombh2, omch2 = self.omch2, mnu=0.0, omk=0.0, tau=0.09, num_massive_neutrinos=0.0, nnu=3.04)
        self.params.InitPower.set_params(As = self.As, ns = self.ns, r=0.0)
        self.params.set_for_lmax(2500, lens_potential_accuracy=0.0);
        self.params.set_matter_power(redshifts=self.zs, kmax=100.0, silent = self.silent)
        
        self.params.Transfer.high_precision = True
        self.params.Transfer.kmax = 100
        
    def NonLinMode(self):
        """
        Get whether to use non-linear mode.
        
        Returns
        -------
        boolean
            Whether to use non-linear spectra or not.
        """
        return self.NonLin

    def PowerSpectrum(self, minkh=1e-4, kmax=100, npoints = 200):
        
        """
        Get the linear/non-linear matter power spectrum in Mpc^3/h^3.
        
        Parameters
        ----------
        minkh : float, default=1e-4
            Minimum h/Mpc for desired power spectrum. 
        kmax : float, default=100
            Mainly to get accurate kBao.
        npoints : int, default=200
            The number of points in which to calculate the spectrum.
            
        Returns
        -------
        kh : numpy.ndarray
            The values of k*h in 1/Mpc.
        z_sol : numpy.ndarray
            The redshift array of the matter power spectrum.
        pk_out : numpy.ndarray
            The linear/non-linear matter power spectrum in Mpc^3/h^3.
        """

        if self.NonLinMode():
            #NonLinear spectra (Halofit)
            self.params.NonLinear = model.NonLinear_both
        else:
            #Linear spectra
            self.params.NonLinear = model.NonLinear_none
        
        self.results = camb.get_results(self.params)
        kh, z_sol, pk = self.results.get_matter_power_spectrum(minkh=minkh, maxkh=kmax, npoints=npoints)


        if len(z_sol) > 1:
            pk_out=[];
            for i, (redshift, line) in enumerate(zip(z_sol,['-','--'])):
                pk_out.append(pk[i,:])
        else:
            pk_out=pk

        return kh, z_sol, np.array(pk_out).flatten()

    def f_sigma8(self, z = np.linspace(0., 10., 100), kmax=100.0):
        
        """
        Computes the f_sigma8 function.

        Parameters
        ----------
        z : np.ndarray
        The redshift array.
        kmax : float, optional
        The maximum k value, by default 100.0

        Returns
        -------
        interp1d
        A cubic spline interpolation object of f_sigma8.
        """

        self.params.set_matter_power(redshifts = z, kmax=kmax, silent = True)
        self.results = camb.get_results(self.params)
        self.results.calc_power_spectra(params=self.params)
        f_sigma8_unsorted = self.results.get_fsigma8()
        f_sigma8_sorted = np.flip(f_sigma8_unsorted)
        f_sigma8 = interp1d(z, f_sigma8_sorted, kind='cubic')

        return f_sigma8

    def Sigma8(self, zin, kmax=100.0):
        
        """
        Computes the Sigma8 function.

        Parameters\n        ----------\n        zin : float
        The maximum redshift.
        kmax : float, optional
        The maximum k value, by default 100.0

        Returns
        -------
        interp1d
        A cubic spline interpolation object of Sigma8.
        """
        
        redshifts = np.linspace(0., zin, 100)

        self.params.set_matter_power(redshifts = redshifts, kmax=kmax, silent = True)
        self.results = camb.get_results(self.params)
        self.results.calc_power_spectra(params=self.params)
        sigma8_unsorted = self.results.get_sigma8()
        sigma8_sorted = np.flip(sigma8_unsorted)
        sigma8 = interp1d(redshifts, sigma8_sorted, kind='cubic', fill_value= 'extrapolate')

        if not self.silent:
            print('Redshifts order restored: latest first.')

        return sigma8

    def Transforms(self, benchmark, integrals = {0 : (0, 0), 1 : (2, 0), 2 : (4, 0), 3 : (1, 1), 4 : (3, 1)}, sampling_pt=2048):
        
        """
        Computes the FFTlog transform of the power spectrum.

        Parameters
        ----------
        benchmark : int
            The benchmark number to select the integral values, 0 to 4.
        integrals : Dict[int, Tuple[int, int]], optional
        The benchmark integral values, by default
            {0: (0, 0), 1: (2, 0), 2: (4, 0), 3: (1, 1), 4: (3, 1)}

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            The tuple contains the separations and FFTlog arrays.
        """
        
        data = self.PowerSpectrum()
        k = data[0]
        pk = data[2]

        l, n = integrals[benchmark]
        # input is k [array], pk [array], l [double or int], n [double or int], sampling points [int]
        result = fftlog.FFTlog(k, pk, l, n, sampling_pt)

        # this does the actual transform with the input being r_0
        # (in this case, since input k is in [h/Mpc], and input P(k) is in [Mpc/h]^3,
        # this needs to be in Mpc/h, i.e. here we set r_0 = 1 Mpc/h)
        # NOTE: results are stored in class members `x_fft` (the separations), and `y_fft` (the FFTlog)
        result.transform(1)

        # we don't care about results above 1000 Mpc/h
        criterion = result.x_fft <= 1000

        return result.x_fft[criterion], result.y_fft[criterion]

    def interpolate_multi(self):

        multi_dict = {
            'mu0' : self.interpolate_mu0(),
            'mu2' : self.interpolate_mu2(),
            'mu4' : self.interpolate_mu4(),
            'nu1' : self.interpolate_nu1(),
            'nu3' : self.interpolate_nu3()
        }

        #print( 'Returns a dictionary with the interpolated functions mu0, mu2, mu4 and nu1.' )

        return multi_dict

    def interpolate_mu0(self):

        data = self.Transforms(0)
        val_d = data[0]
        val_int = data[1]
        mu0 = interp1d(val_d, val_int, kind ='cubic', fill_value="extrapolate")

        return mu0

    def interpolate_nu1(self):

        #Note: This is in fact the function nu_1 WITHOUT the factor d * H0

        data = self.Transforms(3)
        val_d = data[0]
        val_int = data[1]
        nu1 = interp1d(val_d, val_int, kind ='cubic', fill_value="extrapolate")

        return nu1

    def interpolate_mu2(self):

        data = self.Transforms(1)
        val_d = data[0]
        val_int = data[1]
        mu2 = interp1d(val_d, val_int, kind ='cubic', fill_value="extrapolate")

        return mu2
    
    def interpolate_mu4(self):

        data = self.Transforms(2)
        val_d = data[0]
        val_int = data[1]
        mu4 = interp1d(val_d, val_int, kind ='cubic', fill_value="extrapolate")

        return mu4
    
    def interpolate_nu3(self):
        
        #Note: This is in fact the function nu_1 WITHOUT the factor d * H0

        data = self.Transforms(4)
        val_d = data[0]
        val_int = data[1]
        nu3 = interp1d(val_d, val_int, kind ='cubic', fill_value="extrapolate")
        
        return nu3
        



       

        



        




