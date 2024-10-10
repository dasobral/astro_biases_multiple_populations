#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Feb 2023

@author: dasobral

"""
import numpy as np

def cov_matrix_survey(d, z_bins, pixel_size=4, small_off_diagonal=True, rtol=1e-4, split=[50, 50]):
    """
    Load the covariance matrix for a survey.

    Parameters:
    - d (array-like): The distance bins.
    - z_bins (array-like): The redshift bins.
    - pixel_size (float, optional): The pixel size. Default is 4.
    - small_off_diagonal (bool, optional): Whether to remove small off-diagonal terms. Default is True.
    - rtol (float, optional): The relative difference threshold for removing small off-diagonal terms. Default is 1e-4.
    - split (list, optional): The population split percentages. Default is [50, 50].

    Returns:
    - CovMatrix (list of arrays): The covariance matrices for each redshift bin.
    """

    d_min = d[0]
    d_max = d[-1]

    print('\n ... Loading Covariance Matrices [MONOBB, MONOBF, MONOFF, DIPBF, QUADBB, QUADBF, QUADFF, HEXAT, OCTBF] ... \n')
    print(f'Population splitted in {split[0]}% BRIGHT - {split[1]}% FAINT \n')

    CovMatrix = [np.loadtxt('Covariance/octupole/CovarianceMatrixOCT_at_z'+str(z)+'_d'+str(int(d_min))+'to160_msplit-'+str(split[0])+'B_'+str(split[1])+'F.txt') for z in z_bins]

    if d_max != 160:
        for z,_ in enumerate(z_bins):
            ld = len(d)
            ld_max = len(np.arange(d_min, 164, pixel_size))
            rows_to_remove = np.r_[ld:ld_max, (ld+ld_max):int(2*ld_max), (ld+int(2*ld_max)):int(3*ld_max), (ld+int(3*ld_max)):int(4*ld_max), (ld+int(4*ld_max)):int(5*ld_max), (ld+int(5*ld_max)):int(6*ld_max), (ld+int(6*ld_max)):int(7*ld_max), (ld+int(7*ld_max)):int(8*ld_max)]
            cols_to_remove = np.c_[ld:ld_max, (ld+ld_max):int(2*ld_max), (ld+int(2*ld_max)):int(3*ld_max), (ld+int(3*ld_max)):int(4*ld_max), (ld+int(4*ld_max)):int(5*ld_max), (ld+int(5*ld_max)):int(6*ld_max), (ld+int(6*ld_max)):int(7*ld_max), (ld+int(7*ld_max)):int(8*ld_max)]
            CovMatrix[z] = np.delete(CovMatrix[z], rows_to_remove, axis=0)
            CovMatrix[z] = np.delete(CovMatrix[z], cols_to_remove, axis=1)

    if small_off_diagonal == False:
        print('\n ... Removing small off-diagonal terms vs diagonal terms ... \n')
        print(f'\n Relative difference = {rtol} \n')
        for n, z in enumerate(z_bins):
            Matrix = np.copy(CovMatrix[n])
            r, c = np.shape(Matrix)
            for i in np.arange(r):
                for j in np.arange(c):
                    value = Matrix[i,j]/np.sqrt(Matrix[i,i] * Matrix[j,j])
                    if value <= rtol:
                        Matrix[i,j] = 0.
                    else: 
                        Matrix[i,j] = Matrix[i,j]
            CovMatrix[n] = Matrix

    print('\n ... Success! ... \n')

    return CovMatrix

def cov_matrix_joint(d, z_bins, small_off_diagonal=True, rtol=1e-4, splits=[50, 30], contributions='all'):
    """
    Load the covariance matrix for a joint analysis of two population splits.

    Parameters:
    - d (array-like): The distance bins.
    - z_bins (array-like): The redshift bins.
    - small_off_diagonal (bool, optional): Whether to remove small off-diagonal terms. Default is True.
    - rtol (float, optional): The relative difference threshold for removing small off-diagonal terms. Default is 1e-4.
    - splits (list, optional): The bright population split percentages. Default is [50, 30].
    - contributions (str, optional): The contributions to include in the covariance matrix. Default is 'all'.

    Returns:
    - CovMatrix (list of arrays): The covariance matrices for each redshift bin.
    """

    d_min = d[0]
    d_max = d[-1]

    print('\n ... Loading Covariance Matrices (Joint-splittings Analysis) [' + str(contributions) + '] [MONOBB, MONOBF, MONOFF, DIPBF, QUADBB, QUADBF, QUADFF, HEXAT, OCTBF] ... \n')

    CovMatrix = [np.loadtxt('Covariance/octupole/CovarianceMatrixOCT_' + str(contributions) + '_' + str(splits[0]) + 'x' + str(splits[1]) + '_at_z' + str(z) + '_d' + str(int(d_min)) + 'to160.txt') for z in z_bins]

    if small_off_diagonal == False:
        print('\n ... Removing small off-diagonal terms vs diagonal terms ... \n')
        print(f'\n Relative difference = {rtol} \n')
        for n, z in enumerate(z_bins):
            Matrix = np.copy(CovMatrix[n])
            r, c = np.shape(Matrix)
            for i in np.arange(r):
                for j in np.arange(c):
                    value = Matrix[i, j] / np.sqrt(Matrix[i, i] * Matrix[j, j])
                    if value <= rtol:
                        Matrix[i, j] = 0.
                    else:
                        Matrix[i, j] = Matrix[i, j]
            CovMatrix[n] = Matrix

    return CovMatrix

def inverse_cov_matrix_survey(d, z_bins, pixel_size=4, small_off_diagonal=True, rtol=1e-4, split=[50, 50]):
    """
    Compute the inverse covariance matrix for a survey.

    Parameters:
    - d (array-like): The distance bins.
    - z_bins (array-like): The redshift bins.
    - pixel_size (float, optional): The pixel size. Default is 4.
    - small_off_diagonal (bool, optional): Whether to remove small off-diagonal terms. Default is True.
    - rtol (float, optional): The relative difference threshold for removing small off-diagonal terms. Default is 1e-4.
    - split (list, optional): The population split percentages. Default is [50, 50].

    Returns:
    - InvCovMatrix (list of arrays): The inverse covariance matrices for each redshift bin.
    """
    
    d_min = d[0]
    d_max = d[-1]
    
    print('\n ... Loading Covariance Matrices [MONOBB, MONOBF, MONOFF, DIPBF, QUADBB, QUADBF, QUADFF, HEXAT, OCTBF] ... \n')
    print(f'Population splitted in {split[0]}% BRIGHT - {split[1]}% FAINT \n')
    
    CovMatrix = [np.loadtxt('Covariance/octupole/CovarianceMatrixOCT_at_z'+str(z)+'_d'+str(int(d_min))+'to160_msplit-'+str(split[0])+'B_'+str(split[1])+'F.txt') for z in z_bins]
    
    if d_max != 160:
        for z,_ in enumerate(z_bins):
            ld = len(d)
            ld_max = len(np.arange(d_min, 164, pixel_size))
            rows_to_remove = np.r_[ld:ld_max, (ld+ld_max):int(2*ld_max), (ld+int(2*ld_max)):int(3*ld_max), (ld+int(3*ld_max)):int(4*ld_max), (ld+int(4*ld_max)):int(5*ld_max), (ld+int(5*ld_max)):int(6*ld_max), (ld+int(6*ld_max)):int(7*ld_max), (ld+int(7*ld_max)):int(8*ld_max)]
            cols_to_remove = np.c_[ld:ld_max, (ld+ld_max):int(2*ld_max), (ld+int(2*ld_max)):int(3*ld_max), (ld+int(3*ld_max)):int(4*ld_max), (ld+int(4*ld_max)):int(5*ld_max), (ld+int(5*ld_max)):int(6*ld_max), (ld+int(6*ld_max)):int(7*ld_max), (ld+int(7*ld_max)):int(8*ld_max)]
            CovMatrix[z] = np.delete(CovMatrix[z], rows_to_remove, axis=0)
            CovMatrix[z] = np.delete(CovMatrix[z], cols_to_remove, axis=1)
            
    if small_off_diagonal == False:
        print('\n ... Removing small off-diagonal terms vs diagonal terms ... \n')
        print(f'\n Relative difference = {rtol} \n')
        for n, z in enumerate(z_bins):
            Matrix = np.copy(CovMatrix[n])
            r, c = np.shape(Matrix)
            for i in np.arange(r):
                for j in np.arange(c):
                    value = Matrix[i,j]/np.sqrt(Matrix[i,i] * Matrix[j,j])
                    if value <= rtol:
                        Matrix[i,j] = 0.
                    else: 
                        Matrix[i,j] = Matrix[i,j]
            CovMatrix[n] = Matrix
            
    print('\n ... Computing the Inverse [MONOBB, MONOBF, MONOFF, DIPBF, QUADBB, QUADBF, QUADFF, HEXAT, OCTBF] ... \n')
            
    InvCovMatrix = [np.linalg.inv(CovMatrix[i]) for i,_ in enumerate(z_bins)]
    
    print('---------------------------------------- DONE. ---------------------------------------- \n')
    
    return InvCovMatrix

def inverse_cov_matrix_joint(d, z_bins, pixel_size=4, small_off_diagonal=True, rtol=1e-4, splits=[50, 30], contributions='all'):
    """
    Compute the inverse covariance matrix for a joint analysis of two population splits.

    Parameters:
    - d (array-like): The distance bins.
    - z_bins (array-like): The redshift bins.
    - pixel_size (float, optional): The pixel size. Default is 4.
    - small_off_diagonal (bool, optional): Whether to remove small off-diagonal terms. Default is True.
    - rtol (float, optional): The relative difference threshold for removing small off-diagonal terms. Default is 1e-4.
    - splits (list, optional): The bright population split percentages. Default is [50, 30].
    - contributions (str, optional): The contributions to include in the covariance matrix. Default is 'all'.

    Returns:
    - InvCovMatrix (list of arrays): The inverse covariance matrices for each redshift bin.
    """

    d_min = d[0]
    d_max = d[-1]

    print('\n ... Loading Covariance Matrices (Joint-splittings Analysis) [' + str(contributions) + '] [MONOBB, MONOBF, MONOFF, DIPBF, QUADBB, QUADBF, QUADFF, HEXAT, OCTBF] ... \n')

    CovMatrix = [np.loadtxt('Covariance/octupole/CovarianceMatrixOCT_' + str(contributions) + '_' + str(splits[0]) + 'x' + str(splits[1]) + '_at_z' + str(z) + '_d' + str(int(d_min)) + 'to160.txt') for z in z_bins]

    if small_off_diagonal == False:
        print('\n ... Removing small off-diagonal terms vs diagonal terms ... \n')
        print(f'\n Relative difference = {rtol} \n')
        for n, z in enumerate(z_bins):
            Matrix = np.copy(CovMatrix[n])
            r, c = np.shape(Matrix)
            for i in np.arange(r):
                for j in np.arange(c):
                    value = Matrix[i, j] / np.sqrt(Matrix[i, i] * Matrix[j, j])
                    if value <= rtol:
                        Matrix[i, j] = 0.
                    else:
                        Matrix[i, j] = Matrix[i, j]
            CovMatrix[n] = Matrix

    print('\n ... Computing the Inverse [MONOBB, MONOBF, MONOFF, DIPBF, QUADBB, QUADBF, QUADFF, HEXAT, OCTBF] ... \n')

    InvCovMatrix = [np.linalg.pinv(CovMatrix[i]) for i, _ in enumerate(z_bins)]

    print('\n ... Success! ... \n')

    return InvCovMatrix
        
def fisher_matrix(d_signal, inv_cov, z_bins, bins=False):
    """
    Compute the Fisher matrix for a given signal and inverse covariance matrix.

    Parameters:
    - d_signal (array-like): The signal as a function of redshift and distance.
    - inv_cov (list of arrays): The inverse covariance matrices for each redshift bin.
    - z_bins (array-like): The redshift bins.
    - bins (bool, optional): Whether to return the Fisher matrix for each redshift bin. Default is False.

    Returns:
    - FisherMatrix (array): The Fisher matrix.
    - FisherMatrixz (array, optional): The Fisher matrix for each redshift bin if bins=True.
    """

    print('\n ... Computing the Fisher Matrix ... \n')
    print('Signal & Cov must be given as functions (z,d) and (z,d,d\') respectively. \n')

    fisher_dim = len(d_signal)
    n_zs = len(z_bins)

    FisherMatrixz = np.empty((n_zs, fisher_dim, fisher_dim))

    for i in np.arange(fisher_dim):
        for j in np.arange(fisher_dim):
            for k in np.arange(len(z_bins)):
                FisherMatrixz[k, i, j] = d_signal[i, k] @ inv_cov[k] @ d_signal[j, k]

    FisherMatrix = np.sum(FisherMatrixz, axis=0)

    print('\n ... Success! ... \n')

    if bins == False:
        return FisherMatrix
    else:
        return FisherMatrixz
    
def signal_to_noise(signal, cov, verbose=1):
    """
    Compute the signal-to-noise ratio (S/N) for a given signal and covariance matrix.

    Parameters:
    - signal (array-like): The signal as a function of redshift and distance.
    - cov (list of arrays): The covariance matrices for each redshift bin.
    - verbose (int, optional): Whether to print the individual SNRs and cumulative SNR. Default is 1.

    Returns:
    - snr (float): The cumulative signal-to-noise ratio.
    - individual_snrs (array): The individual signal-to-noise ratios for each redshift bin.
    """

    print('\n ... Computing the S/N ... \n')
    print('Signal & Cov must be given as functions (z,d) and (z,d,d\') respectively. \n')
    print('-'*50)

    individual_snrs = []
    cumulative_snr = 0
            
    for i in range(len(signal)):
        S = signal[i]
        C = cov[i]

        # Calculate the SNR for the current element
        C_inv = np.linalg.inv(C)
        element_snr = np.sqrt(np.dot(S, np.dot(C_inv, S)))
        individual_snrs.append(element_snr)

    # Compute the cumulative SNR
    cumulative_snr = np.sum(np.square(individual_snrs))
        
    if verbose == 1:
        # Print individual SNRs
        print("\n Individual SNRs: \n")
        for i, snr in enumerate(individual_snrs):
            print(f"Bin {i + 1}: {snr}")

        print()
        print('-'*50)
        # Print cumulative SNR
        print("\n Cumulative SNR :", np.sqrt(cumulative_snr), '\n')
        print('-'*50)
        
    return np.sqrt(cumulative_snr), np.array(individual_snrs)