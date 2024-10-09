#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Created on Dic 2023

@author: dasobral

"""

import numpy as np
import pickle as pk
import pprint
from multipole_signal import Derivatives
import time

# Record start time

start_time = time.time()
start_cpu_time = time.process_time()

# ------------------------- OPEN CONFIG -------------------------------------

config_file_path = 'config_derivatives.pkl'

try:
    with open(config_file_path, 'rb') as config_file:
        config = pk.load(config_file)
except FileNotFoundError:
    print(f"Error: Configuration file '{config_file_path}' not found. Please check the file path.")
    # Add additional error handling or exit the script as needed.
    exit()
    
print('\n'+'-'*10+' Configuartion settings '+'-'*10+'\n')
pp = pprint.PrettyPrinter(indent=1, width=1, sort_dicts=True)
pp.pprint(config)

print('\n'+'-' * 100)

# ------------------------- FUNCTIONS ---------------------------------------

def signals_survey(n_split = 2.0, delta=1.0, wide_angle=False, dist_correction = False):
    
    print('\n ... Computing derivatives of the Signal ... \n')
    print('\n'+f' Population Split: m = {n_split}')
    print(f' Galaxy bias difference, delta_b = {delta}')
    print('\n'+f"Derivatives have shapes : (num_params, num_bins, 9 * num_separations)"+'\n')
    
    deriv_class = Derivatives(params_dict0=config['params_dict0'], n_split=n_split, delta=delta, wide_angle=wide_angle, dist_correction=dist_correction)
    
    dev_dict = {
        
        'dsignal_dsbias' : deriv_class.dsignal_magbias_fit(d = config['dist'], z = config['z_bins'], absolute = True),
        'dsignal_dnumev' : deriv_class.dsignal_number_logfit(d = config['dist'], z = config['z_bins']),
        'dsignal_dgbias' : deriv_class.dsignal_gbias_fit(d = config['dist'], z = config['z_bins']),
        'dsignal_dcosmic': deriv_class.dsignal_cosmic_params(d = config['dist'], z = config['z_bins'], steps = config['steps'])
    
        }
    
    print('\n Derivatives are stored in a Dictionary with keys: \n')
    for key in dev_dict.keys():
        print(key)
    
    print('\n ... Sucess! ... \n')
    print('-' * 20)
    return dev_dict
    
# ------------------------- COMPUTATION -------------------------------------

deriv_signal_50x50 = signals_survey(n_split=2.)
deriv_signal_30x70 = signals_survey(n_split=10./3., delta=1.2)
deriv_signal_70x30 = signals_survey(n_split=10./7., delta=0.8)

# Other splits
deriv_signal_40x60 = signals_survey(n_split=5./2., delta=1.1)
deriv_signal_10x90 = signals_survey(n_split=10., delta=1.4)
deriv_signal_20x80 = signals_survey(n_split=5., delta=1.3)


# Record end time
end_time = time.time()
end_cpu_time = time.process_time()

deriv_dict = {
    'split_50x50' : deriv_signal_50x50,
    'split_30x70' : deriv_signal_30x70,
    'split_70x30' : deriv_signal_70x30,
    'split_10x90' : deriv_signal_10x90,
    'split_20x80' : deriv_signal_20x80,
    'split_40x60' : deriv_signal_40x60
}

# Calculate elapsed time
elapsed_time = end_time - start_time
elapsed_cpu_time = end_cpu_time - start_cpu_time

print('\n Saving the Dictionary to a file: derivatives.pkl \n')
for key in deriv_dict.keys():
        print(key)

with open('derivatives.pkl', 'wb') as pk_file:
    pk.dump(deriv_dict, pk_file)
    
print(f'\n Total elapsed real time: {elapsed_time:.2f} seconds')
print(f' Total elapsed CPU time: {elapsed_cpu_time:.2f} seconds')
    
print('\n ... Sucess! ...\n')
print('\n'+'-' * 100)
