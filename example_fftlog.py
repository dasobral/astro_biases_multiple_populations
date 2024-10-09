import numpy as np
import fftlog
import matplotlib.pyplot as plt
import camb
from camb import model, initialpower

# load a simple power spectrum (generated with CLASS)
# data = np.loadtxt('power_test.dat')

# Set up a new set of parameters for CAMB

def mnu_integrals(z):
    
    params = camb.CAMBparams()
    params.set_cosmology(H0 = 67.4, ombh2=0.0224, omch2= 0.120, mnu=0.00, omk=0.0, tau=0.054)
    params.InitPower.set_params(As=2.1e-9, ns=0.965, r=0.0)
    params.set_dark_energy(w=-1.0)
    # Calculate CAMB results for these parameter settings
    results = camb.get_results(params)

    #Now get matter power spectra and sigma8 at redshift 10
    #Note non-linear corrections couples to smaller scales than you want
    params.set_matter_power(redshifts=[z], kmax=100.0, silent=True)

    #Linear spectra
    params.NonLinear = model.NonLinear_none
    results = camb.get_results(params)
    kh, z, pk = results.get_matter_power_spectrum(minkh=1e-4, maxkh=3., npoints = 1000)
    np.savetxt('Pkz10.dat', np.array([kh, pk[0]]))

    k = kh
    pk = pk

    # map between the benchmark integral filenames and the pairs (l, n)
    integrals = {
        0 : (0, 0),
        1 : (2, 0),
        2 : (4, 0),
        3 : (1, 1),
        #4 : (3, 1),
        #5 : (0, 2),
        #6 : (2, 2),
        #7 : (1, 3),
    }

    filenames = {
        0: 'mu0',
        1: 'mu2',
        2: 'mu4',
        3: 'nu1'
        }

    for mult in [0, 1, 2, 3]:
    
        l, n = integrals[mult]

        # input is k [array], pk [array], l [double or int], n [double or int], sampling points [int]
        result = fftlog.FFTlog(k, pk, l, n, 2048)

        # this does the actual transform with the input being r_0
        # (in this case, since input k is in [h/Mpc], and input P(k) is in [Mpc/h]^3,
        # this needs to be in Mpc/h, i.e. here we set r_0 = 1 Mpc/h)
        # NOTE: results are stored in class members `x_fft` (the separations), and `y_fft` (the FFTlog)
        result.transform(1)

        # we don't care about results above 350 Mpc/h
        criterion = result.x_fft <= 350
        x = result.x_ftt[criterion]

        # remove elements in y_ftt
        idmax_crit = len(x)
        y = np.take(result.y_fft, range(idmax_crit))

        # Stack the columns
        sol = np.column_stack((x, y))
        
        filename = filenames[mult]+'_z'+str(z)+'_fid.dat'

