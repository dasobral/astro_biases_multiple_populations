# Astrophysical Biases of Galaxy Populations

This is a Python (+ some Mathematica) code used to constrain magnification and evolution biases using the multipoles of the galaxy 2-point correlation function in the flat-sky approximation. We make use of the three even multipoles and two odd multipoles containing all the information available. This code includes for the first time the modeling both of the dipole and the octupole as predicted by General Relativity, and exploits them simultaneously to put constraints on the magnification and the evolution biases, using solely information from the large-scale distribution of galaxies. 

This code was designed by [D. Sobral-Blanco](https://github.com/dasobral) to provide the numerical support to the paper [Using relativistic effects in large-scale structure to constrain astrophysical properties of galaxy populations](https://arxiv.org/abs/2406.19908). Any comment or inquiry should be adressed to [D. Sobral-Blanco](https://github.com/dasobral).

# Code Description

The code is organized as follows:

    - The Python sector:

    ```
    biasmodels.py 
        Modeling of the astrophysical biases for Bright and Faint galaxy populations: galaxy bias, magnification bias and evolution bias

    CAMBsolver.py 
        Logic for computing the Power Spectrum and its Fast Fourier Transform for the different multipoles. We use the Python extension of [CAMB](https://github.com/cmbant/CAMB)
        and this version of the [FFTLOG](https://github.com/JCGoran/fftlog-python) algorithm (also included in the repository as fftlog.py).

    compute_derivatives.py
        Script for computing the all the derivatives with respect to the parameters involved in the analysis. It stores the results as a dictionary in 'Results/derivatives.pkl'.

    config_file.ipynb
        Notebook for easy manipulation of the input parameters for the compute_derivatives.py module. The parameters are stored in 'Data/config_derivatives.pkl'. 

    cosmofuncs.py
        Set of utils to compute cosmological quantities.

    fishermat.py
        Set of utils to load the covariance matrices and compute the inverse and fisher matrices. It also contains an additional util to study the SNR.

    multipole_signal.py
        Module for computing the multipoles signals and their derivatives with respect to the parameters involved in the analysis. The derivatives with respect to the cosmic parameters are
        computed numerically. The rest of the derivatives can be computed analytically.
    ```

# Citation

If using this software, please cite this repository and the paper "Using relativistic effects in large-scale structure to constrain astrophysical properties of galaxy populations" [ 	arXiv:2406.19908](https://arxiv.org/abs/2406.19908). Bibtex:

```
@article{Sobral-Blanco:2024qlb,
    author = "Sobral-Blanco, Daniel and Bonvin, Camille and Clarkson, Chris and Maartens, Roy",
    title = "{Using relativistic effects in large-scale structure to constrain astrophysical properties of galaxy populations}",
    eprint = "2406.19908",
    archivePrefix = "arXiv",
    primaryClass = "astro-ph.CO",
    month = "6",
    year = "2024"
}
```
