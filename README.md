# Astrophysical Biases of Galaxy Populations

This is a Python (+ some Mathematica) code used to constrain magnification and evolution biases using the multipoles of the galaxy 2-point correlation function in the flat-sky approximation. We make use of the three even multipoles and two odd multipoles containing all the information available. This code includes for the first time the modeling both of the dipole and the octupole as predicted by General Relativity and a new modeling for splitting the galaxy population in two luminosity classes, bright (B) and faint (F). We then show that we can constrain the magnification and the evolution biases solely using information from the large-scale distribution of galaxies. 

This code was designed by [D. Sobral-Blanco](https://github.com/dasobral) to provide the numerical support to the paper [Using relativistic effects in large-scale structure to constrain astrophysical properties of galaxy populations](https://arxiv.org/abs/2406.19908). Any comment or inquiry should be adressed to [D. Sobral-Blanco](https://github.com/dasobral).

# Requirements

The necessary depencencies can be easily installed via the ```requirements.txt``` file. For clarity, we will need:

```python
camb==1.5.8
numpy==1.23.1
pandas==1.4.3
scipy==1.8.1
matplotlib
getdist
```

The only non-standard package is ```camb```. This is just the Python version of [CAMB](https://github.com/cmbant/CAMB) (Code for Anisotropies in the Microwave Background). For more specific information about this module, you can visit the official [documentation](https://camb.readthedocs.io/en/latest/).

If you need to install the specific packages in your machine or preferred environment, run the following command:

``` 
    pip install -r requirements.txt
```

# Code Description

The code is organized as follows:

- The Python sector. This is the core of the code, computes everything except for the covariance matrices. Results can be stored y the '/Results' folder. 
As examples of code usage, we also provide some of the Python notebooks generated for the paper in the folder '/Notebooks':

    ```
    biasmodels.py 
        Modeling of the astrophysical biases for Bright and Faint galaxy 
        populations: galaxy bias, magnification bias and evolution bias

    CAMBsolver.py 
        Logic for computing the Power Spectrum and its Fast Fourier Transform 
        for the different multipoles. We use the Python extension of 
        [CAMB](https://github.com/cmbant/CAMB)
        and this version of the [FFTLOG](https://github.com/JCGoran/fftlog-python) 
        algorithm (also included in the repository as fftlog.py).

    compute_derivatives.py
        Script for computing the all the derivatives with respect to the 
        parameters involved in the analysis. 
        It stores the results as a dictionary in 'Results/derivatives.pkl'.

    config_file.ipynb
        Notebook for easy manipulation of the input parameters for the 
        compute_derivatives.py module. 
        The parameters are stored in 'Data/config_derivatives.pkl'. 

    cosmofuncs.py
        Set of utils to compute cosmological quantities.

    fishermat.py
        Set of utils to load the covariance matrices and compute 
        the inverse and fisher matrices. 
        It also contains an additional util to study the SNR.

    multipole_signal.py
        Module for computing the multipoles signals and their derivatives 
        with respect to the parameters involved in the analysis. 
        The derivatives with respect to the cosmic parameters are
        computed numerically. 
        The rest of the derivatives can be computed analytically.
    ```

- The Mathematica sector. It is self-contained in the folder '/Covariance'. The inner folder '/Covariance/integrals' contains .dat files with the 2D FFT transforms
needed to compute the real-space covariances. Note that these have to be obtained from elsewhere, using a suitable method. We do not include the module used to compute
these integrals. The output of the Notebooks are stored in the folders '/Covariance/multi_split' (even multipoles + dipole) and '/Covariance/octupole' (even multipoles + dipole + octupole).

    ```
    sigma8_CAMB.dat
        File containing the values for $\sigma8$ as computed by CAMB.

    CovarianceCalculator_BxF.nb
        Notebooks for computing the Covariance Matrices for different 
        population splits, denoted by the % of B and F galaxies. We
        include 6 examples. The output the results to 

    CovarianceCalculator_Joint_Bxb.nb
        Notebooks for computing the Covariance Matrices for the 
        joint analysis of two splits, denoted by the % of Bright galaxies
        of each split (B and b, respectively).
    ```

# Citation

If using this software, please cite this repository and our paper: ["Using relativistic effects in large-scale structure to constrain astrophysical properties of galaxy populations"](https://arxiv.org/abs/2406.19908). Bibtex:

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
